"""Workstream 2, Phase 2 (GPU, cluster-only): unfreeze the Swin backbone on d.

Trains the BTCV SwinUNETR backbone -- via LoRA adapters OR full fine-tune -- plus
a ContrastiveMLP_v2 head, on the CONTRASTIVE split's d_mod3 (loss geometry chosen
by --loss_mode: euclidean / rank_kendall_basic / hybrid_basic / regress_d). It
then exports the ADAPTED 768-d backbone embeddings for ALL images across the three
splits, in the same on-disk layout as the frozen encoder
(`swin_latent.npy` + `image_id_order.npy`).

Downstream stays identical to Phase 1: point the frozen CPU drivers
(run_w0_phase0_dvalue.py / run_w0_phase1_diagnosis.py / run_w0_conversion_3way.py)
at this --out_dir with `--versions raw`. Because the downstream pipeline is byte-
for-byte the same as on the original frozen features, any change is attributable to
the backbone having adapted -- the clean "adapted-768 vs original-768" comparison.

CANNOT be validated off-cluster (needs GPU + raw NIfTI volumes + the BTCV
swinunetr package). The companion sbatch runs a forward/backward smoke + prints the
module tree BEFORE the full job so wrong LoRA targets fail loudly on the first run.

Compute contract (see plan.md 8.8): physical batch 8 + gradient accumulation to an
effective batch (default 16 -> 128), gradient checkpointing on. NOTE (plan 8.7 pt5):
accumulation does NOT enlarge the in-batch pair set for the pairwise geometries --
only regress_d is exact under accumulation.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

BTCV_PATH = "/dcs07/zwang/data/pmrc/SwinUNETR/BTCV"
if BTCV_PATH not in sys.path:
    sys.path.append(BTCV_PATH)

from minimal_v0_contrastive import (
    ContrastiveMLP_v2,
    y_aware_euclidean_loss,
    kendall_loss_basic,
    soft_kendall_loss,
)

IMAGE_COL_CANDIDATES = ("Image_Data_ID", "Image Data ID")
SPLITS = ("contrastive", "finetune", "test")


# --------------------------------------------------------------------------- #
# volume preprocessing (identical to scripts/encode_swin_smri.py:load_image)
# --------------------------------------------------------------------------- #
def load_volume(path, size):
    import nibabel as nib
    arr = np.asarray(nib.load(path).get_fdata(dtype=np.float32), dtype=np.float32)
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, 0.0)
    mask = finite & (np.abs(arr) > 1e-6)
    if int(mask.sum()) < 1000:
        mask = finite
    vals = arr[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError(f"near-zero intensity std: {path}")
    arr = (arr - mean) / std
    arr[~mask] = 0.0
    x = torch.from_numpy(arr[None, None])              # (1,1,X,Y,Z)
    if tuple(arr.shape) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
    return x[0]                                         # (1,X,Y,Z)


class VolumeDataset(Dataset):
    def __init__(self, frame, size):
        self.paths = frame["sMRI_path"].tolist()
        self.d = frame["d_mod3"].to_numpy(dtype=np.float32)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_volume(self.paths[i], self.size), self.d[i]


def image_col(df):
    for c in IMAGE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("no image-id column")


def split_frame(split_dir, d_csv, split, d_lookup):
    """One row per image for a split: sMRI_path + d_mod3 (from the D table)."""
    csv = Path(split_dir) / f"matched_{split.upper()}.csv"
    df = pd.read_csv(csv)
    ic = image_col(df)
    df = (df.dropna(subset=[ic, "sMRI_path"])
            .sort_values([ic, "sMRI_path"]).drop_duplicates(ic).reset_index(drop=True))
    df["Image_Data_ID"] = df[ic].astype(str)
    df["d_mod3"] = df["Image_Data_ID"].map(d_lookup).astype(float)
    return df[["Image_Data_ID", "sMRI_path", "d_mod3"]]


# --------------------------------------------------------------------------- #
# LoRA (self-contained; no peft dependency)
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))


def inject_lora(module, target_re, r, alpha, replaced):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and target_re.search(name):
            setattr(module, name, LoRALinear(child, r, alpha))
            replaced.append(name)
        else:
            inject_lora(child, target_re, r, alpha, replaced)


# --------------------------------------------------------------------------- #
# encoder = Swin backbone -> global-mean-pooled 768 -> ContrastiveMLP_v2 head
# --------------------------------------------------------------------------- #
class SwinEncoderWithHead(nn.Module):
    def __init__(self, backbone, hidden=256, latent=128, use_ckpt=True):
        super().__init__()
        self.backbone = backbone
        self.head = ContrastiveMLP_v2(in_dim=768, hidden=hidden, latent=latent)
        self.use_ckpt = use_ckpt

    def pool(self, x):
        feats = self.backbone(x)
        return feats[-1].mean(dim=(2, 3, 4))           # (B,768)

    def forward(self, x):
        pooled = (checkpoint(self.pool, x, use_reentrant=False)
                  if self.use_ckpt else self.pool(x))
        return self.head.score(pooled)                 # (z, s)

    @torch.no_grad()
    def embed(self, x):
        self.eval()
        return self.pool(x)


def build_backbone(model_id, device):
    from swinunetr import SwinUnetrModelForInference
    model = SwinUnetrModelForInference.from_pretrained(model_id, local_files_only=True)
    return model.model.swinViT.to(device), model.config


def compute_loss(z, s, d, args):
    if args.loss_mode == "euclidean":
        return y_aware_euclidean_loss(z, d, tau=args.tau, beta=args.beta)
    if args.loss_mode in ("rank_kendall_basic",):
        return kendall_loss_basic(s, d, nu=args.rank_nu)
    if args.loss_mode == "rank_kendall":
        return soft_kendall_loss(s, d, alpha=args.rank_alpha)
    if args.loss_mode == "hybrid_basic":
        return (y_aware_euclidean_loss(z, d, tau=args.tau, beta=args.beta)
                + args.lambda_rank * kendall_loss_basic(s, d, nu=args.rank_nu))
    if args.loss_mode == "regress_d":
        return F.mse_loss(s, d)
    raise ValueError(f"Unknown loss_mode: {args.loss_mode}")


def train(model, loader, args, device):
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train:,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = pending = 0.0
        n_micro = 0
        for step, (vol, d) in enumerate(loader):
            vol = vol.to(device, non_blocking=True).contiguous()
            d = d.to(device, non_blocking=True)
            if vol.shape[0] < 2:                       # BatchNorm in the head
                continue
            z, s = model(vol)
            loss = compute_loss(z, s, d, args) / args.accum_steps
            loss.backward()
            running += loss.item() * args.accum_steps
            pending += 1
            n_micro += 1
            if pending == args.accum_steps:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); pending = 0
        if pending > 0:                                # flush leftover micro-batches
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"  [{args.adapt}/{args.loss_mode}] epoch {epoch+1}/{args.epochs} "
                  f"loss={running / max(n_micro, 1):.4f}", flush=True)


@torch.no_grad()
def export_embeddings(model, frames, args, device, out_dir):
    """Encode every unique image across all splits -> adapted 768-d features."""
    all_imgs = pd.concat(frames.values()).drop_duplicates("Image_Data_ID").reset_index(drop=True)
    feats = np.zeros((len(all_imgs), 768), dtype=np.float32)
    for i, row in all_imgs.iterrows():
        vol = load_volume(row["sMRI_path"], args.img_size).unsqueeze(0).to(device).contiguous()
        feats[i] = model.embed(vol).squeeze(0).float().cpu().numpy()
        if (i + 1) % args.print_every == 0 or (i + 1) == len(all_imgs):
            print(f"  exported {i+1}/{len(all_imgs)}", flush=True)
    if not np.isfinite(feats).all():
        raise SystemExit("STOP: exported adapted features contain NaN/Inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "swin_latent.npy", feats)
    np.save(out_dir / "image_id_order.npy", all_imgs["Image_Data_ID"].to_numpy())
    print(f"Wrote adapted embeddings {feats.shape} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2way", "3way"], default="3way",
                    help="2way: pretrain on train, export train+test")
    ap.add_argument("--split_dir", default="../data/splits_3way_20260627_v2")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016",
                    help="2way: dir with matched_TRAIN.csv / matched_TEST.csv")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--model_id", default="darragh/swinunetr-btcv-base")
    ap.add_argument("--img_size", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--adapt", choices=["lora", "full"], default="lora")
    ap.add_argument("--loss_mode", default="regress_d",
                    choices=["euclidean", "rank_kendall", "rank_kendall_basic",
                             "hybrid_basic", "regress_d"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)      # physical micro-batch
    ap.add_argument("--accum_steps", type=int, default=16)    # -> effective 128
    ap.add_argument("--lr", type=float, default=None)         # default set by --adapt
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_targets", default=r"(qkv|proj|fc1|fc2)$",
                    help="regex over immediate Linear attribute names in swinViT")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rank_alpha", type=float, default=10.0)
    ap.add_argument("--rank_nu", type=float, default=None)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--no_checkpoint", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap contrastive images")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--print_modules", action="store_true", help="print swinViT tree and exit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 5e-4 if args.adapt == "lora" else 1e-4

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} adapt={args.adapt} loss_mode={args.loss_mode} "
          f"batch={args.batch_size}x{args.accum_steps} (eff {args.batch_size*args.accum_steps})")

    backbone, config = build_backbone(args.model_id, device)
    print(f"loaded {args.model_id}; img_size={getattr(config,'img_size',None)}")

    if args.print_modules:
        for n, m in backbone.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  Linear {n}  ({m.in_features}->{m.out_features})")
        return

    # adaptation mode
    if args.adapt == "lora":
        for p in backbone.parameters():
            p.requires_grad_(False)
        replaced = []
        inject_lora(backbone, re.compile(args.lora_targets), args.lora_rank,
                    args.lora_alpha, replaced)
        print(f"LoRA injected into {len(replaced)} Linear modules "
              f"(targets={args.lora_targets}); e.g. {replaced[:6]}")
        if not replaced:
            raise SystemExit("STOP: LoRA target regex matched 0 modules; run "
                             "--print_modules to see the real attribute names.")
    else:
        for p in backbone.parameters():
            p.requires_grad_(True)
        print("full fine-tune: all backbone params trainable")

    model = SwinEncoderWithHead(backbone, hidden=args.hidden, latent=args.latent,
                                use_ckpt=not args.no_checkpoint).to(device)

    # d lookup from the authoritative D table (same source as load_features_3way)
    dtab = pd.read_csv(args.d_csv)
    ic = image_col(dtab)
    d_lookup = dict(zip(dtab[ic].astype(str), dtab["d_mod3"].astype(float)))

    if args.mode == "2way":
        split_names, src_dir, train_split = ("train", "test"), args.master_dir, "train"
    else:
        split_names, src_dir, train_split = SPLITS, args.split_dir, "contrastive"
    frames = {s: split_frame(src_dir, args.d_csv, s, d_lookup) for s in split_names}
    for s, fr in frames.items():
        print(f"  {s:11s}: {len(fr)} images, d valid {int(np.isfinite(fr['d_mod3']).sum())}")

    train_frame = frames[train_split]
    train_frame = train_frame[np.isfinite(train_frame["d_mod3"])].reset_index(drop=True)
    if args.limit:
        train_frame = train_frame.iloc[:args.limit].reset_index(drop=True)
    loader = DataLoader(VolumeDataset(train_frame, args.img_size),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)

    train(model, loader, args, device)
    export_embeddings(model, frames, args, device, Path(args.out_dir))
    print("W2 PHASE 2 TRAIN+EXPORT DONE")


if __name__ == "__main__":
    main()
