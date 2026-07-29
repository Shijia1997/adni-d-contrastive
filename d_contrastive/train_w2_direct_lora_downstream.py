"""Direct downstream Swin fine-tuning on diagnosis/conversion labels.

This is the missing W2 arm: unlike Phase 2 d-adaptation, the Swin backbone is
adapted directly on downstream labels. It supports LoRA adapters or full
fine-tuning, uses the same 2-way split and raw SyN-registered T1 volumes, and
reports held-out test AUCs. Age/APOE are not used.
"""
import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from experiment_utils import bootstrap_auc
from minimal_v0_contrastive import load_features_2way
from run_w0_conversion_3way import TASKS, cohort
from train_w2_phase2_swin import build_backbone, inject_lora, load_volume


DX_TASKS = [("CN_vs_AD", "NORMAL", "AD"), ("CN_vs_MCI", "NORMAL", "MCI"),
            ("MCI_vs_AD", "MCI", "AD")]


class TaskVolumeDataset(Dataset):
    def __init__(self, frame, label_col, img_size):
        self.frame = frame.reset_index(drop=True)
        self.y = self.frame[label_col].to_numpy()
        self.img_size = tuple(img_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return load_volume(self.frame.loc[idx, "sMRI_path"], self.img_size), self.y[idx]


class DirectClassifier(nn.Module):
    def __init__(self, backbone, n_classes=1):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x):
        feats = self.backbone(x)[-1].mean(dim=(2, 3, 4))
        logits = self.classifier(feats)
        return logits.squeeze(1) if self.n_classes == 1 else logits


def auc_ci(y, score):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, score)
    _, lo, hi = bootstrap_auc(y, score)
    return float(auc), lo, hi


def make_dx_frames(data, neg, pos):
    out = {}
    for split in ["train", "test"]:
        df = data[split]["meta"].copy()
        df = df[df["dx"].isin([neg, pos])].drop_duplicates("Image_Data_ID").copy()
        df["label"] = (df["dx"] == pos).astype(int)
        out[split] = df
    return out["train"], out["test"]


def make_dx3_frames(data):
    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    out = {}
    for split in ["train", "test"]:
        df = data[split]["meta"].copy()
        df = df[df["dx"].isin(dx_map)].drop_duplicates("Image_Data_ID").copy()
        df["label"] = df["dx"].map(dx_map).astype(int)
        out[split] = df
    return out["train"], out["test"]


def make_conversion_frames(data, task, base, tgt, horizon):
    frames = {}
    for split in ["train", "test"]:
        c = cohort(data[split]["long_meta"], split, task, base, tgt, horizon)
        if len(c) == 0:
            frames[split] = pd.DataFrame()
            continue
        meta = data[split]["meta"].reset_index(drop=True)
        rows = meta.iloc[c["feature_idx"].astype(int).values].copy()
        rows["label"] = c["converted"].astype(int).values
        frames[split] = rows.drop_duplicates("Image_Data_ID")
    return frames["train"], frames["test"]


def train_eval(train_df, test_df, args, task_name, n_classes=1):
    ytr = train_df["label"].astype(int).values
    yte = test_df["label"].astype(int).values
    if n_classes == 1:
        bad = (len(train_df) < 10 or ytr.sum() < 2 or len(test_df) < 5 or
               yte.sum() < 1 or len(np.unique(yte)) < 2)
    else:
        bad = (len(train_df) < 10 or len(test_df) < 5 or
               len(np.unique(ytr)) < n_classes or len(np.unique(yte)) < n_classes)
    if bad:
        print(f"[skip] {task_name}: train n={len(train_df)} pos={int(ytr.sum()) if len(ytr) else 0}, "
              f"test n={len(test_df)} pos={int(yte.sum()) if len(yte) else 0}")
        return np.nan, np.nan, np.nan

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone, _ = build_backbone(args.model_id, device)
    if args.adapt == "lora":
        for p in backbone.parameters():
            p.requires_grad_(False)
        replaced = []
        inject_lora(backbone, re.compile(args.lora_targets), args.lora_rank, args.lora_alpha, replaced)
        if not replaced:
            raise SystemExit("LoRA target regex matched 0 modules")
    elif args.adapt == "full":
        for p in backbone.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown adapt mode: {args.adapt}")
    model = DirectClassifier(backbone, n_classes=n_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if n_classes == 1:
        pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32))
    loader = DataLoader(TaskVolumeDataset(train_df, "label", args.img_size),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    print(f"[{args.adapt}/{task_name}] train n={len(train_df)} labels={np.bincount(ytr, minlength=n_classes).tolist()} "
          f"test n={len(test_df)} labels={np.bincount(yte, minlength=n_classes).tolist()} "
          f"trainable={sum(p.numel() for p in params):,}")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = n = pending = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous()
            if n_classes == 1:
                y = y.to(device, non_blocking=True).float()
            else:
                y = y.to(device, non_blocking=True).long()
            loss = loss_fn(model(x), y) / args.accum_steps
            loss.backward()
            running += loss.item() * args.accum_steps
            n += 1
            pending += 1
            if pending == args.accum_steps:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"  epoch {epoch+1}/{args.epochs} loss={running / max(n, 1):.4f}", flush=True)

    model.eval()
    scores = []
    tloader = DataLoader(TaskVolumeDataset(test_df, "label", args.img_size),
                         batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, _ in tloader:
            logits = model(x.to(device).contiguous())
            if n_classes == 1:
                scores.append(torch.sigmoid(logits).cpu().item())
            else:
                scores.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    if n_classes == 1:
        return auc_ci(yte, np.asarray(scores))
    prob = np.asarray(scores)
    auc = roc_auc_score(yte, prob, multi_class="ovr", average="macro")
    return float(auc), np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016")
    ap.add_argument("--d_csv", default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w2_direct_lora_downstream")
    ap.add_argument("--adapt", choices=["lora", "full"], default="lora")
    ap.add_argument("--model_id", default="darragh/swinunetr-btcv-base")
    ap.add_argument("--img_size", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_targets", default=r"(qkv|proj|fc1|fc2)$")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_features_2way(args.features_dir, args.master_dir, args.d_csv, version="raw")
    method_name = f"direct_{args.adapt}_downstream"
    result_stem = "direct_lora_downstream" if args.adapt == "lora" else "direct_full_downstream"

    rows = []
    tr, te = make_dx3_frames(data)
    auc, lo, hi = train_eval(tr, te, args, "dx_CN_MCI_AD", n_classes=3)
    rows.append({"task_kind": "dx3", "task": "CN_MCI_AD", "horizon_years": np.nan,
                 "n": len(te), "n_positive": np.nan,
                 "method": method_name, "auc": auc, "ci_lo": lo, "ci_hi": hi})

    for name, neg, pos in DX_TASKS:
        tr, te = make_dx_frames(data, neg, pos)
        auc, lo, hi = train_eval(tr, te, args, f"dx_{name}")
        rows.append({"task_kind": "dx_binary", "task": name, "horizon_years": np.nan,
                     "n": len(te), "n_positive": int(te["label"].sum()) if len(te) else 0,
                     "method": method_name, "auc": auc, "ci_lo": lo, "ci_hi": hi})

    for task, base, tgt in TASKS:
        for h in args.horizons:
            tr, te = make_conversion_frames(data, task, base, tgt, h)
            auc, lo, hi = train_eval(tr, te, args, f"conv_{task}_{h}y")
            rows.append({"task_kind": "conversion", "task": task, "horizon_years": h,
                         "n": len(te), "n_positive": int(te["label"].sum()) if len(te) else 0,
                         "method": method_name, "auc": auc, "ci_lo": lo, "ci_hi": hi})

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{result_stem}.csv", index=False)
    with open(out / f"{result_stem}.md", "w") as f:
        f.write(f"# Direct {args.adapt} downstream fine-tuning\n\n")
        f.write(f"Swin is adapted with `{args.adapt}` directly on downstream diagnosis/conversion labels, not on d_mod3.\n\n")
        f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
