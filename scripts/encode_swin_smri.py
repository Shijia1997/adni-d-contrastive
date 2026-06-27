#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


BTCV_PATH = "/dcs07/zwang/data/pmrc/SwinUNETR/BTCV"
if BTCV_PATH not in sys.path:
    sys.path.append(BTCV_PATH)

from swinunetr import SwinUnetrModelForInference


def load_image(path, size):
    img = nib.load(path)
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError(f"no finite voxels: {path}")
    arr = np.where(finite, arr, 0.0)
    mask = finite & (np.abs(arr) > 1e-6)
    if int(mask.sum()) < 1000:
        mask = finite
    vals = arr[mask]
    mean = float(vals.mean())
    std = float(vals.std())
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError(f"near-zero intensity std: {path}")
    arr = (arr - mean) / std
    arr[~mask] = 0.0
    x = torch.from_numpy(arr[None, None])
    if tuple(arr.shape) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
    return x


def unique_images(matched_csv):
    df = pd.read_csv(matched_csv)
    required = ["Image_Data_ID", "sMRI_path", "RID", "EXAMDATE.x", "dx"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required matched columns: {missing}")
    imgs = (
        df.dropna(subset=["Image_Data_ID", "sMRI_path"])
        .sort_values(["Image_Data_ID", "sMRI_path"])
        .drop_duplicates("Image_Data_ID")
        [["Image_Data_ID", "sMRI_path"]]
        .reset_index(drop=True)
    )
    missing_paths = imgs[~imgs["sMRI_path"].map(lambda p: isinstance(p, str) and Path(p).exists())]
    if len(missing_paths):
        print(missing_paths.head(20).to_string(index=False))
        raise SystemExit(f"STOP: {len(missing_paths)} unique image paths do not exist")
    return df, imgs


def build_model(device, model_id):
    model = SwinUnetrModelForInference.from_pretrained(model_id)
    backbone = model.model.swinViT.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone, model.config


def encode(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched, imgs = unique_images(args.matched_csv)
    if args.limit:
        imgs = imgs.iloc[: args.limit].copy()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Unique images to encode: {len(imgs)}")
    backbone, config = build_model(device, args.model_id)
    print(f"Loaded model_id={args.model_id}, img_size={config.img_size}, feature_size={config.feature_size}")

    latents = []
    failures = []
    for i, row in imgs.iterrows():
        image_id = str(row["Image_Data_ID"])
        path = row["sMRI_path"]
        try:
            x = load_image(path, args.img_size).to(device, non_blocking=True)
            with torch.no_grad():
                feats = backbone(x.contiguous())
                deepest = feats[-1]
                latent = deepest.mean(dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(latent).all():
                raise ValueError("latent has NaN/Inf")
            latents.append(latent)
        except Exception as exc:
            failures.append({"image_id": image_id, "path": path, "error": repr(exc)})
            print(f"FAILED {image_id}: {exc}", flush=True)
            if len(failures) > args.max_failures:
                raise SystemExit(f"STOP: too many encoding failures ({len(failures)})")
            latents.append(np.full((768,), np.nan, dtype=np.float32))
        if (i + 1) % args.print_every == 0 or (i + 1) == len(imgs):
            print(f"Encoded {i + 1}/{len(imgs)}", flush=True)

    latents = np.stack(latents, axis=0)
    if failures:
        fail_path = out_dir / "swin_encoding_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2))
        raise SystemExit(f"STOP: {len(failures)} image encodings failed; see {fail_path}")

    image_ids = imgs["Image_Data_ID"].astype(str).to_numpy()
    paths = imgs["sMRI_path"].astype(str).to_numpy()
    np.save(out_dir / "swin_latent.npy", latents)
    np.save(out_dir / "image_id_order.npy", image_ids)
    np.save(out_dir / "swin_image_path_order.npy", paths)

    print("\nSTEP 4 VERIFY")
    print(f"Latent shape: {latents.shape}")
    print(f"Image ID order shape: {image_ids.shape}")
    print(f"NaN count: {int(np.isnan(latents).sum())}")
    variances = latents.var(axis=0)
    print(f"Per-dim variance min/median/max: {variances.min():.8g} / {np.median(variances):.8g} / {variances.max():.8g}")
    if latents.shape[0] != len(image_ids):
        raise SystemExit("STOP: latent row count does not match image_id_order")
    if np.isnan(latents).any():
        raise SystemExit("STOP: latent contains NaN")
    if np.any(variances <= 0):
        raise SystemExit("STOP: at least one latent dimension has zero variance")

    cosine_sanity(matched, image_ids, latents)


def cosine_sanity(matched, image_ids, latents):
    emb = pd.DataFrame({"Image_Data_ID": image_ids, "row": np.arange(len(image_ids))})
    m = matched.merge(emb, on="Image_Data_ID", how="inner")
    counts = m.groupby("RID")["row"].nunique()
    multi_rids = counts[counts >= 2].index[:20]
    within = []
    for rid in multi_rids:
        rows = sorted(m.loc[m["RID"] == rid, "row"].unique())
        sims = cosine_similarity(latents[rows])
        tri = sims[np.triu_indices_from(sims, k=1)]
        within.extend(tri.tolist())

    rng = np.random.default_rng(42)
    rid_rows = m.drop_duplicates(["RID", "row"])[["RID", "row"]]
    between = []
    if rid_rows["RID"].nunique() >= 2:
        for _ in range(min(200, len(rid_rows) * 2)):
            sample = rid_rows.sample(2, random_state=int(rng.integers(0, 1_000_000)))
            if sample["RID"].iloc[0] != sample["RID"].iloc[1]:
                r = sample["row"].to_numpy()
                between.append(float(cosine_similarity(latents[r[:1]], latents[r[1:2]])[0, 0]))

    print("\nCosine sanity")
    print(f"  within-subject pairs from first 20 multi-image RIDs: {len(within)}")
    if within and between:
        print(f"  within mean: {np.mean(within):.6f}")
        print(f"  between mean: {np.mean(between):.6f}")
        if np.mean(within) >= np.mean(between):
            print("  NOTE: within-subject cosine is >= between-subject cosine, which is expected for identity-like encoders.")
        else:
            print("  CAUTION: within-subject cosine is < between-subject cosine.")
    else:
        print("  skipped: not enough pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched_csv", default="data/master_smri/D_with_image_paths_matched.csv")
    parser.add_argument("--out_dir", default="data/embeddings")
    parser.add_argument("--model_id", default="darragh/swinunetr-btcv-base")
    parser.add_argument("--img_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--max_failures", type=int, default=0)
    args = parser.parse_args()
    encode(args)


if __name__ == "__main__":
    main()
