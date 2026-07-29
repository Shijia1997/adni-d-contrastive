#!/usr/bin/env python
"""Fast Workstream-0 rank ablation.

Runs only Setup3 (pretrain + frozen linear/Ridge probe) for the loss modes in
plan.md. This avoids re-running Setup1/2/4 for every loss setting.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from minimal_v0_contrastive import load_features_and_labels, run_setup3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--d_csv", default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w0_setup3_only")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", nargs="+",
                    default=["euclidean", "rank_kendall", "rank_kendall_basic", "hybrid", "hybrid_basic"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rank_alpha", type=float, default=10.0)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--rank_nu", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--latent", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for version in args.versions:
        data = load_features_and_labels(args.features_dir, args.d_csv, version=version)
        for mode in args.modes:
            print(f"\n=== W0 setup3-only version={version} mode={mode} ===", flush=True)
            mode_dir = out / f"mode_{mode}"
            mode_dir.mkdir(parents=True, exist_ok=True)
            results = {}
            run_setup3(
                data, results, device=args.device, tau=args.tau, beta=args.beta,
                epochs=args.epochs, output_dir=mode_dir, version=version,
                hidden=args.hidden, latent=args.latent, loss_mode=mode,
                rank_alpha=args.rank_alpha, lambda_rank=args.lambda_rank,
                rank_nu=args.rank_nu,
            )
            for setup, metrics in results.items():
                row = {"input_version": version, "loss_mode": mode, "setup": setup}
                row.update(metrics)
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(out / "w0_setup3_only_all_results.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "w0_setup3_only_all_results.csv", index=False)
    setup3 = df[df["setup"] == "setup3"].copy()
    setup3.to_csv(out / "w0_setup3_only_summary.csv", index=False)
    print("\n=== W0 setup3-only summary ===")
    print(setup3.to_string(index=False))
    print(f"Wrote {out / 'w0_setup3_only_all_results.csv'}")
    print(f"Wrote {out / 'w0_setup3_only_summary.csv'}")


if __name__ == "__main__":
    main()
