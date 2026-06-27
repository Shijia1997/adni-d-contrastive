#!/usr/bin/env python
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


EXISTING_META = Path("data/ADNI_all_with_path.csv")
NEW_JAC_DIR = Path("/dcs07/zwang/data/syn_jacobian")


def summarize_image(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    finite = np.isfinite(data)
    nonzero = data[finite & (np.abs(data) > 0.0001)]
    if len(nonzero) == 0:
        return {
            "ok": False,
            "shape": img.shape,
            "zooms": img.header.get_zooms(),
            "reason": "no nonzero finite voxels",
        }
    q = np.percentile(nonzero, [1, 5, 50, 95, 99])
    return {
        "ok": True,
        "shape": img.shape,
        "zooms": img.header.get_zooms(),
        "min": float(nonzero.min()),
        "max": float(nonzero.max()),
        "mean": float(nonzero.mean()),
        "std": float(nonzero.std()),
        "median": float(np.median(nonzero)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "frac_negative": float((nonzero < 0).mean()),
        "frac_positive": float((nonzero > 0).mean()),
    }


def infer_transform(stats_list):
    ok = [s for s in stats_list if s.get("ok")]
    if not ok:
        return "unknown_no_valid_samples"
    med = np.median([s["median"] for s in ok])
    frac_neg = np.median([s["frac_negative"] for s in ok])
    min_val = np.median([s["min"] for s in ok])
    max_val = np.median([s["max"] for s in ok])
    if frac_neg > 0.01 and abs(med) < 0.25:
        return "likely_log_jacobian"
    if frac_neg < 0.001 and 0.5 <= med <= 1.5 and min_val >= 0:
        return "likely_raw_jacobian"
    return f"ambiguous(median={med:.4f}, frac_negative={frac_neg:.4f}, median_min={min_val:.4f}, median_max={max_val:.4f})"


def print_stats(label, path, stats):
    print(f"  {label}")
    print(f"    path: {path}")
    if not stats.get("ok"):
        print(f"    {stats.get('reason', 'invalid')}")
        print(f"    shape: {stats['shape']}, voxel: {stats['zooms']}")
        return
    print(f"    shape: {stats['shape']}, voxel: {stats['zooms']}")
    print(f"    range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(
        f"    mean/std/median: {stats['mean']:.4f} / "
        f"{stats['std']:.4f} / {stats['median']:.4f}"
    )
    print(
        f"    q01/q05/q95/q99: {stats['q01']:.4f} / {stats['q05']:.4f} / "
        f"{stats['q95']:.4f} / {stats['q99']:.4f}"
    )
    print(
        f"    frac negative/positive: "
        f"{stats['frac_negative']:.4f} / {stats['frac_positive']:.4f}"
    )


def main():
    print("Step M1: Verify Jacobian sources, file existence, transform status")
    if not EXISTING_META.exists():
        raise SystemExit(f"STOP: existing metadata not found: {EXISTING_META}")
    if not NEW_JAC_DIR.exists():
        raise SystemExit(f"STOP: new Jacobian dir not found: {NEW_JAC_DIR}")

    existing_meta = pd.read_csv(EXISTING_META)
    print(f"existing metadata: {len(existing_meta)} rows")
    print(f"  columns: {existing_meta.columns.tolist()}")
    if "JSM_path" not in existing_meta.columns:
        raise SystemExit("STOP: existing metadata has no JSM_path column")
    print("  JSM_path sample:")
    print(existing_meta["JSM_path"].dropna().head(3).tolist())

    existing_has_jac = existing_meta["JSM_path"].apply(
        lambda p: os.path.exists(str(p)) if pd.notna(p) else False
    )
    print(f"  Files exist: {int(existing_has_jac.sum())} / {len(existing_meta)}")

    new_jac_files = [
        f.name
        for f in NEW_JAC_DIR.iterdir()
        if f.is_file() and f.name.startswith("syn_log_jacobian_I") and f.name.endswith(".nii.gz")
    ]
    print(f"\nNew Jacobian in {NEW_JAC_DIR}: {len(new_jac_files)} files")
    print(f"  Sample names: {sorted(new_jac_files)[:3]}")

    candidates = sorted(glob.glob("**/ready_download_498*.csv", recursive=True))
    print("\nNew 488 metadata candidates:")
    for candidate in candidates:
        print(f"  {candidate}")

    print("\n" + "=" * 70)
    print("Step M2: Intensity range diagnostic - infer transform status")
    print("=" * 70)

    print("\nExisting Jacobian (JSM_path):")
    existing_paths = [Path(p) for p in existing_meta.loc[existing_has_jac, "JSM_path"].dropna().head(5)]
    existing_stats = []
    for path in existing_paths:
        stats = summarize_image(path)
        existing_stats.append(stats)
        print_stats(path.name, path, stats)
    existing_infer = infer_transform(existing_stats)
    print(f"  Inference: {existing_infer}")

    print("\nNew Jacobian (syn_log_jacobian_*):")
    new_paths = [NEW_JAC_DIR / fn for fn in sorted(new_jac_files)[:5]]
    new_stats = []
    for path in new_paths:
        stats = summarize_image(path)
        new_stats.append(stats)
        print_stats(path.name, path, stats)
    new_infer = infer_transform(new_stats)
    print(f"  Inference: {new_infer}")

    print("\nStep M3: Shape + voxel size consistency check")
    all_sample_stats = []
    for path in existing_paths + new_paths:
        if path.exists():
            stats = summarize_image(path)
            all_sample_stats.append((path.name, stats["shape"], tuple(round(float(z), 6) for z in stats["zooms"])))
            print(f"  {path.name}: shape {stats['shape']}, voxel {stats['zooms']}")
    shape_counts = pd.Series([s[1] for s in all_sample_stats]).value_counts()
    voxel_counts = pd.Series([s[2] for s in all_sample_stats]).value_counts()

    print("\nFinal summary")
    print(f"  Existing Jacobian files found: {int(existing_has_jac.sum())} / {len(existing_meta)}")
    print(f"  Existing transform inference: {existing_infer}")
    print(f"  New Jacobian files found: {len(new_jac_files)} in {NEW_JAC_DIR}")
    print(f"  New transform inference: {new_infer}")
    print(f"  Sample shape counts: {shape_counts.to_dict()}")
    print(f"  Sample voxel counts: {voxel_counts.to_dict()}")
    print("  Naming difference: existing uses JSM_path; new uses syn_log_jacobian_I*.nii.gz")
    if existing_infer != new_infer:
        print("  WARNING: existing and new transform inference differ or are ambiguous; do not merge yet.")
    print("STOP: Jacobian source verification complete. Did not build unified dataset.")


if __name__ == "__main__":
    main()
