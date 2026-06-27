#!/usr/bin/env python
import os
from pathlib import Path

import pandas as pd


EXISTING_META = Path("data/ADNI_all_with_path.csv")
NEW_JAC_DIR = Path("/dcs07/zwang/data/syn_jacobian")
MASTER_DIR = Path("data/master_smri")


def image_col(df):
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    raise KeyError("No Image Data ID column found")


def build_lookup():
    print("Step N1: Build Jacobian path lookup table")
    existing_meta = pd.read_csv(EXISTING_META)
    if "JSM_path" not in existing_meta.columns:
        raise SystemExit("STOP: existing metadata has no JSM_path column")
    existing_col = image_col(existing_meta)
    existing_meta["has_jsm"] = existing_meta["JSM_path"].apply(
        lambda p: os.path.exists(str(p)) if pd.notna(p) else False
    )
    print(f"Existing source: {int(existing_meta['has_jsm'].sum())} / {len(existing_meta)} have Jacobian")

    if not NEW_JAC_DIR.exists():
        raise SystemExit(f"STOP: new Jacobian dir not found: {NEW_JAC_DIR}")
    new_jac_files = [
        f.name
        for f in NEW_JAC_DIR.iterdir()
        if f.is_file() and f.name.startswith("syn_log_jacobian_I") and f.name.endswith(".nii.gz")
    ]
    new_jac_lookup = {}
    for filename in new_jac_files:
        iid = filename.replace("syn_log_jacobian_", "").replace(".nii.gz", "")
        new_jac_lookup[iid] = str(NEW_JAC_DIR / filename)
    print(f"New source: {len(new_jac_lookup)} Jacobian files in {NEW_JAC_DIR}")

    print("\nStep N2: Build unified Image Data ID -> Jacobian path")
    unified = {}
    for _, row in existing_meta.iterrows():
        iid = str(row[existing_col])
        path = row["JSM_path"]
        if pd.notna(path) and os.path.exists(str(path)):
            unified[iid] = str(path)

    new_added = 0
    for iid, path in new_jac_lookup.items():
        if iid not in unified:
            unified[iid] = path
            new_added += 1

    existing_added = len(unified) - new_added
    overlap = set(existing_meta[existing_col].astype(str)) & set(new_jac_lookup)
    print(f"Unified lookup: {len(unified)} total")
    print(f"  From existing: {existing_added}")
    print(f"  From new dir: {new_added}")
    print(f"  Image IDs in both sources: {len(overlap)}")
    if overlap:
        print(f"  Sample overlap IDs: {sorted(overlap)[:5]}")
        print("  -> Using existing source for overlap IDs")
    return unified


def add_jacobian_path(df, lookup):
    col = image_col(df)
    out = df.copy()
    out["jacobian_path"] = out[col].astype(str).map(lookup)
    out["has_jacobian"] = out["jacobian_path"].apply(
        lambda p: os.path.exists(str(p)) if pd.notna(p) else False
    )
    return out


def print_dataset_summary(name, df):
    col = image_col(df)
    has = df["has_jacobian"]
    print(f"\n{name} matched dataset:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with jacobian_path: {int(df['jacobian_path'].notna().sum())}")
    print(f"  Rows with file existing: {int(has.sum())}")
    print(f"  Unique images with Jacobian: {df.loc[has, col].nunique()}")


def missing_diagnostic(name, df):
    col = image_col(df)
    missing = df[~df["has_jacobian"]]
    missing_unique = missing.drop_duplicates(col)
    print(f"\n{name} images without Jacobian: {len(missing_unique)}")
    if len(missing_unique) == 0:
        return
    if "source" in missing_unique.columns:
        print("  By source:")
        print(missing_unique["source"].value_counts(dropna=False).to_string())
    else:
        print("  no source column")
    phase_cols = [c for c in missing_unique.columns if "phase" in c.lower()]
    if phase_cols:
        print(f"\n  By phase ({phase_cols[0]}):")
        print(missing_unique[phase_cols[0]].value_counts(dropna=False).to_string())
    print(f"\n  Sample missing image IDs: {missing_unique[col].astype(str).head(10).tolist()}")


def main():
    lookup = build_lookup()

    print("\nStep N3: Apply lookup to matched dataset")
    train = pd.read_csv(MASTER_DIR / "matched_TRAIN_with_batch.csv")
    test = pd.read_csv(MASTER_DIR / "matched_TEST_with_batch.csv")
    train_jac = add_jacobian_path(train, lookup)
    test_jac = add_jacobian_path(test, lookup)
    print_dataset_summary("Train", train_jac)
    print_dataset_summary("Test", test_jac)

    print("\nStep N4: Save unified datasets")
    train_out = MASTER_DIR / "matched_TRAIN_with_jac.csv"
    test_out = MASTER_DIR / "matched_TEST_with_jac.csv"
    train_jac.to_csv(train_out, index=False)
    test_jac.to_csv(test_out, index=False)
    print(f"Saved:")
    print(f"  {train_out}")
    print(f"  {test_out}")

    print("\nStep N5: Missing Jacobian diagnostic")
    missing_diagnostic("Train", train_jac)
    missing_diagnostic("Test", test_jac)

    train_col = image_col(train_jac)
    test_col = image_col(test_jac)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Train: {int(train_jac['has_jacobian'].sum())} / {len(train_jac)} rows have Jacobian")
    print(f"Train: {train_jac.loc[train_jac['has_jacobian'], train_col].nunique()} unique images with Jacobian")
    print(f"Test:  {int(test_jac['has_jacobian'].sum())} / {len(test_jac)} rows have Jacobian")
    print(f"Test:  {test_jac.loc[test_jac['has_jacobian'], test_col].nunique()} unique images with Jacobian")
    print("STOP: Jacobian paths added. Did not load voxel data or run PCA.")


if __name__ == "__main__":
    main()
