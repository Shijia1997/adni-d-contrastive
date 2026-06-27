#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "d": ["RID", "EXAMDATE.x", "dx", "d_mod3"],
    "cdr": ["PTID", "RID", "VISDATE", "PHASE"],
    "existing": ["Image Data ID", "Subject", "sMRI_path", "study"],
    "new": ["Image Data ID", "Subject", "Acq Date", "syn_path", "preprocess_status"],
    "old_join": ["Image_Data_ID", "scan_date"],
}


def require_columns(name, df):
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise SystemExit(f"STOP: {name} missing required columns: {missing}")


def read_csv(path, name):
    if not Path(path).exists():
        raise SystemExit(f"STOP: missing {name}: {path}")
    df = pd.read_csv(path)
    require_columns(name, df)
    return df


def phase_from_date(scan_date):
    # ADNI phase labels are needed only for source sanity checks. The new
    # download metadata lacks a phase column, so use broad ADNI calendar eras.
    if pd.isna(scan_date):
        return np.nan
    if scan_date < pd.Timestamp("2016-01-01"):
        return "ADNI2_or_GO"
    if scan_date < pd.Timestamp("2022-01-01"):
        return "ADNI3"
    return "ADNI4"


def build_ptid_rid_map(cdr):
    mapping = (
        cdr.dropna(subset=["PTID", "RID"])
        .assign(RID=lambda x: pd.to_numeric(x["RID"], errors="coerce"))
        .dropna(subset=["RID"])
        .sort_values(["PTID", "RID"])
        .drop_duplicates("PTID")[["PTID", "RID"]]
    )
    mapping["RID"] = mapping["RID"].astype(int)
    return mapping.rename(columns={"PTID": "Subject"})


def build_images(args, cdr):
    ptid_rid = build_ptid_rid_map(cdr)

    old_join = read_csv(args.old_join_csv, "old_join")
    image_dates = (
        old_join.dropna(subset=["Image_Data_ID", "scan_date"])
        .assign(scan_date=lambda x: pd.to_datetime(x["scan_date"], errors="coerce"))
        .dropna(subset=["scan_date"])
        .drop_duplicates("Image_Data_ID")[["Image_Data_ID", "scan_date"]]
        .rename(columns={"Image_Data_ID": "Image Data ID"})
    )

    existing = read_csv(args.existing_csv, "existing")
    existing = existing.merge(image_dates, on="Image Data ID", how="left")
    existing = existing.merge(ptid_rid, on="Subject", how="left")
    existing["source"] = "existing"
    existing["image_phase"] = existing["study"]
    existing["image_CDGLOBAL"] = pd.to_numeric(existing.get("CDGLOBAL"), errors="coerce")
    existing = existing.rename(columns={"JSM_path": "JSM_path"})
    existing["path_exists"] = existing["sMRI_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))

    new = read_csv(args.new_csv, "new")
    new["scan_date"] = pd.to_datetime(new["Acq Date"], errors="coerce")
    new = new.merge(ptid_rid, on="Subject", how="left")
    new["source"] = "new_488"
    new["image_phase"] = new["scan_date"].map(phase_from_date)
    new["sMRI_path"] = new["syn_path"]
    new["JSM_path"] = new.get("jacobian_path")
    new["image_CDGLOBAL"] = np.nan
    new["study"] = new["image_phase"]
    new["path_exists"] = new["sMRI_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))

    image_cols = [
        "Image Data ID",
        "Subject",
        "RID",
        "sMRI_path",
        "JSM_path",
        "scan_date",
        "source",
        "image_phase",
        "image_CDGLOBAL",
        "path_exists",
    ]
    images = pd.concat([existing[image_cols], new[image_cols]], ignore_index=True)
    images = images.dropna(subset=["RID", "scan_date", "sMRI_path"])
    images = images[images["path_exists"]].copy()
    images["RID"] = images["RID"].astype(int)
    images["image_id"] = images["Image Data ID"].astype(str)

    # If an image appears in both sources, keep the new local SyN path.
    images["source_rank"] = images["source"].map({"new_488": 0, "existing": 1}).fillna(9)
    images = images.sort_values(["image_id", "source_rank"]).drop_duplicates("image_id", keep="first")
    return images.drop(columns=["source_rank"])


def match_visits(d, images):
    d = d.copy()
    d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
    d = d.dropna(subset=["RID"]).copy()
    d["RID"] = d["RID"].astype(int)
    d["exam_date"] = pd.to_datetime(d["EXAMDATE.x"], errors="coerce")
    bad_exam = d["exam_date"].isna().sum()
    if bad_exam:
        print(f"WARNING: {bad_exam} D rows have empty/unparsable EXAMDATE.x; keeping them unmatched")
    d["merge_row_id"] = np.arange(len(d))

    dated = d.dropna(subset=["exam_date"])
    candidates = dated[["merge_row_id", "RID", "exam_date"]].merge(images, on="RID", how="inner")
    candidates["gap_days"] = (candidates["scan_date"] - candidates["exam_date"]).dt.days
    candidates["abs_gap_days"] = candidates["gap_days"].abs()
    candidates = candidates[candidates["abs_gap_days"] <= 180].copy()
    candidates["source_rank"] = candidates["source"].map({"new_488": 0, "existing": 1}).fillna(9)
    best = (
        candidates.sort_values(["merge_row_id", "abs_gap_days", "source_rank", "image_id"])
        .drop_duplicates("merge_row_id", keep="first")
    )

    best_cols = [
        "merge_row_id",
        "Image Data ID",
        "sMRI_path",
        "JSM_path",
        "source",
        "gap_days",
        "scan_date",
        "image_phase",
        "image_CDGLOBAL",
    ]
    out = d.merge(best[best_cols], on="merge_row_id", how="left")
    out["has_image"] = out["sMRI_path"].notna()
    out = out.rename(columns={"Image Data ID": "Image_Data_ID"})
    out = out.drop(columns=["merge_row_id", "exam_date"])
    return out, candidates, images


def print_sanity(full, images, args):
    matched = full[full["has_image"]].copy()
    print("\nSTEP 2 SANITY CHECK")
    print("(1) Totals")
    print(f"  Total D visit: {len(full)}")
    print(f"  Matched visits: {len(matched)}")
    print(f"  Unique image used: {matched['Image_Data_ID'].nunique()}")
    print(f"  Unique RID with >=1 matched image: {matched['RID'].nunique()}")
    print(f"  Available image metadata after path/date filters: {len(images)}")
    print(f"  Available images by source:\n{images.groupby('source').size().to_string()}")

    print("\n(2) Source breakdown")
    print(matched.groupby("source").size().to_string())

    print("\n(3) Gap days")
    print(matched["gap_days"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

    print("\n(4) Phase distribution")
    print(matched.groupby(["source", "image_phase"]).size().to_string())

    print("\n(5) Within-subject D span")
    per_visit = matched.groupby(["RID", "EXAMDATE.x"])["d_mod3"].median().reset_index()
    within_span = per_visit.groupby("RID")["d_mod3"].agg(lambda x: x.max() - x.min())
    within_span = within_span[within_span > 0]
    print(within_span.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())
    n03 = int((within_span >= 0.3).sum())
    n05 = int((within_span >= 0.5).sum())
    n10 = int((within_span >= 1.0).sum())
    print(f"  >=0.3 deltaD subjects: {n03}")
    print(f"  >=0.5 deltaD subjects: {n05}")
    print(f"  >=1.0 deltaD subjects: {n10}")

    print("\n(6) Image visit per subject")
    matched_unique = matched.drop_duplicates(["RID", "EXAMDATE.x", "sMRI_path"])
    n_per_subj = matched_unique.groupby("RID").size()
    for k in [1, 2, 3, 4, 5]:
        print(f"  >={k} unique image visits: {(n_per_subj >= k).sum()}")

    print("\n(7) Diagnostic distribution")
    print(matched.groupby("dx").size().to_string())

    failures = []
    if len(matched) <= args.min_matched:
        failures.append(f"matched visits {len(matched)} <= {args.min_matched}")
    if matched["Image_Data_ID"].nunique() <= args.min_unique_images:
        failures.append(f"unique images {matched['Image_Data_ID'].nunique()} <= {args.min_unique_images}")
    if matched["RID"].nunique() <= args.min_unique_rids:
        failures.append(f"unique RIDs {matched['RID'].nunique()} <= {args.min_unique_rids}")
    source_counts = matched.groupby("source").size()
    if source_counts.get("new_488", 0) < args.min_new_rows:
        failures.append(f"new_488 matched rows {source_counts.get('new_488', 0)} < {args.min_new_rows}")
    if within_span.median() < args.min_within_span_median:
        failures.append(f"within-subject median deltaD {within_span.median():.3f} < {args.min_within_span_median}")
    if n05 < args.min_delta05_subjects:
        failures.append(f">=0.5 deltaD subjects {n05} < {args.min_delta05_subjects}")
    dx_counts = matched.groupby("dx").size()
    for dx in ["NORMAL", "MCI", "AD"]:
        if dx_counts.get(dx, 0) == 0:
            failures.append(f"missing dx group {dx}")

    if failures:
        print("\nSTOP: sanity check failed")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(2)
    print("\nPASS: Step 2 sanity checks met configured thresholds")


def split_train_test(matched, out_dir):
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(matched, groups=matched["RID"]))
    train = matched.iloc[train_idx].copy()
    test = matched.iloc[test_idx].copy()
    train_rids = np.array(sorted(train["RID"].unique()))
    test_rids = np.array(sorted(test["RID"].unique()))
    overlap = set(train_rids).intersection(set(test_rids))
    if overlap:
        raise SystemExit(f"STOP: train/test RID overlap: {sorted(list(overlap))[:10]}")

    np.save(out_dir / "train_rids.npy", train_rids)
    np.save(out_dir / "test_rids.npy", test_rids)
    train.to_csv(out_dir / "matched_TRAIN.csv", index=False)
    test.to_csv(out_dir / "matched_TEST.csv", index=False)

    print("\nSTEP 3 TRAIN/TEST SPLIT")
    print(f"  Train rows/RIDs: {len(train)} / {len(train_rids)}")
    print(f"  Test rows/RIDs: {len(test)} / {len(test_rids)}")
    print(f"  RID overlap: {len(overlap)}")
    print("  Train dx distribution:")
    print(train.groupby("dx").size().to_string())
    print("  Test dx distribution:")
    print(test.groupby("dx").size().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_csv", default="data/d_value.csv")
    parser.add_argument("--cdr_csv", default="data/ADNI_CDR.csv")
    parser.add_argument("--existing_csv", default="data/ADNI_all_with_path.csv")
    parser.add_argument("--new_csv", default="data/ready_download_498d_5_10_2026_with_syn_path_with_jac.csv")
    parser.add_argument("--old_join_csv", default="data/D_with_image_paths.csv")
    parser.add_argument("--out_dir", default="data/master_smri")
    parser.add_argument("--min_matched", type=int, default=2700)
    parser.add_argument("--min_unique_images", type=int, default=2000)
    parser.add_argument("--min_unique_rids", type=int, default=900)
    parser.add_argument("--min_new_rows", type=int, default=300)
    parser.add_argument("--min_within_span_median", type=float, default=0.12)
    parser.add_argument("--min_delta05_subjects", type=int, default=80)
    args = parser.parse_args()

    d = read_csv(args.d_csv, "d")
    cdr = read_csv(args.cdr_csv, "cdr")
    images = build_images(args, cdr)
    full, candidates, images = match_visits(d, images)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_out = out_dir / "D_with_image_paths_full.csv"
    matched_out = out_dir / "D_with_image_paths_matched.csv"
    images_out = out_dir / "smri_image_metadata_used.csv"
    full.to_csv(full_out, index=False)
    full[full["has_image"]].to_csv(matched_out, index=False)
    images.to_csv(images_out, index=False)

    print("STEP 1 OUTPUT")
    print(f"  Full D with image columns: {full_out}")
    print(f"  Matched only: {matched_out}")
    print(f"  Image metadata used: {images_out}")

    print_sanity(full, images, args)
    split_train_test(full[full["has_image"]].copy(), out_dir)


if __name__ == "__main__":
    main()
