#!/usr/bin/env python
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import master_merge_smri as mm


class Args:
    d_csv = "data/d_value.csv"
    cdr_csv = "data/ADNI_CDR.csv"
    existing_csv = "data/ADNI_all_with_path.csv"
    new_csv = "data/ready_download_498d_5_10_2026_with_syn_path_with_jac.csv"
    old_join_csv = "data/D_with_image_paths.csv"


def compute_span(df):
    per_visit = (
        df.dropna(subset=["EXAMDATE.x"])
        .groupby(["RID", "EXAMDATE.x"])["d_mod3"]
        .median()
        .reset_index()
    )
    span = per_visit.groupby("RID")["d_mod3"].agg(lambda x: x.max() - x.min())
    return span[span > 0]


def match_with_window(d, images, window):
    dd = d.copy()
    dd["RID"] = pd.to_numeric(dd["RID"], errors="coerce")
    dd = dd.dropna(subset=["RID"]).copy()
    dd["RID"] = dd["RID"].astype(int)
    dd["exam_date"] = pd.to_datetime(dd["EXAMDATE.x"], errors="coerce")
    dd["merge_row_id"] = np.arange(len(dd))

    dated = dd.dropna(subset=["exam_date"])
    cand = dated[["merge_row_id", "RID", "exam_date"]].merge(images, on="RID", how="inner")
    cand["gap_days"] = (cand["scan_date"] - cand["exam_date"]).dt.days
    cand["abs_gap_days"] = cand["gap_days"].abs()
    cand = cand[cand["abs_gap_days"] <= window].copy()
    cand["source_rank"] = cand["source"].map({"new_488": 0, "existing": 1}).fillna(9)
    best = (
        cand.sort_values(["merge_row_id", "abs_gap_days", "source_rank", "image_id"])
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
    out = dd.merge(best[best_cols], on="merge_row_id", how="left")
    out = out.rename(columns={"Image Data ID": "Image_Data_ID"})
    return out[out["sMRI_path"].notna()].copy(), cand


def main():
    args = Args()
    d = pd.read_csv(args.d_csv)

    print("=== CHECK 1: ALL D DATA (no image filter) within-span ===")
    within_span_all = compute_span(d)
    print(within_span_all.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())
    print(f">=0.3 deltaD subjects: {(within_span_all >= 0.3).sum()}")
    print(f">=0.5 deltaD subjects: {(within_span_all >= 0.5).sum()}")
    print(f">=1.0 deltaD subjects: {(within_span_all >= 1.0).sum()}")
    print(f"total longitudinal subjects with span>0: {len(within_span_all)}")

    cdr = mm.read_csv(args.cdr_csv, "cdr")
    images = mm.build_images(args, cdr)

    print("\n=== CHECK 2: Matching window impact ===")
    for window in [90, 180, 365]:
        matched, candidates = match_with_window(d, images, window)
        span = compute_span(matched)
        print(
            f"Window +/-{window}: "
            f"matched rows={len(matched)}, "
            f"unique images={matched.Image_Data_ID.nunique()}, "
            f"unique RIDs={matched.RID.nunique()}, "
            f"median span={span.median():.6f}, "
            f">=0.5 deltaD subjects={(span >= 0.5).sum()}, "
            f">=1.0={(span >= 1.0).sum()}"
        )

    print("\n=== CHECK 3: Unmatched usable images and RID overlap ===")
    matched180, candidates180 = match_with_window(d, images, 180)
    used_ids = set(matched180["Image_Data_ID"].astype(str))
    matched_rids = set(matched180["RID"].astype(int))
    unmatched_images = images[~images["Image Data ID"].astype(str).isin(used_ids)].copy()
    unmatched_image_subjects = unmatched_images.groupby("RID").size().sort_values(ascending=False)
    overlap = set(unmatched_image_subjects.index.astype(int)) & matched_rids
    cand_ids = set(candidates180["Image Data ID"].astype(str))
    has_candidate = unmatched_images["Image Data ID"].astype(str).isin(cand_ids)

    print(f"usable images total: {len(images)}")
    print(f"used unique images at +/-180: {len(used_ids)}")
    print(f"unmatched usable images: {len(unmatched_images)}")
    print(f"unmatched image RIDs: {unmatched_images.RID.nunique()}")
    print(f"RIDs with unmatched images already in matched RID set: {len(overlap)}")
    print("unmatched images by source:")
    print(unmatched_images.groupby("source").size().to_string())
    print("top 20 RIDs by unmatched usable image count:")
    print(unmatched_image_subjects.head(20).to_string())
    print(
        "unmatched images with at least one D visit candidate within +/-180 but never selected: "
        f"{int(has_candidate.sum())}"
    )
    print(
        "unmatched images with no D visit within +/-180: "
        f"{int((~has_candidate).sum())}"
    )


if __name__ == "__main__":
    main()
