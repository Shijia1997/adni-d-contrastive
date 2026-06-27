#!/usr/bin/env python
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import master_merge_smri as mm
from diagnose_smri_matching import compute_span, match_with_window


class Args:
    d_csv = "data/d_value.csv"
    cdr_csv = "data/ADNI_CDR.csv"
    existing_csv = "data/ADNI_all_with_path.csv"
    new_csv = "data/ready_download_498d_5_10_2026_with_syn_path_with_jac.csv"
    old_join_csv = "data/D_with_image_paths.csv"
    current_matched_csv = "data/master_smri/D_with_image_paths_matched.csv"


TRAJECTORIES = ["MCI_to_AD", "CN_to_MCI", "CN_to_AD"]


def dx_join(x):
    vals = sorted(set(str(v) for v in x.dropna()))
    return "|".join(vals)


def classify_trajectory(row):
    dx_set = row.get("dx_set", "")
    if not isinstance(dx_set, str) or dx_set == "":
        return "unknown"
    dxs = dx_set.split("|")
    if len(dxs) == 1:
        return f"stable_{dxs[0]}"
    if "NORMAL" in dxs and "AD" in dxs:
        return "CN_to_AD"
    if "NORMAL" in dxs and "MCI" in dxs:
        return "CN_to_MCI"
    if "MCI" in dxs and "AD" in dxs:
        return "MCI_to_AD"
    return "other"


def add_subject_trajectory(df):
    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp["EXAMDATE.x"], errors="coerce")
    subj = (
        tmp.sort_values(["RID", "_date"])
        .groupby("RID")
        .agg(
            span=("d_mod3", lambda x: x.max() - x.min()),
            dx_set=("dx", dx_join),
            n_visits=("EXAMDATE.x", "nunique"),
            baseline_dx=("dx", "first"),
            final_dx=("dx", "last"),
        )
        .reset_index()
    )
    subj["trajectory"] = subj.apply(classify_trajectory, axis=1)
    return subj


def build_raw_image_meta(args, cdr):
    ptid_rid = (
        cdr[["PTID", "RID"]]
        .dropna()
        .assign(RID=lambda x: pd.to_numeric(x["RID"], errors="coerce"))
        .dropna()
        .drop_duplicates("PTID")
    )
    ptid_rid["RID"] = ptid_rid["RID"].astype(int)

    old_join = pd.read_csv(args.old_join_csv)
    inferred_dates = (
        old_join.dropna(subset=["Image_Data_ID", "scan_date"])
        .assign(scan_date=lambda x: pd.to_datetime(x["scan_date"], errors="coerce"))
        .dropna(subset=["scan_date"])
        .drop_duplicates("Image_Data_ID")[["Image_Data_ID", "scan_date"]]
        .rename(columns={"Image_Data_ID": "image_id"})
    )

    existing = pd.read_csv(args.existing_csv).copy()
    existing = existing.rename(columns={"Image Data ID": "image_id", "Subject": "PTID"})
    existing["source"] = "existing"
    existing["scan_date"] = pd.NaT
    existing["has_syn_path"] = existing["sMRI_path"].map(lambda p: isinstance(p, str) and Path(p).exists())
    existing = existing.merge(inferred_dates, on="image_id", how="left", suffixes=("", "_inferred"))
    existing["scan_date"] = existing["scan_date_inferred"]
    existing = existing.drop(columns=["scan_date_inferred"])

    new = pd.read_csv(args.new_csv).copy()
    new = new.rename(columns={"Image Data ID": "image_id", "Subject": "PTID"})
    new["source"] = "new_488"
    new["scan_date"] = pd.to_datetime(new["Acq Date"], errors="coerce")
    new["sMRI_path"] = new["syn_path"]
    new["has_syn_path"] = new["syn_path"].map(lambda p: isinstance(p, str) and Path(p).exists())

    cols = ["image_id", "PTID", "source", "scan_date", "sMRI_path", "has_syn_path"]
    raw = pd.concat([existing[cols], new[cols]], ignore_index=True)
    raw = raw.merge(ptid_rid, on="PTID", how="left")
    return raw


def print_converter_image_counts(full_subjects, raw_images, current_matched):
    matched_subjects = add_subject_trajectory(current_matched)
    for traj in TRAJECTORIES:
        converter_rids = set(full_subjects.loc[full_subjects["trajectory"] == traj, "RID"].astype(int))
        visits_per = full_subjects.loc[full_subjects["RID"].isin(converter_rids)].set_index("RID")["n_visits"]
        traj_images = raw_images[raw_images["RID"].isin(converter_rids)].copy()
        images_per = traj_images.groupby("RID").size()
        usable_per = traj_images[traj_images["has_syn_path"]].groupby("RID").size()
        matched_rids = set(matched_subjects.loc[matched_subjects["trajectory"] == traj, "RID"].astype(int))
        has_image_no_match = set(images_per.index.astype(int)) - matched_rids
        has_usable_no_match = set(usable_per.index.astype(int)) - matched_rids

        print(f"\n=== {traj} converter image coverage ===")
        print(f"Total converters in full D: {len(converter_rids)}")
        print(f"D visits per converter: median {visits_per.median()}, max {visits_per.max()}")
        print(
            "Total images per converter, any preprocessing state: "
            f"subjects_with_image {len(images_per)}, median {images_per.median()}, max {images_per.max()}"
        )
        print(
            "Usable SyN images per converter: "
            f"subjects_with_usable {len(usable_per)}, median {usable_per.median()}, max {usable_per.max()}"
        )
        print(f"Converters with any image but NOT in current matched set: {len(has_image_no_match)}")
        print(f"Converters with usable image but NOT in current matched set: {len(has_usable_no_match)}")


def print_gap_distribution(d, full_subjects, raw_images):
    d = d.copy()
    d["visit_date"] = pd.to_datetime(d["EXAMDATE.x"], errors="coerce")
    usable = raw_images[raw_images["has_syn_path"] & raw_images["scan_date"].notna()].copy()
    for traj in TRAJECTORIES:
        converter_rids = set(full_subjects.loc[full_subjects["trajectory"] == traj, "RID"].astype(int))
        gaps = []
        rids_with_usable_scan_date = 0
        for rid in converter_rids:
            rid_visits = d[(d["RID"] == rid) & d["visit_date"].notna()][["visit_date"]].drop_duplicates()
            rid_images = usable[usable["RID"] == rid][["scan_date"]].drop_duplicates()
            if rid_images.empty:
                continue
            rids_with_usable_scan_date += 1
            for _, visit in rid_visits.iterrows():
                min_gap = (rid_images["scan_date"] - visit["visit_date"]).abs().dt.days.min()
                if pd.notna(min_gap):
                    gaps.append(float(min_gap))
        arr = np.array(gaps)
        print(f"\n=== Closest-image-to-D-visit gap: {traj} ===")
        print(f"RIDs with usable image scan_date: {rids_with_usable_scan_date}")
        print(f"converter D visits with at least one usable image: {len(arr)}")
        if len(arr) == 0:
            continue
        print(f"  median: {np.median(arr):.1f}")
        print(f"  75%: {np.percentile(arr, 75):.1f}")
        print(f"  90%: {np.percentile(arr, 90):.1f}")
        print(f"  95%: {np.percentile(arr, 95):.1f}")
        print(f"  <=180: {(arr <= 180).sum()} / {len(arr)}")
        print(f"  <=365: {(arr <= 365).sum()} / {len(arr)}")
        print(f"  <=730: {(arr <= 730).sum()} / {len(arr)}")


def print_window_metrics(d, images):
    for window in [365, 730]:
        matched, _ = match_with_window(d, images, window)
        subj = add_subject_trajectory(matched)
        span = compute_span(matched)
        high = subj[subj["span"] >= 0.5]
        print(f"\n=== Window +/-{window} matching metrics ===")
        print(f"Total matched visits: {len(matched)}")
        print("Matched converter count by trajectory:")
        print(subj["trajectory"].value_counts().reindex(TRAJECTORIES).fillna(0).astype(int).to_string())
        print("Within-span describe:")
        print(span.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())
        print(">=0.5 deltaD count by trajectory:")
        print(high["trajectory"].value_counts().reindex(TRAJECTORIES).fillna(0).astype(int).to_string())


def main():
    args = Args()
    d = pd.read_csv(args.d_csv)
    cdr = mm.read_csv(args.cdr_csv, "cdr")
    raw_images = build_raw_image_meta(args, cdr)
    usable_images = mm.build_images(args, cdr)
    current_matched = pd.read_csv(args.current_matched_csv)
    full_subjects = add_subject_trajectory(d)

    print_converter_image_counts(full_subjects, raw_images, current_matched)
    print_gap_distribution(d, full_subjects, raw_images)
    print_window_metrics(d, usable_images)


if __name__ == "__main__":
    main()
