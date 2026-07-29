from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("d_contrastive/download_targets_20260515")
INPUTS = [
    BASE / "20260515_addimages_d_5_15_2026.csv",
    BASE / "20260515_addimages_d_5_15_2026_add.csv",
]
OUT_TXT = BASE / "final_image_ids_by_phase_to_download_combined.txt"
OUT_CSV = BASE / "final_image_ids_by_phase_to_download_combined.csv"
EXCLUDED_CSV = BASE / "excluded_duplicate_or_existing_image_ids_combined.csv"


def load_input(path, batch):
    df = pd.read_csv(path, dtype=str)
    df["source_csv"] = batch
    df["Image Data ID"] = df["Image Data ID"].astype(str).str.strip()
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Acq Date Parsed"] = pd.to_datetime(df["Acq Date"], errors="coerce")
    return df


new = pd.concat([load_input(p, p.name) for p in INPUTS], ignore_index=True)

existing_ids = set()
for p in [
    Path("data/master_smri/D_with_image_paths_full.csv"),
    Path("data/master_smri/D_with_image_paths_matched.csv"),
    Path("data/master_smri/smri_image_metadata_used.csv"),
    Path("data/ADNI_all_with_path.csv"),
    Path("data/ready_download_498d_5_10_2026_with_syn_path_with_jac.csv"),
]:
    if not p.exists():
        continue
    cols = pd.read_csv(p, nrows=0).columns.tolist()
    for c in [c for c in cols if c in {"Image Data ID", "Image_Data_ID", "image_id", "IMAGEUID"}]:
        existing_ids.update(pd.read_csv(p, usecols=[c], dtype=str)[c].dropna().astype(str).str.strip().tolist())

dup_mask = new.duplicated("Image Data ID", keep="first")
dups = new[dup_mask].copy()
unique = new.drop_duplicates("Image Data ID", keep="first").copy()
unique["already_existing_project"] = unique["Image Data ID"].isin(existing_ids)

mri = pd.read_csv("data/MRI3META_13May2026.csv", dtype=str, usecols=["PHASE", "PTID", "EXAMDATE"])
mri = mri[mri["PHASE"].isin(["ADNI2", "ADNI3", "ADNI4"])].copy()
mri["EXAMDATE Parsed"] = pd.to_datetime(mri["EXAMDATE"], errors="coerce")
phase_map_exact = (
    mri.dropna(subset=["PTID", "EXAMDATE Parsed", "PHASE"])
    .drop_duplicates(["PTID", "EXAMDATE Parsed", "PHASE"])
    .groupby(["PTID", "EXAMDATE Parsed"])["PHASE"]
    .agg(lambda x: "|".join(sorted(set(x))))
    .reset_index()
)
unique = unique.merge(
    phase_map_exact,
    left_on=["Subject", "Acq Date Parsed"],
    right_on=["PTID", "EXAMDATE Parsed"],
    how="left",
)
unique = unique.rename(columns={"PHASE": "phase_exact"})

trace = pd.read_csv(BASE / "ptid_targets_by_phase_top_priority.csv", dtype=str)
fallback = trace.groupby("PTID")["PHASE"].agg(lambda x: "|".join(sorted(set(x)))).reset_index()
unique = unique.merge(fallback, left_on="Subject", right_on="PTID", how="left", suffixes=("", "_fallback"))
unique = unique.rename(columns={"PHASE": "phase_fallback"})


def infer_phase(row):
    exact = row.get("phase_exact")
    if pd.notna(exact):
        phases = [p for p in str(exact).split("|") if p in {"ADNI2", "ADNI3", "ADNI4"}]
        if phases:
            return phases[0], "exact_date"
    fb = row.get("phase_fallback")
    fb_phases = [p for p in str(fb).split("|") if p in {"ADNI2", "ADNI3", "ADNI4"}] if pd.notna(fb) else []
    if len(fb_phases) == 1:
        return fb_phases[0], "ptid_single_phase"
    d = row.get("Acq Date Parsed")
    if pd.notna(d):
        if d < pd.Timestamp("2016-09-01"):
            guess = "ADNI2"
        elif d < pd.Timestamp("2022-01-01"):
            guess = "ADNI3"
        else:
            guess = "ADNI4"
        if not fb_phases or guess in fb_phases:
            return guess, "date_rule"
        return fb_phases[0], "ptid_multi_phase_fallback"
    if fb_phases:
        return fb_phases[0], "ptid_multi_phase_fallback"
    return pd.NA, "no_phase_match"


assigned = unique.apply(infer_phase, axis=1, result_type="expand")
unique["PHASE"] = assigned[0]
unique["phase_assignment_method"] = assigned[1]

excluded = unique[unique["already_existing_project"] | unique["PHASE"].isna()].copy()
excluded["exclude_reason"] = np.where(excluded["already_existing_project"], "already_in_project", "no_phase_match")
if len(dups):
    dd = dups.copy()
    dd["exclude_reason"] = "duplicate_image_id_in_input_union"
    excluded = pd.concat([excluded, dd], ignore_index=True, sort=False)

final = unique[(~unique["already_existing_project"]) & unique["PHASE"].notna()].copy()
final = final.drop_duplicates("Image Data ID", keep="first").sort_values(
    ["PHASE", "Subject", "Acq Date Parsed", "Image Data ID"]
)
final.to_csv(OUT_CSV, index=False)
excluded.to_csv(EXCLUDED_CSV, index=False)

lines = []
for ph in ["ADNI2", "ADNI3", "ADNI4"]:
    ids = final.loc[final["PHASE"] == ph, "Image Data ID"].drop_duplicates().sort_values().tolist()
    lines.append(ph + ": " + ",".join(ids))
OUT_TXT.write_text("\n".join(lines) + "\n")

previous = pd.read_csv(BASE / "final_image_ids_by_phase_to_download.csv", dtype=str)
prev_ids = set(previous["Image Data ID"].dropna().astype(str))
final_ids = set(final["Image Data ID"].dropna().astype(str))

print("Input rows:", len(new))
print("Input unique Image Data IDs:", new["Image Data ID"].nunique())
print("Duplicate Image IDs across input union:", len(new) - new["Image Data ID"].nunique())
print("Existing project image IDs scanned:", len(existing_ids))
print("Previous final unique IDs:", len(prev_ids))
print("Combined final unique IDs:", len(final_ids))
print("New IDs beyond previous final:", len(final_ids - prev_ids))
print("Previous IDs retained:", len(final_ids & prev_ids))
print("Excluded rows:", len(excluded))
print("Exclude reasons:")
print(excluded["exclude_reason"].value_counts().to_string())
print("Counts by phase:")
print(final["PHASE"].value_counts().reindex(["ADNI2", "ADNI3", "ADNI4"]).fillna(0).astype(int).to_string())
print("No duplicates in final:", final["Image Data ID"].is_unique)
print("Wrote txt:", OUT_TXT)
print("Wrote trace csv:", OUT_CSV)
print("Wrote excluded csv:", EXCLUDED_CSV)
