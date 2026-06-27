from pathlib import Path

import pandas as pd


BASE = Path("d_contrastive/download_targets_20260515")
V3_PATH = BASE / "20260515_addimages_d_5_15_2026_v3.csv"
PREV_FINAL = BASE / "final_image_ids_by_phase_to_download_combined.csv"
OUT_TXT = BASE / "final_image_ids_to_download_v3_all.txt"
OUT_CSV = BASE / "final_image_ids_to_download_v3_all.csv"
EXCLUDED_CSV = BASE / "excluded_v3_existing_or_duplicate.csv"
SUMMARY_TXT = BASE / "projected_coverage_v3_summary.txt"


def collect_existing_ids():
    existing = set()
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
            existing.update(pd.read_csv(p, usecols=[c], dtype=str)[c].dropna().astype(str).str.strip().tolist())
    return existing


def match(visits, imgs, window):
    out = []
    by_rid = {rid: g[["image_id", "scan_date", "source"]] for rid, g in imgs.groupby("RID")}
    for _, v in visits.iterrows():
        g = by_rid.get(v.RID)
        if g is None:
            continue
        gaps = (g.scan_date - v["EXAMDATE.x"]).dt.days
        abs_gaps = gaps.abs()
        m = abs_gaps <= window
        if not m.any():
            continue
        j = abs_gaps[m].idxmin()
        r = v.to_dict()
        r.update({"image_id": g.loc[j, "image_id"], "gap_days": int(gaps.loc[j]), "image_source": g.loc[j, "source"]})
        out.append(r)
    return pd.DataFrame(out)


def traj_table(visits):
    rows = []
    for rid, g in visits.sort_values("EXAMDATE.x").groupby("RID"):
        dxs = sorted(set(g.dx.dropna().astype(str)))
        if len(dxs) == 1:
            traj = "stable_" + dxs[0]
        elif "NORMAL" in dxs and "AD" in dxs:
            traj = "CN_to_AD"
        elif "MCI" in dxs and "AD" in dxs:
            traj = "MCI_to_AD"
        elif "NORMAL" in dxs and "MCI" in dxs:
            traj = "CN_to_MCI"
        else:
            traj = "other"
        d = g.d_mod3.dropna()
        rows.append({"RID": rid, "trajectory": traj, "span_all": d.max() - d.min() if len(d) else pd.NA})
    return pd.DataFrame(rows)


def stats(matched, name, traj):
    unique_img = matched.drop_duplicates(["RID", "image_id"])
    n_per = unique_img.groupby("RID").size()
    per_visit = matched.groupby(["RID", "EXAMDATE.x"])["d_mod3"].median().reset_index()
    span = per_visit.groupby("RID")["d_mod3"].agg(lambda x: x.max() - x.min())
    span_pos = span[span > 0]
    mt = traj.merge(pd.DataFrame({"RID": n_per.index, "n_images": n_per.values}), on="RID", how="inner")
    return {
        "name": name,
        "matched_visits": len(matched),
        "unique_images": matched.image_id.nunique(),
        "unique_rids": matched.RID.nunique(),
        "rids_ge2_images": int((n_per >= 2).sum()),
        "rids_ge3_images": int((n_per >= 3).sum()),
        "rids_ge4_images": int((n_per >= 4).sum()),
        "span_median": float(span_pos.median()) if len(span_pos) else pd.NA,
        "span_ge05": int((span >= 0.5).sum()),
        "converter_rids": int(mt.trajectory.isin(["MCI_to_AD", "CN_to_AD", "CN_to_MCI"]).sum()),
        "mci_to_ad": int((mt.trajectory == "MCI_to_AD").sum()),
        "cn_to_ad": int((mt.trajectory == "CN_to_AD").sum()),
        "cn_to_mci": int((mt.trajectory == "CN_to_MCI").sum()),
    }


v3 = pd.read_csv(V3_PATH, dtype=str)
v3["Image Data ID"] = v3["Image Data ID"].astype(str).str.strip()
v3["Subject"] = v3["Subject"].astype(str).str.strip()
v3["scan_date"] = pd.to_datetime(v3["Acq Date"], errors="coerce")

existing_ids = collect_existing_ids()
prev_ids = set(pd.read_csv(PREV_FINAL, dtype=str)["Image Data ID"].dropna().astype(str).str.strip()) if PREV_FINAL.exists() else set()

mri = pd.read_csv("data/MRI3META_13May2026.csv", dtype=str, usecols=["PTID", "RID", "EXAMDATE"]).drop_duplicates()
mri["EXAMDATE_parsed"] = pd.to_datetime(mri["EXAMDATE"], errors="coerce")
map_exact = (
    mri.dropna(subset=["PTID", "RID", "EXAMDATE_parsed"])
    .drop_duplicates(["PTID", "EXAMDATE_parsed"])[["PTID", "RID", "EXAMDATE_parsed"]]
)
v3 = v3.merge(map_exact, left_on=["Subject", "scan_date"], right_on=["PTID", "EXAMDATE_parsed"], how="left")
rid_by_ptid = (
    mri.dropna(subset=["PTID", "RID"])
    .drop_duplicates("PTID")[["PTID", "RID"]]
    .rename(columns={"RID": "RID_fb"})
)
v3 = v3.merge(rid_by_ptid, left_on="Subject", right_on="PTID", how="left", suffixes=("", "_fb2"))
v3["RID_final"] = pd.to_numeric(v3["RID"].fillna(v3["RID_fb"]), errors="coerce")

v3["duplicate_in_v3"] = v3.duplicated("Image Data ID", keep="first")
unique = v3.drop_duplicates("Image Data ID", keep="first").copy()
unique["already_existing_project"] = unique["Image Data ID"].isin(existing_ids)
unique["was_in_previous_combined_final"] = unique["Image Data ID"].isin(prev_ids)
final = unique[~unique["already_existing_project"]].copy().sort_values(["Subject", "scan_date", "Image Data ID"])
final.to_csv(OUT_CSV, index=False)
OUT_TXT.write_text(",".join(final["Image Data ID"].drop_duplicates().tolist()) + "\n")

excluded = pd.concat(
    [
        unique[unique["already_existing_project"]].assign(exclude_reason="already_in_project"),
        v3[v3["duplicate_in_v3"]].assign(exclude_reason="duplicate_in_v3"),
    ],
    ignore_index=True,
    sort=False,
)
excluded.to_csv(EXCLUDED_CSV, index=False)

D = pd.read_csv("data/master_smri/D_with_image_paths_full.csv")
D["EXAMDATE.x"] = pd.to_datetime(D["EXAMDATE.x"])
img_col = "Image_Data_ID" if "Image_Data_ID" in D.columns else "Image Data ID"
visit = D.groupby(["RID", "EXAMDATE.x"], as_index=False).agg(
    d_mod3=("d_mod3", "median"),
    dx=("dx", lambda x: x.dropna().iloc[0] if x.dropna().size else pd.NA),
)
cur = D[D[img_col].notna()].copy()
cur["scan_date"] = pd.to_datetime(cur["scan_date"], errors="coerce")
cur_imgs = (
    cur.dropna(subset=["RID", img_col, "scan_date"])
    .drop_duplicates(["RID", img_col])[["RID", img_col, "scan_date"]]
    .rename(columns={img_col: "image_id"})
)
cur_imgs["source"] = "current"
add_imgs = (
    final.dropna(subset=["RID_final", "Image Data ID", "scan_date"])
    .drop_duplicates(["RID_final", "Image Data ID"])[["RID_final", "Image Data ID", "scan_date"]]
    .rename(columns={"RID_final": "RID", "Image Data ID": "image_id"})
)
add_imgs["RID"] = add_imgs["RID"].astype(int)
add_imgs["source"] = "v3_candidate"
proj_imgs = pd.concat([cur_imgs, add_imgs], ignore_index=True).drop_duplicates(["RID", "image_id"])

traj = traj_table(visit)
cur180 = match(visit, cur_imgs, 180)
proj180 = match(visit, proj_imgs, 180)
cur365 = match(visit, cur_imgs, 365)
proj365 = match(visit, proj_imgs, 365)
stats_df = pd.DataFrame(
    [
        stats(cur180, "current_180", traj),
        stats(proj180, "v3_projected_180", traj),
        stats(cur365, "current_365", traj),
        stats(proj365, "v3_projected_365", traj),
    ]
)

lines = [
    "=== V3 final image ID list + projected coverage ===",
    f"V3 input rows: {len(v3)}",
    f"V3 input unique Image Data IDs: {v3['Image Data ID'].nunique()}",
    f"Excluded already in project unique IDs: {int(unique.already_existing_project.sum())}",
    f"Final V3 download unique IDs: {final['Image Data ID'].nunique()}",
    f"Additional beyond previous combined final: {len(set(final['Image Data ID']) - prev_ids)}",
    f"Overlap with previous combined final retained: {len(set(final['Image Data ID']) & prev_ids)}",
    f"Final V3 candidate unique RIDs with RID mapped: {add_imgs.RID.nunique()}",
    "",
    stats_df.to_string(index=False),
]
SUMMARY_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("Wrote txt:", OUT_TXT)
print("Wrote csv:", OUT_CSV)
print("Wrote excluded:", EXCLUDED_CSV)
print("Wrote summary:", SUMMARY_TXT)
