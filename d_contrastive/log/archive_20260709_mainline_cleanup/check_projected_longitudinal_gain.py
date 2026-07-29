from pathlib import Path

import numpy as np
import pandas as pd


base = Path("d_contrastive/download_targets_20260515")
d_path = Path("data/master_smri/D_with_image_paths_full.csv")
add_csv = base / "final_image_ids_by_phase_to_download.csv"
out_txt = base / "projected_longitudinal_gain_summary.txt"

D = pd.read_csv(d_path)
D["EXAMDATE.x"] = pd.to_datetime(D["EXAMDATE.x"])
img_col = "Image_Data_ID" if "Image_Data_ID" in D.columns else "Image Data ID"

visit = D.groupby(["RID", "EXAMDATE.x"], as_index=False).agg(
    d_mod3=("d_mod3", "median"),
    dx=("dx", lambda x: x.dropna().iloc[0] if x.dropna().size else np.nan),
    phase=("PHASE.x", lambda x: x.dropna().iloc[0] if x.dropna().size else np.nan),
)

cur = D[D[img_col].notna()].copy()
cur["scan_date"] = pd.to_datetime(cur["scan_date"], errors="coerce")
cur_imgs = (
    cur.dropna(subset=["RID", img_col, "scan_date"])
    .drop_duplicates(["RID", img_col])[["RID", img_col, "scan_date"]]
    .rename(columns={img_col: "image_id"})
)
cur_imgs["source"] = "current"

add = pd.read_csv(add_csv, dtype=str)
date_col = "Acq Date Parsed" if "Acq Date Parsed" in add.columns else "Acq Date"
add["scan_date"] = pd.to_datetime(add[date_col], errors="coerce")
if "RID" in add.columns:
    add["RID"] = pd.to_numeric(add["RID"], errors="coerce")
else:
    mri = pd.read_csv("data/MRI3META_13May2026.csv", dtype=str, usecols=["PTID", "RID"]).drop_duplicates()
    add = add.merge(mri, left_on="Subject", right_on="PTID", how="left")
    add["RID"] = pd.to_numeric(add["RID"], errors="coerce")
add_imgs = (
    add.dropna(subset=["RID", "Image Data ID", "scan_date"])
    .drop_duplicates(["RID", "Image Data ID"])[["RID", "Image Data ID", "scan_date"]]
    .rename(columns={"Image Data ID": "image_id"})
)
add_imgs["RID"] = add_imgs["RID"].astype(int)
add_imgs["source"] = "candidate"

proj_imgs = pd.concat([cur_imgs, add_imgs], ignore_index=True).drop_duplicates(["RID", "image_id"])


def match_visits(visits, imgs, window=180):
    out = []
    img_by_rid = {rid: g[["image_id", "scan_date", "source"]] for rid, g in imgs.groupby("RID")}
    for _, v in visits.iterrows():
        rid = v["RID"]
        g = img_by_rid.get(rid)
        if g is None or len(g) == 0:
            continue
        gaps = (g["scan_date"] - v["EXAMDATE.x"]).dt.days
        abs_g = gaps.abs()
        m = abs_g <= window
        if not m.any():
            continue
        j = abs_g[m].idxmin()
        row = v.to_dict()
        row.update(
            {
                "image_id": g.loc[j, "image_id"],
                "scan_date": g.loc[j, "scan_date"],
                "gap_days": int(gaps.loc[j]),
                "image_source": g.loc[j, "source"],
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def traj_table(visits):
    rows = []
    for rid, g in visits.sort_values("EXAMDATE.x").groupby("RID"):
        dxs = [str(x) for x in g["dx"].dropna().tolist()]
        s = sorted(set(dxs))
        if len(s) == 1:
            traj = f"stable_{s[0]}"
        elif "NORMAL" in s and "AD" in s:
            traj = "CN_to_AD"
        elif "MCI" in s and "AD" in s:
            traj = "MCI_to_AD"
        elif "NORMAL" in s and "MCI" in s:
            traj = "CN_to_MCI"
        else:
            traj = "other"
        d = g["d_mod3"].dropna()
        rows.append(
            {
                "RID": rid,
                "trajectory": traj,
                "d_span_all": d.max() - d.min() if len(d) else np.nan,
                "n_d_visits_all": g["EXAMDATE.x"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def subj_metrics(matched, name, traj):
    if len(matched) == 0:
        return {"name": name}
    unique_img = matched.drop_duplicates(["RID", "image_id"])
    n_per = unique_img.groupby("RID").size()
    per_visit = matched.groupby(["RID", "EXAMDATE.x"])["d_mod3"].median().reset_index()
    span = per_visit.groupby("RID")["d_mod3"].agg(lambda x: x.max() - x.min())
    span_pos = span[span > 0]
    mtraj = traj.merge(pd.DataFrame({"RID": n_per.index, "n_images": n_per.values}), on="RID", how="inner")
    return {
        "name": name,
        "matched_visits": len(matched),
        "unique_images": matched["image_id"].nunique(),
        "unique_rids": matched["RID"].nunique(),
        "rids_ge2_images": int((n_per >= 2).sum()),
        "rids_ge3_images": int((n_per >= 3).sum()),
        "rids_ge4_images": int((n_per >= 4).sum()),
        "median_images_per_rid": float(n_per.median()),
        "span_median": float(span_pos.median()) if len(span_pos) else np.nan,
        "span_ge05": int((span >= 0.5).sum()),
        "span_ge10": int((span >= 1.0).sum()),
        "converter_rids": int(mtraj["trajectory"].isin(["MCI_to_AD", "CN_to_AD", "CN_to_MCI"]).sum()),
        "mci_to_ad_rids": int((mtraj["trajectory"] == "MCI_to_AD").sum()),
        "cn_to_ad_rids": int((mtraj["trajectory"] == "CN_to_AD").sum()),
        "cn_to_mci_rids": int((mtraj["trajectory"] == "CN_to_MCI").sum()),
    }


traj = traj_table(visit)
cur_m180 = match_visits(visit, cur_imgs, 180)
proj_m180 = match_visits(visit, proj_imgs, 180)
cur_m365 = match_visits(visit, cur_imgs, 365)
proj_m365 = match_visits(visit, proj_imgs, 365)

stats_df = pd.DataFrame(
    [
        subj_metrics(cur_m180, "current_180", traj),
        subj_metrics(proj_m180, "projected_180", traj),
        subj_metrics(cur_m365, "current_365", traj),
        subj_metrics(proj_m365, "projected_365", traj),
    ]
)

add_rids = set(add_imgs["RID"].astype(int))
cur_counts = cur_imgs.groupby("RID").size()
proj_counts = proj_imgs.groupby("RID").size()
impact = traj[traj["RID"].isin(add_rids)].copy()
impact["current_images"] = impact["RID"].map(cur_counts).fillna(0).astype(int)
impact["projected_images"] = impact["RID"].map(proj_counts).fillna(0).astype(int)
impact["added_images"] = impact["projected_images"] - impact["current_images"]
conv_mask = impact["trajectory"].isin(["MCI_to_AD", "CN_to_AD", "CN_to_MCI"])

lines = []
lines.append("=== Projected longitudinal gain if all final Image IDs are downloaded/preprocessed ===")
lines.append(f"Candidate unique images: {add_imgs.image_id.nunique()}")
lines.append(f"Candidate unique RIDs: {add_imgs.RID.nunique()}")
lines.append("")
lines.append(stats_df.to_string(index=False))
lines.append("")
lines.append("Candidate RID impact:")
lines.append(f"  RIDs getting added images: {impact.RID.nunique()}")
lines.append(f"  converters getting added images: {int(conv_mask.sum())}")
lines.append(f"    MCI_to_AD: {int((impact.trajectory == 'MCI_to_AD').sum())}")
lines.append(f"    CN_to_AD: {int((impact.trajectory == 'CN_to_AD').sum())}")
lines.append(f"    CN_to_MCI: {int((impact.trajectory == 'CN_to_MCI').sum())}")
lines.append(f"  d_span>=0.5 RIDs getting added images: {int((impact.d_span_all >= 0.5).sum())}")
lines.append(f"  current 0-image RIDs getting images: {int((impact.current_images == 0).sum())}")
lines.append(f"  current <=1-image RIDs getting images: {int((impact.current_images <= 1).sum())}")
lines.append("")
lines.append("Projected image count distribution among candidate RIDs:")
lines.append(impact["projected_images"].describe().to_string())

out_txt.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nWrote:", out_txt)
