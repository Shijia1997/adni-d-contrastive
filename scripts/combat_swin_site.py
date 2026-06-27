#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat, neuroCombatFromTraining
from scipy.stats import f_oneway, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def image_col(df):
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    raise KeyError("No image id column found")


def add_subject_and_site(df, image_meta):
    col = image_col(df)
    out = df.copy()
    meta = image_meta[["Image Data ID", "Subject"]].drop_duplicates("Image Data ID")
    out = out.merge(meta, left_on=col, right_on="Image Data ID", how="left", suffixes=("", "_meta"))
    if out["Subject"].isna().any():
        missing = out.loc[out["Subject"].isna(), col].drop_duplicates().head(10).tolist()
        raise SystemExit(f"STOP: missing Subject for image ids: {missing}")
    out["ptid_site"] = out["Subject"].astype(str).str.split("_").str[0].str.zfill(3)
    return out


def mri3meta_diagnostic(mri3_path, matched):
    if not Path(mri3_path).exists():
        print(f"MRI3META not found: {mri3_path}")
        return
    meta = pd.read_csv(mri3_path)
    print("\nMRI3META diagnostic")
    print(f"  rows/cols: {meta.shape}")
    if "FIELD_STRENGTH" in meta.columns:
        print("  FIELD_STRENGTH counts:")
        print(meta["FIELD_STRENGTH"].value_counts(dropna=False).to_string())
    if "SITEID" in meta.columns:
        print(f"  MRI3META SITEID unique: {meta['SITEID'].nunique(dropna=True)}")
    if {"PTID", "SITEID"}.issubset(meta.columns):
        site_map = meta[["PTID", "SITEID"]].dropna().drop_duplicates("PTID")
        tmp = matched[["Subject", "ptid_site"]].drop_duplicates().merge(site_map, left_on="Subject", right_on="PTID", how="left")
        print(f"  matched subjects with MRI3META SITEID: {tmp['SITEID'].notna().sum()} / {len(tmp)}")
        have_site = tmp[tmp["SITEID"].notna()].copy()
        siteid_str = pd.to_numeric(have_site["SITEID"], errors="coerce").astype("Int64").astype(str).str.zfill(3)
        disagree = have_site[have_site["ptid_site"].astype(str).values != siteid_str.values]
        print(f"  PTID-prefix site differs from MRI3META SITEID for subjects: {len(disagree)} / {len(have_site)}")
        print("  Note: using PTID prefix as requested for batch, not MRI3META SITEID.")


def choose_cutoff(train_unique):
    site_counts = train_unique["ptid_site"].value_counts()
    print("\nStep A: Site distribution")
    print(f"Total unique PTID-prefix sites in train: {site_counts.size}")
    print("Top 20 sites by image count:")
    print(site_counts.head(20).to_string())
    print("\nSite size distribution:")
    print(site_counts.describe().to_string())
    for k in [50, 30, 20]:
        print(f"Sites with >={k} images: {(site_counts >= k).sum()}")
    print(f"Sites with <10 images: {(site_counts < 10).sum()}")

    n30 = int((site_counts >= 30).sum())
    cutoff = 30
    if n30 < 5 or n30 > 25:
        candidates = sorted(set(site_counts.values), reverse=True)
        best = None
        for c in candidates:
            n = int((site_counts >= c).sum()) + 1
            if 8 <= n <= 15:
                best = c
                break
        if best is None:
            ranked = [(abs((int((site_counts >= c).sum()) + 1) - 12), c) for c in candidates]
            best = sorted(ranked)[0][1]
        cutoff = int(best)
    big_sites = set(site_counts[site_counts >= cutoff].index.astype(str))
    print(f"\nSelected site cutoff: >={cutoff} train images")
    print(f"Big site count: {len(big_sites)}")
    return cutoff, big_sites


def assign_batches(df, big_sites):
    out = df.copy()
    out["batch"] = out["ptid_site"].map(lambda s: f"s{s}" if str(s) in big_sites else "other_small")
    return out


def print_batch_distribution(train, test):
    train_unique = train.drop_duplicates(image_col(train))
    test_unique = test.drop_duplicates(image_col(test))
    train_counts = train_unique["batch"].value_counts()
    test_counts = test_unique["batch"].value_counts()
    print("\nFinal batch distribution, train unique images:")
    print(train_counts.to_string())
    print("\nFinal batch distribution, test unique images:")
    print(test_counts.to_string())
    n_batches = train_counts.size
    small = train_counts[train_counts < 10]
    if n_batches < 5:
        raise SystemExit(f"STOP: final batch count <5: {n_batches}")
    if n_batches > 25:
        raise SystemExit(f"STOP: final batch count >25: {n_batches}")
    if len(small):
        raise SystemExit(f"STOP: final train batches <10 images:\n{small.to_string()}")
    cold = set(test_counts.index) - set(train_counts.index)
    if cold:
        print(f"WARNING: test batches not present in train: {sorted(cold)}")
    return train_counts, test_counts


def latent_for_unique(df, image_ids, swin):
    col = image_col(df)
    id_to_idx = {str(iid): i for i, iid in enumerate(image_ids.astype(str))}
    unique = df.drop_duplicates(col).copy()
    unique["_iid_str"] = unique[col].astype(str)
    unique["latent_idx"] = unique["_iid_str"].map(id_to_idx)
    unique = unique.dropna(subset=["latent_idx"]).copy()
    unique["latent_idx"] = unique["latent_idx"].astype(int)
    X = swin[unique["latent_idx"].values]
    return unique, X


def batch_auc(X, y, label):
    counts = pd.Series(y).value_counts()
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        print(f"{label} batch AUC skipped: too few samples per class")
        return np.array([np.nan])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, multi_class="ovr", class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc_ovr")
    print(f"{label} batch (PTID-site) AUC (OvR): {scores.mean():.3f} +/- {scores.std():.3f}")
    print("  folds:", ", ".join(f"{s:.3f}" for s in scores))
    return scores


def anova_site_effect(X, y_batch):
    pvals = []
    batches = np.unique(y_batch)
    for k in range(X.shape[1]):
        groups = [X[y_batch == b, k] for b in batches if (y_batch == b).sum() > 5]
        if len(groups) >= 2:
            try:
                _, p = f_oneway(*groups)
            except Exception:
                p = 1.0
        else:
            p = 1.0
        pvals.append(p)
    pvals = np.asarray(pvals)
    n_sig = int((pvals < 0.05 / X.shape[1]).sum())
    print(f"Dim with site effect (Bonferroni p<0.05/{X.shape[1]}): {n_sig}/{X.shape[1]}")
    return pvals


def sex_value(v):
    s = str(v).strip().upper()
    if s.startswith("M") or s == "1":
        return "M"
    return "F"


def dx_value(v):
    return {"NORMAL": 0, "MCI": 1, "AD": 2}.get(str(v), 1)


def select_age_col(df):
    for c in ["ageori", "Age", "AGE", "age"]:
        if c in df.columns:
            return c
    raise KeyError("No age column found")


def combat_inputs(df, swin, image_ids, split_name):
    col = image_col(df)
    id_to_idx = {str(iid): i for i, iid in enumerate(image_ids.astype(str))}
    age_col = select_age_col(df)
    rows = []
    latent = []
    for iid, sub in df.sort_values(["scan_date", "EXAMDATE.x"]).groupby(col, sort=False):
        iid_str = str(iid)
        if iid_str not in id_to_idx:
            continue
        first = sub.iloc[0]
        if pd.isna(first[age_col]) or pd.isna(first["d_mod3"]) or pd.isna(first["batch"]):
            continue
        rows.append(
            {
                "Image_Data_ID": iid_str,
                "batch": first["batch"],
                "age": float(first[age_col]),
                "sex": sex_value(first.get("PTGENDER", first.get("sex", ""))),
                "dx": dx_value(first.get("dx", "MCI")),
                "D": float(first["d_mod3"]),
            }
        )
        latent.append(swin[id_to_idx[iid_str]])
    cov = pd.DataFrame(rows)
    X = np.asarray(latent, dtype=np.float32)
    print(f"{split_name} ComBat image-level input: X={X.shape}, covars={cov.shape}, age_col={age_col}")
    print(cov.head().to_string(index=False))
    if cov.empty:
        raise SystemExit(f"STOP: no {split_name} images available for ComBat")
    return cov, X


def cn_ad_auc(df_with_batch, X_h, ids_ordered, label):
    col = image_col(df_with_batch)
    iid_to_idx = {iid: i for i, iid in enumerate(ids_ordered)}
    base = (
        df_with_batch.dropna(subset=["EXAMDATE.x", "dx", col])
        .sort_values(["RID", "EXAMDATE.x"])
        .groupby("RID", as_index=False)
        .first()
    )
    base = base[base["dx"].isin(["NORMAL", "AD"])].copy()
    base["_iid"] = base[col].astype(str)
    base["combat_idx"] = base["_iid"].map(iid_to_idx)
    base = base.dropna(subset=["combat_idx"]).copy()
    X = X_h[base["combat_idx"].astype(int).values]
    y = (base["dx"].values == "AD").astype(int)
    counts = pd.Series(base["dx"]).value_counts()
    n_splits = min(5, int(counts.min()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"{label} CN-vs-AD AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    print("  folds:", ", ".join(f"{s:.3f}" for s in scores))
    return scores, base, X


def d_spearman(X, y_d, label):
    corrs = []
    for k in range(X.shape[1]):
        rho = spearmanr(X[:, k], y_d).statistic
        corrs.append(abs(rho) if np.isfinite(rho) else 0.0)
    corrs = np.asarray(corrs)
    print(f"{label} top 10 |Spearman| with D:")
    print(np.sort(corrs)[-10:][::-1])
    print(f"{label} Dim |rho|>0.2/0.3/0.4: {(corrs>0.2).sum()}/{(corrs>0.3).sum()}/{(corrs>0.4).sum()}")
    return corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/master_smri/matched_TRAIN.csv")
    parser.add_argument("--test_csv", default="data/master_smri/matched_TEST.csv")
    parser.add_argument("--image_meta", default="data/master_smri/smri_image_metadata_used.csv")
    parser.add_argument("--mri3meta", default="data/MRI3META_13May2026.csv")
    parser.add_argument("--latent", default="data/embeddings/swin_latent.npy")
    parser.add_argument("--image_ids", default="data/embeddings/image_id_order.npy")
    parser.add_argument("--out_dir", default="data/embeddings")
    parser.add_argument("--pre_cn_ad_auc", type=float, default=0.799178)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    image_meta = pd.read_csv(args.image_meta)
    train = add_subject_and_site(train, image_meta)
    test = add_subject_and_site(test, image_meta)
    mri3meta_diagnostic(args.mri3meta, pd.concat([train, test], ignore_index=True))

    train_unique = train.drop_duplicates(image_col(train)).copy()
    cutoff, big_sites = choose_cutoff(train_unique)
    top10 = train_unique["ptid_site"].value_counts().head(10).index
    print("\nSite x dx (top 10 PTID-prefix sites, train unique images):")
    print(pd.crosstab(train_unique[train_unique["ptid_site"].isin(top10)]["ptid_site"], train_unique[train_unique["ptid_site"].isin(top10)]["dx"]).to_string())

    train = assign_batches(train, big_sites)
    test = assign_batches(test, big_sites)
    train_counts, test_counts = print_batch_distribution(train, test)

    image_ids = np.load(args.image_ids, allow_pickle=True).astype(str)
    swin = np.load(args.latent)

    train_unique, X_pre = latent_for_unique(train, image_ids, swin)
    y_batch = train_unique["batch"].values
    print("\nStep B: pre-ComBat batch signal")
    pre_batch_scores = batch_auc(X_pre, y_batch, "Pre-ComBat")
    anova_site_effect(X_pre, y_batch)

    print("\nStep C: run ComBat")
    cov_train, X_train = combat_inputs(train, swin, image_ids, "Train")
    combat_train = neuroCombat(
        dat=X_train.T,
        covars=cov_train[["batch", "age", "sex", "dx", "D"]],
        batch_col="batch",
        continuous_cols=["age", "D"],
        categorical_cols=["sex", "dx"],
        eb=True,
        parametric=True,
        mean_only=False,
    )
    X_train_h = combat_train["data"].T.astype(np.float32)
    train_ids = cov_train["Image_Data_ID"].astype(str).to_numpy()
    with open(out_dir / "combat_estimates.pkl", "wb") as f:
        pickle.dump(combat_train["estimates"], f)
    print(f"ComBat done. X_train_h shape: {X_train_h.shape}")

    cov_test, X_test = combat_inputs(test, swin, image_ids, "Test")
    train_batches = set(map(str, combat_train["estimates"]["batches"]))
    unseen = set(cov_test["batch"].astype(str)) - train_batches
    if unseen:
        print(f"WARNING: test batches not seen in train before remap: {sorted(unseen)}")
        cov_test["batch"] = cov_test["batch"].map(lambda b: "other_small" if str(b) in unseen else b)
    remaining_unseen = set(cov_test["batch"].astype(str)) - train_batches
    if remaining_unseen:
        raise SystemExit(f"STOP: test batches still unseen after remap: {sorted(remaining_unseen)}")
    combat_test = neuroCombatFromTraining(
        dat=X_test.T,
        batch=cov_test["batch"].astype(str).to_numpy(),
        estimates=combat_train["estimates"],
    )
    X_test_h = combat_test["data"].T.astype(np.float32)
    test_ids = cov_test["Image_Data_ID"].astype(str).to_numpy()

    np.save(out_dir / "swin_combat_train.npy", X_train_h)
    np.save(out_dir / "swin_combat_test.npy", X_test_h)
    np.save(out_dir / "swin_combat_train_ids.npy", train_ids)
    np.save(out_dir / "swin_combat_test_ids.npy", test_ids)
    train_with_batch = Path(args.train_csv).with_name("matched_TRAIN_with_batch.csv")
    test_with_batch = Path(args.test_csv).with_name("matched_TEST_with_batch.csv")
    train.to_csv(train_with_batch, index=False)
    test.to_csv(test_with_batch, index=False)

    print("\nStep D: verify ComBat")
    post_batch_scores = batch_auc(X_train_h, cov_train["batch"].values, "Post-ComBat")
    post_auc, base_post, X_base_post = cn_ad_auc(train, X_train_h, train_ids, "Post-ComBat")
    pre_drop = args.pre_cn_ad_auc - float(np.nanmean(post_auc))
    print(f"  Compare to pre-ComBat CN-vs-AD AUC {args.pre_cn_ad_auc:.3f}; drop={pre_drop:.3f}")
    if pre_drop > 0.10:
        raise SystemExit("STOP: Post-ComBat CN-vs-AD AUC dropped by >0.10; possible over-correction")
    d_spearman(X_base_post, base_post["d_mod3"].astype(float).values, "Post-ComBat")

    print("\nFinal summary")
    print(f"  Batch variable: PTID prefix site (Subject split before '_'), grouped with cutoff >= {cutoff}")
    print(f"  Final train batch count: {train_counts.size}")
    print(f"  Train unique images for ComBat: {len(train_ids)}")
    print(f"  Test unique images for ComBat: {len(test_ids)}")
    print(f"  Pre batch AUC mean: {np.nanmean(pre_batch_scores):.3f}")
    print(f"  Post batch AUC mean: {np.nanmean(post_batch_scores):.3f}")
    print(f"  Post CN-vs-AD AUC mean: {np.nanmean(post_auc):.3f}")
    print("  Outputs:")
    print(f"    {out_dir / 'swin_combat_train.npy'}")
    print(f"    {out_dir / 'swin_combat_test.npy'}")
    print(f"    {out_dir / 'swin_combat_train_ids.npy'}")
    print(f"    {out_dir / 'swin_combat_test_ids.npy'}")
    print(f"    {train_with_batch}")
    print(f"    {test_with_batch}")
    print("STOP: ComBat verification complete. Do not proceed to PCA in this script.")


if __name__ == "__main__":
    main()
