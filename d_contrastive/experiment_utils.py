from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from minimal_v0_contrastive import load_features_and_labels


ROOT = Path("/dcs07/zwang/data/adni_d")
DCON = ROOT / "d_contrastive"
FEATURES_DIR = ROOT / "data/embeddings_128_05152016"
D_CSV = ROOT / "data/master_smri_05152016/D_with_image_paths_full.csv"


def regression_metrics(y_true, pred, prefix=""):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    ss_res = ((y_true - pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        f"{prefix}r2_d_mod3": r2,
        f"{prefix}pearson_d_mod3": pearsonr(y_true, pred)[0] if len(y_true) > 1 else np.nan,
        f"{prefix}spearman_d_mod3": spearmanr(y_true, pred)[0] if len(y_true) > 1 else np.nan,
    }


def bootstrap_auc(y, score, n_boot=5000, seed=42):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    if not aucs:
        return np.nan, np.nan, np.nan
    aucs = np.asarray(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(y, score_base, score_method, n_boot=5000, seed=42):
    y = np.asarray(y, dtype=int)
    score_base = np.asarray(score_base)
    score_method = np.asarray(score_method)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], score_method[idx]) - roc_auc_score(y[idx], score_base[idx]))
    if not deltas:
        return np.nan, np.nan, np.nan, np.nan
    deltas = np.asarray(deltas)
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(p)


def build_censored_conversion_cohort(meta, split, task_name, baseline_dx, target_dx, horizon_years=2):
    rows = []
    meta = meta.sort_values(["RID", "EXAMDATE.x"]).copy()
    for rid, group in meta.groupby("RID"):
        first = group.iloc[0]
        if first["dx"] != baseline_dx:
            continue
        baseline_date = first["EXAMDATE.x"]
        horizon_date = baseline_date + pd.DateOffset(years=horizon_years)
        follow = group[(group["EXAMDATE.x"] > baseline_date) & (group["EXAMDATE.x"] <= horizon_date)]
        target_follow = follow[follow["dx"] == target_dx]
        converted = int(len(target_follow) > 0)
        max_followup_days = np.nan
        if len(group[group["EXAMDATE.x"] > baseline_date]):
            max_followup_days = int((group[group["EXAMDATE.x"] > baseline_date]["EXAMDATE.x"].max() - baseline_date).days)
        days_to_conversion = np.nan
        if converted:
            days_to_conversion = int((target_follow.iloc[0]["EXAMDATE.x"] - baseline_date).days)
        has_adequate_followup = bool(pd.notna(max_followup_days) and max_followup_days >= horizon_years * 365)
        valid = converted or has_adequate_followup
        rows.append({
            "split": split,
            "task": task_name,
            "RID": rid,
            "feature_idx": int(first.name),
            "Image Data ID": first.get("Image Data ID", first.get("Image_Data_ID", np.nan)),
            "baseline_dx": baseline_dx,
            "target_dx": target_dx,
            "baseline_date": baseline_date,
            "horizon_years": horizon_years,
            "converted": converted,
            "valid_censored": valid,
            "excluded_for_short_followup": not valid,
            "days_to_conversion": days_to_conversion,
            "max_followup_days": max_followup_days,
            "d_mod3": first["d_mod3"],
        })
    return pd.DataFrame(rows)


def build_all_censored_cohorts(data_by_version, horizon_years=2):
    rows = []
    # Cohorts depend only on metadata split, so raw/combat are identical. Use raw.
    data = data_by_version["raw"]
    tasks = [
        ("MCI_to_AD_2y", "MCI", "AD"),
        ("CN_to_MCI_2y", "NORMAL", "MCI"),
    ]
    for split, meta in [("train", data["train_meta"]), ("test", data["test_meta"])]:
        for task_name, baseline_dx, target_dx in tasks:
            rows.append(build_censored_conversion_cohort(
                meta, split, task_name, baseline_dx, target_dx, horizon_years=horizon_years
            ))
    return pd.concat(rows, ignore_index=True)


def get_valid_cohort(cohorts, split, task):
    sub = cohorts[(cohorts["split"] == split) & (cohorts["task"] == task) & (cohorts["valid_censored"])].copy()
    return sub


def auc_from_scores(score, cohort):
    idx = cohort["feature_idx"].astype(int).values
    y = cohort["converted"].astype(int).values
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, np.asarray(score)[idx])
    _, lo, hi = bootstrap_auc(y, np.asarray(score)[idx])
    return float(auc), lo, hi


def fit_logistic_conversion_score(x_train, train_cohort, x_test, test_cohort):
    tr_idx = train_cohort["feature_idx"].astype(int).values
    te_idx = test_cohort["feature_idx"].astype(int).values
    y_train = train_cohort["converted"].astype(int).values
    y_test = test_cohort["converted"].astype(int).values
    if len(y_train) < 10 or y_train.sum() < 3 or y_test.sum() < 3:
        return np.full(len(te_idx), np.nan)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_train[tr_idx], y_train)
    return clf.predict_proba(x_test[te_idx])[:, 1]


def evaluate_current_classification(x_train, train_meta, x_test, test_meta):
    out = {}
    pairs = [
        ("cn_ad_auc", "NORMAL", "AD"),
        ("cn_mci_cls_auc", "NORMAL", "MCI"),
        ("mci_ad_cls_auc", "MCI", "AD"),
    ]
    for key, neg, pos in pairs:
        tr = train_meta["dx"].isin([neg, pos]).values
        te = test_meta["dx"].isin([neg, pos]).values
        y_train = (train_meta.loc[tr, "dx"] == pos).astype(int).values
        y_test = (test_meta.loc[te, "dx"] == pos).astype(int).values
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            out[key] = np.nan
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_train[tr], y_train)
        out[key] = roc_auc_score(y_test, clf.predict_proba(x_test[te])[:, 1])

    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    y_train = train_meta["dx"].map(dx_map).fillna(-1).astype(int).values
    y_test = test_meta["dx"].map(dx_map).fillna(-1).astype(int).values
    tr = y_train >= 0
    te = y_test >= 0
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="ovr")
    clf.fit(x_train[tr], y_train[tr])
    p = clf.predict_proba(x_test[te])
    pred = clf.predict(x_test[te])
    aucs = []
    for c in range(3):
        try:
            aucs.append(roc_auc_score((y_test[te] == c).astype(int), p[:, c]))
        except Exception:
            pass
    out["3class_acc"] = accuracy_score(y_test[te], pred)
    out["3class_macro_auc"] = float(np.mean(aucs)) if aucs else np.nan
    return out


def load_raw_combat_data():
    return {
        "raw": load_features_and_labels(FEATURES_DIR, D_CSV, version="raw"),
        "combat": load_features_and_labels(FEATURES_DIR, D_CSV, version="combat"),
    }


def scaled_features(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data["train_features"]), scaler.transform(data["test_features"])
