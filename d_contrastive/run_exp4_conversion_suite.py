#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from experiment_utils import (
    DCON,
    auc_from_scores,
    bootstrap_auc,
    build_censored_conversion_cohort,
    load_raw_combat_data,
    paired_bootstrap_delta,
    scaled_features,
)


EUCLIDEAN_DIR = DCON / "results_plain_euclidean_05152016_s60_c150_h384_l256_cls_tasks"
N_BOOT = 200


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) < 5 or len(np.unique(y)) < 2 or np.isnan(score).any():
        return np.nan
    return float(roc_auc_score(y, score))


def fit_binary_score(x_train, train_cohort, x_test, test_cohort):
    tr_idx = train_cohort["feature_idx"].astype(int).values
    te_idx = test_cohort["feature_idx"].astype(int).values
    y_train = train_cohort["converted"].astype(int).values
    if len(y_train) < 10 or y_train.sum() < 3 or (len(y_train) - y_train.sum()) < 3:
        return np.full(len(te_idx), np.nan)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_train[tr_idx], y_train)
    return clf.predict_proba(x_test[te_idx])[:, 1]


def conversion_row(method, input_version, task, horizon, test_cohort, score, direct_score=None):
    y = test_cohort["converted"].astype(int).values
    auc = safe_auc(y, score)
    _, ci_lo, ci_hi = bootstrap_auc(y, score, n_boot=N_BOOT) if np.isfinite(auc) else (np.nan, np.nan, np.nan)
    row = {
        "method": method,
        "input_version": input_version,
        "task": task,
        "horizon_years": horizon,
        "n": len(test_cohort),
        "n_converters": int(y.sum()),
        "n_nonconverters": int(len(y) - y.sum()),
        "auc": auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }
    if direct_score is not None and np.isfinite(auc) and np.isfinite(safe_auc(y, direct_score)):
        delta, lo, hi, p = paired_bootstrap_delta(y, direct_score, score, n_boot=N_BOOT)
        row.update({
            "delta_vs_direct": delta,
            "delta_ci_lo": lo,
            "delta_ci_hi": hi,
            "delta_p": p,
        })
    else:
        row.update({
            "delta_vs_direct": np.nan,
            "delta_ci_lo": np.nan,
            "delta_ci_hi": np.nan,
            "delta_p": np.nan,
        })
    return row


def time_to_conversion_rows(input_version, x_train, x_test, z_train, z_test, train_meta, test_meta):
    rows = []
    train_cohort = build_censored_conversion_cohort(
        train_meta, "train", "MCI_to_AD_4y", "MCI", "AD", horizon_years=4
    )
    test_cohort = build_censored_conversion_cohort(
        test_meta, "test", "MCI_to_AD_4y", "MCI", "AD", horizon_years=4
    )
    train_conv = train_cohort[(train_cohort["converted"] == 1) & train_cohort["days_to_conversion"].notna()]
    test_conv = test_cohort[(test_cohort["converted"] == 1) & test_cohort["days_to_conversion"].notna()]
    if len(train_conv) < 5 or len(test_conv) < 3:
        return rows

    methods = {
        "ridge_d_hat": None,
        "euclidean_contrastive_z_setup3": (z_train, z_test),
        "direct_768_ridge_time": (x_train, x_test),
        "oracle_true_d_mod3": "oracle",
    }
    for method, arrays in methods.items():
        tr_idx = train_conv["feature_idx"].astype(int).values
        te_idx = test_conv["feature_idx"].astype(int).values
        y_train_days = train_conv["days_to_conversion"].astype(float).values
        y_test_days = test_conv["days_to_conversion"].astype(float).values
        if arrays == "oracle":
            score = -test_conv["d_mod3"].astype(float).values
            pred_days = score
        elif arrays is None:
            model = Ridge(alpha=1.0)
            model.fit(x_train[tr_idx], train_conv["d_mod3"].astype(float).values)
            train_dhat = model.predict(x_train[tr_idx]).reshape(-1, 1)
            test_dhat = model.predict(x_test[te_idx]).reshape(-1, 1)
            model2 = Ridge(alpha=1.0).fit(train_dhat, y_train_days)
            pred_days = model2.predict(test_dhat)
            score = -test_dhat.ravel()
        else:
            atr, ate = arrays
            model = Ridge(alpha=1.0).fit(atr[tr_idx], y_train_days)
            pred_days = model.predict(ate[te_idx])
            score = -pred_days
        rows.append({
            "method": method,
            "input_version": input_version,
            "task": "MCI_time_to_AD_4y_converters",
            "horizon_years": 4,
            "n": len(test_conv),
            "n_converters": len(test_conv),
            "n_nonconverters": 0,
            "auc": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "delta_vs_direct": np.nan,
            "delta_ci_lo": np.nan,
            "delta_ci_hi": np.nan,
            "delta_p": np.nan,
            "r2_days": r2_score(y_test_days, pred_days) if len(test_conv) > 1 else np.nan,
            "pearson_score_neg_days": pearsonr(score, -y_test_days)[0] if len(test_conv) > 1 else np.nan,
            "spearman_score_neg_days": spearmanr(score, -y_test_days)[0] if len(test_conv) > 1 else np.nan,
        })
    return rows


def main():
    print(f"Running EXP-4 conversion suite with N_BOOT={N_BOOT}", flush=True)
    data_by_version = load_raw_combat_data()
    rows = []
    cohort_rows = []
    for input_version, data in data_by_version.items():
        x_train, x_test = scaled_features(data)
        z_train = np.load(EUCLIDEAN_DIR / f"setup3_contrastive_z_train_{input_version}.npy")
        z_test = np.load(EUCLIDEAN_DIR / f"setup3_contrastive_z_test_{input_version}.npy")
        train_meta = data["train_meta"]
        test_meta = data["test_meta"]

        ridge = Ridge(alpha=1.0).fit(x_train, train_meta["d_mod3"].astype(float).values)
        dhat_test = ridge.predict(x_test)
        oracle_test = test_meta["d_mod3"].astype(float).values

        for baseline_dx, target_dx, prefix in [
            ("MCI", "AD", "MCI_to_AD"),
            ("NORMAL", "MCI", "CN_to_MCI"),
        ]:
            for horizon in [1, 2, 3, 4]:
                task = f"{prefix}_{horizon}y"
                train_cohort = build_censored_conversion_cohort(
                    train_meta, "train", task, baseline_dx, target_dx, horizon_years=horizon
                )
                test_cohort = build_censored_conversion_cohort(
                    test_meta, "test", task, baseline_dx, target_dx, horizon_years=horizon
                )
                for split, cohort in [("train", train_cohort), ("test", test_cohort)]:
                    valid = cohort[cohort["valid_censored"].astype(bool)]
                    excluded = (~cohort["valid_censored"].astype(bool)).sum()
                    cohort_rows.append({
                        "input_version": input_version,
                        "split": split,
                        "task": task,
                        "horizon_years": horizon,
                        "n_before_censor": len(cohort),
                        "converters_before_censor": int(cohort["converted"].sum()) if len(cohort) else 0,
                        "n_after_censor": len(valid),
                        "converters_after_censor": int(valid["converted"].sum()) if len(valid) else 0,
                        "nonconverters_after_censor": int(len(valid) - valid["converted"].sum()) if len(valid) else 0,
                        "n_excluded_short_followup": int(excluded) if len(cohort) else 0,
                    })
                train_valid = train_cohort[train_cohort["valid_censored"].astype(bool)].copy()
                test_valid = test_cohort[test_cohort["valid_censored"].astype(bool)].copy()
                if len(test_valid) < 5 or test_valid["converted"].nunique() < 2:
                    continue

                direct_score = fit_binary_score(x_train, train_valid, x_test, test_valid)
                y = test_valid["converted"].astype(int).values
                if np.isnan(direct_score).all():
                    direct_score = np.full(len(test_valid), np.nan)

                method_scores = {
                    "direct_logistic_sparse_labels": direct_score,
                    "ridge_d_hat": dhat_test[test_valid["feature_idx"].astype(int).values],
                    "oracle_true_d_mod3": oracle_test[test_valid["feature_idx"].astype(int).values],
                    "euclidean_contrastive_z_setup3": fit_binary_score(z_train, train_valid, z_test, test_valid),
                }
                direct_for_delta = method_scores["direct_logistic_sparse_labels"]
                for method, score in method_scores.items():
                    rows.append(conversion_row(
                        method, input_version, task, horizon, test_valid, score,
                        direct_score=None if method == "direct_logistic_sparse_labels" else direct_for_delta,
                    ))

        rows.extend(time_to_conversion_rows(input_version, x_train, x_test, z_train, z_test, train_meta, test_meta))

    out = pd.DataFrame(rows)
    cohort_out = pd.DataFrame(cohort_rows).drop_duplicates(["split", "task", "horizon_years"])
    out_path = DCON / "exp4_conversion_task_suite.csv"
    cohort_path = DCON / "exp4_conversion_cohort_counts.csv"
    out.to_csv(out_path, index=False)
    cohort_out.to_csv(cohort_path, index=False)

    print(f"Saved {out_path}")
    print(f"Saved {cohort_path}")
    print("\n=== EXP-4 binary AUC summary ===")
    cols = ["input_version", "task", "method", "n", "n_converters", "auc", "ci_lo", "ci_hi", "delta_vs_direct", "delta_p"]
    binary = out[out["task"].str.contains("_to_")].copy()
    print(binary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== EXP-4 cohort counts ===")
    print(cohort_out.to_string(index=False))


if __name__ == "__main__":
    main()
