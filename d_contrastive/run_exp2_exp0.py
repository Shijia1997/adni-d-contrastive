import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from experiment_utils import (
    DCON,
    auc_from_scores,
    build_all_censored_cohorts,
    bootstrap_auc,
    fit_logistic_conversion_score,
    get_valid_cohort,
    load_raw_combat_data,
    paired_bootstrap_delta,
    regression_metrics,
    scaled_features,
)


EUCLIDEAN_DIR = DCON / "results_rank_euclidean_a10_05152016_s60_c150_h384_l256"


def cohort_report(cohorts):
    rows = []
    for (split, task), group in cohorts.groupby(["split", "task"]):
        valid_mask = group["valid_censored"].astype(bool)
        excluded_mask = group["excluded_for_short_followup"].astype(bool)
        valid = group[valid_mask]
        excluded = group[excluded_mask]
        rows.append({
            "split": split,
            "task": task,
            "n_before_censor": len(group),
            "converters_before_censor": int(group["converted"].sum()),
            "n_after_censor": len(valid),
            "converters_after_censor": int(valid["converted"].sum()),
            "nonconverters_after_censor": int((valid["converted"] == 0).sum()),
            "n_excluded_short_followup": len(excluded),
        })
    return pd.DataFrame(rows)


def score_ci_row(method, input_version, task, score, cohort, baseline_score=None):
    idx = cohort["feature_idx"].astype(int).values
    y = cohort["converted"].astype(int).values
    auc, lo, hi = auc_from_scores(score, cohort)
    row = {
        "method": method,
        "input_version": input_version,
        "task": task,
        "n": len(cohort),
        "n_converters": int(y.sum()),
        "auc": auc,
        "ci_lo": lo,
        "ci_hi": hi,
    }
    if baseline_score is not None and np.isfinite(auc):
        d, dlo, dhi, p = paired_bootstrap_delta(y, np.asarray(baseline_score)[idx], np.asarray(score)[idx])
        row.update({"delta_vs_direct": d, "delta_ci_lo": dlo, "delta_ci_hi": dhi, "delta_p": p})
    return row


def main():
    data_by_version = load_raw_combat_data()
    cohorts = build_all_censored_cohorts(data_by_version, horizon_years=2)
    cohorts.to_csv(DCON / "conversion_cohorts_censored.csv", index=False)
    report = cohort_report(cohorts)
    report.to_csv(DCON / "exp2_censoring_report.csv", index=False)

    print("\n=== EXP-2 censoring report ===")
    print(report.to_string(index=False))

    rows = []
    ridge_rows = []
    tasks = ["MCI_to_AD_2y", "CN_to_MCI_2y"]

    for version, data in data_by_version.items():
        x_train, x_test = scaled_features(data)
        y_train_d = data["train_meta"]["d_mod3"].values
        y_test_d = data["test_meta"]["d_mod3"].values

        ridge = Ridge(alpha=1.0)
        ridge.fit(x_train, y_train_d)
        d_hat_train = ridge.predict(x_train)
        d_hat_test = ridge.predict(x_test)
        ridge_metric = {"input_version": version, "method": "ridge_d_hat"}
        ridge_metric.update(regression_metrics(y_test_d, d_hat_test))
        ridge_rows.append(ridge_metric)

        z_train = np.load(EUCLIDEAN_DIR / f"setup3_contrastive_z_train_{version}.npy")
        z_test = np.load(EUCLIDEAN_DIR / f"setup3_contrastive_z_test_{version}.npy")

        for task in tasks:
            train_cohort = get_valid_cohort(cohorts, "train", task)
            test_cohort = get_valid_cohort(cohorts, "test", task)

            # Direct sparse-label logistic on frozen features.
            p_direct = fit_logistic_conversion_score(x_train, train_cohort, x_test, test_cohort)
            direct_score_full = np.full(len(x_test), np.nan)
            direct_score_full[test_cohort["feature_idx"].astype(int).values] = p_direct
            rows.append(score_ci_row("direct_logistic_sparse_labels", version, task, direct_score_full, test_cohort))

            # Ridge d-hat score.
            rows.append(score_ci_row("ridge_d_hat", version, task, d_hat_test, test_cohort, baseline_score=direct_score_full))

            # Oracle true d_mod3 score.
            rows.append(score_ci_row("oracle_true_d_mod3", version, task, y_test_d, test_cohort, baseline_score=direct_score_full))

            # Existing Euclidean contrastive z Setup3 sparse-label logistic.
            p_z = fit_logistic_conversion_score(z_train, train_cohort, z_test, test_cohort)
            z_score_full = np.full(len(z_test), np.nan)
            z_score_full[test_cohort["feature_idx"].astype(int).values] = p_z
            rows.append(score_ci_row("euclidean_contrastive_z_setup3", version, task, z_score_full, test_cohort, baseline_score=direct_score_full))

    exp0 = pd.DataFrame(rows)
    exp0 = exp0.merge(pd.DataFrame(ridge_rows), on=["input_version", "method"], how="left")
    exp0.to_csv(DCON / "exp0_dhat_vs_contrastive_conversion.csv", index=False)

    print("\n=== EXP-0 d_hat vs contrastive conversion ===")
    print(exp0.to_string(index=False))
    print("\nSaved:")
    print(DCON / "conversion_cohorts_censored.csv")
    print(DCON / "exp2_censoring_report.csv")
    print(DCON / "exp0_dhat_vs_contrastive_conversion.csv")


if __name__ == "__main__":
    main()
