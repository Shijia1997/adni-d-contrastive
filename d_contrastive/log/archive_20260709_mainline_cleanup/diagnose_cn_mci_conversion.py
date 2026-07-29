import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_v0_contrastive import (  # noqa: E402
    build_conversion_labels,
    first_dx_image_indices,
    load_features_and_labels,
)


FEATURES_DIR = Path("/dcs07/zwang/data/adni_d/data/embeddings_128_05152016")
D_CSV = Path("/dcs07/zwang/data/adni_d/data/master_smri_05152016/D_with_image_paths_full.csv")
RESULTS_DIR = Path(
    "/dcs07/zwang/data/adni_d/d_contrastive/"
    "results_plain_euclidean_05152016_s60_c150_h384_l256"
)


def bootstrap_auc(y, pred, rng, n_boot=10000):
    y = np.asarray(y)
    pred = np.asarray(pred)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], pred[idx]))
    aucs = np.asarray(aucs)
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
        len(aucs),
    )


def bootstrap_delta(y, pred_a, pred_b, rng, n_boot=10000):
    y = np.asarray(y)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], pred_b[idx]) - roc_auc_score(y[idx], pred_a[idx]))
    deltas = np.asarray(deltas)
    p_two_sided = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        float(p_two_sided),
        len(deltas),
    )


def cn_mci_info(meta):
    conv = build_conversion_labels(meta, baseline_dx="NORMAL", target_dx="MCI")
    idx, y = first_dx_image_indices(meta, conv, baseline_dx="NORMAL")
    sub = meta.loc[idx].copy()
    sub["feature_idx"] = np.asarray(idx, dtype=int)
    sub["converted_2y"] = np.asarray(y, dtype=int)

    timing_rows = []
    for rid, group in meta.sort_values(["RID", "EXAMDATE.x"]).groupby("RID"):
        first = group.iloc[0]
        if first["dx"] != "NORMAL" or rid not in conv:
            continue
        target_date = first["EXAMDATE.x"] + pd.DateOffset(years=2)
        follow = group[(group["EXAMDATE.x"] > first["EXAMDATE.x"]) & (group["EXAMDATE.x"] <= target_date)]
        mci = follow[follow["dx"] == "MCI"]
        days_to_mci = np.nan
        if len(mci):
            days_to_mci = (mci.iloc[0]["EXAMDATE.x"] - first["EXAMDATE.x"]).days
        max_fu = np.nan
        if len(follow):
            max_fu = (follow["EXAMDATE.x"].max() - first["EXAMDATE.x"]).days
        timing_rows.append({"RID": rid, "days_to_mci": days_to_mci, "max_fu_days_2y": max_fu})
    timing = pd.DataFrame(timing_rows)
    return sub.merge(timing, on="RID", how="left")


def print_d_mod3_group_stats(split_name, sub):
    stable = sub.loc[sub.converted_2y == 0, "d_mod3"].dropna()
    conv = sub.loc[sub.converted_2y == 1, "d_mod3"].dropna()
    t_p = ttest_ind(conv, stable, equal_var=False).pvalue if len(conv) > 1 and len(stable) > 1 else np.nan
    u_p = mannwhitneyu(conv, stable, alternative="two-sided").pvalue if len(conv) and len(stable) else np.nan
    print(
        f"{split_name} d_mod3: stable mean={stable.mean():+.3f} sd={stable.std():.3f} N={len(stable)}; "
        f"converter mean={conv.mean():+.3f} sd={conv.std():.3f} N={len(conv)}; "
        f"Welch p={t_p:.4g}; MW p={u_p:.4g}"
    )


def main():
    rng = np.random.default_rng(42)
    for version in ["raw", "combat"]:
        print("\n" + "=" * 80)
        print(f"CN->MCI DIAGNOSTIC: {version}")
        print("=" * 80)

        data = load_features_and_labels(FEATURES_DIR, D_CSV, version=version)
        train_meta = data["train_meta"]
        test_meta = data["test_meta"]
        train_cn = cn_mci_info(train_meta)
        test_cn = cn_mci_info(test_meta)

        print(
            "Definition: first available image/D row per RID must be baseline dx=NORMAL; "
            "converted=any MCI after baseline EXAMDATE and <= baseline + DateOffset(years=2)."
        )
        print(
            f"Train CN baseline cohort: N={len(train_cn)}, converters={int(train_cn['converted_2y'].sum())}, "
            f"non-converters={int((train_cn['converted_2y'] == 0).sum())}"
        )
        print(
            f"Test CN baseline cohort:  N={len(test_cn)}, converters={int(test_cn['converted_2y'].sum())}, "
            f"non-converters={int((test_cn['converted_2y'] == 0).sum())}"
        )

        conv_days = test_cn.loc[test_cn.converted_2y == 1, "days_to_mci"].dropna()
        stable_fu = test_cn.loc[test_cn.converted_2y == 0, "max_fu_days_2y"].dropna()
        print(
            f"Test converters days-to-MCI: median={conv_days.median():.1f}, "
            f"min={conv_days.min():.0f}, max={conv_days.max():.0f}, N={len(conv_days)}"
        )
        print(
            f"Test stable max follow-up within 2y: median={stable_fu.median():.1f}, "
            f"min={stable_fu.min():.0f}, max={stable_fu.max():.0f}, N={len(stable_fu)}"
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(data["train_features"])
        x_test = scaler.transform(data["test_features"])
        tr_idx = train_cn["feature_idx"].values.astype(int)
        te_idx = test_cn["feature_idx"].values.astype(int)
        y_train = train_cn["converted_2y"].values.astype(int)
        y_test = test_cn["converted_2y"].values.astype(int)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_train[tr_idx], y_train)
        p_direct = clf.predict_proba(x_test[te_idx])[:, 1]

        z_train = np.load(RESULTS_DIR / f"setup3_contrastive_z_train_{version}.npy")
        z_test = np.load(RESULTS_DIR / f"setup3_contrastive_z_test_{version}.npy")
        clf3 = LogisticRegression(max_iter=2000, C=1.0)
        clf3.fit(z_train[tr_idx], y_train)
        p_s3 = clf3.predict_proba(z_test[te_idx])[:, 1]

        auc_direct = roc_auc_score(y_test, p_direct)
        auc_s3 = roc_auc_score(y_test, p_s3)
        print(f"AUC direct Setup1: {auc_direct:.3f}")
        print(f"AUC contrastive Setup3: {auc_s3:.3f}")

        mean1, lo1, hi1, nb1 = bootstrap_auc(y_test, p_direct, rng)
        mean3, lo3, hi3, nb3 = bootstrap_auc(y_test, p_s3, rng)
        mean_delta, lo_delta, hi_delta, p_delta, nb_delta = bootstrap_delta(y_test, p_direct, p_s3, rng)
        print(f"Bootstrap direct AUC mean/95%CI: {mean1:.3f} [{lo1:.3f}, {hi1:.3f}] (B={nb1})")
        print(f"Bootstrap setup3 AUC mean/95%CI: {mean3:.3f} [{lo3:.3f}, {hi3:.3f}] (B={nb3})")
        print(
            f"Paired bootstrap delta setup3-direct: {mean_delta:+.3f} "
            f"[{lo_delta:+.3f}, {hi_delta:+.3f}], two-sided p~{p_delta:.4f} (B={nb_delta})"
        )

        print_d_mod3_group_stats("train", train_cn)
        print_d_mod3_group_stats("test", test_cn)


if __name__ == "__main__":
    main()
