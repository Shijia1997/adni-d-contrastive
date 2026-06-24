import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from minimal_v0_contrastive import (
    build_conversion_labels,
    first_dx_image_indices,
    load_features_and_labels,
)


ROOT = Path("/dcs07/zwang/data/adni_d/d_contrastive")
FEATURES_DIR = Path("/dcs07/zwang/data/adni_d/data/embeddings_128_05152016")
D_CSV = Path("/dcs07/zwang/data/adni_d/data/master_smri_05152016/D_with_image_paths_full.csv")


def bootstrap_auc(y, pred, seed=42, n_boot=2000):
    y = np.asarray(y)
    pred = np.asarray(pred)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], pred[idx]))
    if not aucs:
        return np.nan, np.nan, np.nan
    aucs = np.asarray(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def conversion_probe_ci(z_train, z_test, train_meta, test_meta, baseline_dx, target_dx):
    train_conv = build_conversion_labels(train_meta, baseline_dx=baseline_dx, target_dx=target_dx)
    test_conv = build_conversion_labels(test_meta, baseline_dx=baseline_dx, target_dx=target_dx)
    tr_idx, tr_y = first_dx_image_indices(train_meta, train_conv, baseline_dx=baseline_dx)
    te_idx, te_y = first_dx_image_indices(test_meta, test_conv, baseline_dx=baseline_dx)
    tr_y = np.asarray(tr_y, dtype=int)
    te_y = np.asarray(te_y, dtype=int)
    if len(tr_idx) < 10 or len(te_idx) < 5 or tr_y.sum() < 3 or te_y.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(z_train[tr_idx], tr_y)
    pred = clf.predict_proba(z_test[te_idx])[:, 1]
    auc = roc_auc_score(te_y, pred)
    _, lo, hi = bootstrap_auc(te_y, pred)
    return float(auc), lo, hi, int(te_y.sum())


def conversion_score_ci(score_test, test_meta, baseline_dx, target_dx):
    test_conv = build_conversion_labels(test_meta, baseline_dx=baseline_dx, target_dx=target_dx)
    te_idx, te_y = first_dx_image_indices(test_meta, test_conv, baseline_dx=baseline_dx)
    te_y = np.asarray(te_y, dtype=int)
    if len(te_idx) < 5 or te_y.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    pred = np.asarray(score_test)[te_idx]
    auc = roc_auc_score(te_y, pred)
    _, lo, hi = bootstrap_auc(te_y, pred)
    return float(auc), lo, hi, int(te_y.sum())


def config_label(row):
    mode = row.get("loss_mode")
    alpha = row.get("rank_alpha")
    lam = row.get("lambda_rank")
    if mode == "euclidean":
        return "euclidean"
    if mode == "rank_kendall":
        return f"rank_kendall_a{alpha:g}"
    if mode == "hybrid":
        return f"hybrid_a{alpha:g}_l{lam:g}"
    return str(mode)


def score_setup3(row):
    vals = []
    for c in ["mci_conv_auc", "cn_mci_conv_auc", "r2_d_mod3", "spearman_d_mod3"]:
        v = row.get(c, np.nan)
        if pd.notna(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def main():
    result_dirs = sorted(ROOT.glob("results_rank_*_05152016_s60_c150_h384_l256"))
    rows = []
    ci_rows = []
    data_cache = {}

    for result_dir in result_dirs:
        summary_path = result_dir / "summary_table.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["result_dir"] = str(result_dir)
        rows.append(df)

        for version in ["raw", "combat"]:
            if version not in data_cache:
                data_cache[version] = load_features_and_labels(FEATURES_DIR, D_CSV, version=version)
            data = data_cache[version]
            z_train_path = result_dir / f"setup3_contrastive_z_train_{version}.npy"
            z_test_path = result_dir / f"setup3_contrastive_z_test_{version}.npy"
            s_test_path = result_dir / f"setup3_progression_s_test_{version}.npy"
            if not z_train_path.exists() or not z_test_path.exists():
                continue
            z_train = np.load(z_train_path)
            z_test = np.load(z_test_path)
            cfg = df[(df.input_version == version) & (df.setup == "setup3")]
            cfg_label = config_label(cfg.iloc[0]) if len(cfg) else result_dir.name
            for baseline_dx, target_dx, task in [
                ("MCI", "AD", "mci_to_ad"),
                ("NORMAL", "MCI", "cn_to_mci"),
            ]:
                auc, lo, hi, n_conv = conversion_probe_ci(
                    z_train, z_test, data["train_meta"], data["test_meta"], baseline_dx, target_dx
                )
                ci_rows.append({
                    "config": cfg_label,
                    "result_dir": str(result_dir),
                    "input_version": version,
                    "setup": "setup3",
                    "task": task,
                    "auc": auc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "test_converters": n_conv,
                })
                if s_test_path.exists():
                    s_test = np.load(s_test_path)
                    auc, lo, hi, n_conv = conversion_score_ci(
                        s_test, data["test_meta"], baseline_dx, target_dx
                    )
                    ci_rows.append({
                        "config": cfg_label,
                        "result_dir": str(result_dir),
                        "input_version": version,
                        "setup": "setup3-s",
                        "task": task,
                        "auc": auc,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "test_converters": n_conv,
                    })

    if not rows:
        raise SystemExit("No rank result summaries found.")

    all_df = pd.concat(rows, ignore_index=True)
    all_df["config"] = all_df.apply(config_label, axis=1)
    all_df.to_csv(ROOT / "rank_sweep_all_results.csv", index=False)

    setup3 = all_df[all_df.setup == "setup3"].copy()
    setup3["selection_score"] = setup3.apply(score_setup3, axis=1)
    cfg_scores = setup3.groupby(["loss_mode", "config"], dropna=False)["selection_score"].mean().reset_index()

    selected = []
    for mode in ["euclidean", "rank_kendall", "hybrid"]:
        sub = cfg_scores[cfg_scores.loss_mode == mode]
        if len(sub):
            selected.append(sub.sort_values("selection_score", ascending=False).iloc[0]["config"])

    comparison = all_df[(all_df.setup == "setup3") & (all_df.config.isin(selected))].copy()
    comparison = comparison[[
        "config", "input_version", "cn_ad_auc", "cn_mci_cls_auc", "mci_ad_cls_auc",
        "3class_macro_auc", "mci_conv_auc", "cn_mci_conv_auc", "r2_d_mod3",
        "pearson_d_mod3", "spearman_d_mod3", "alignment_spearman",
        "centroid_alignment_spearman", "z_std_test", "z_active_dims_005",
    ]]
    comparison.to_csv(ROOT / "rank_sweep_best_comparison.csv", index=False)

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(ROOT / "rank_sweep_conversion_ci.csv", index=False)

    print("\n=== Rank Sweep Config Scores ===")
    print(cfg_scores.sort_values(["loss_mode", "selection_score"], ascending=[True, False]).to_string(index=False))
    print("\n=== Best Comparison: setup3 ===")
    print(comparison.to_string(index=False))
    print("\n=== Conversion Bootstrap CI: setup3/setup3-s ===")
    print(ci_df[ci_df["config"].isin(selected)].to_string(index=False))
    print("\nSaved:")
    print(ROOT / "rank_sweep_all_results.csv")
    print(ROOT / "rank_sweep_best_comparison.csv")
    print(ROOT / "rank_sweep_conversion_ci.csv")


if __name__ == "__main__":
    main()
