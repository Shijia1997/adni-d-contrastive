#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("data/embeddings")
MASTER_DIR = Path("data/master_smri")
K_VALUES = [5, 10, 15, 20, 30]
K_EVAL = [5, 10, 15, 20, 30, 50]


def image_col(df):
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    raise KeyError("No image id column found")


def safe_abs_spearman(x, y):
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        return 0.0
    return abs(float(rho))


def abs_spearman_all_dims(x, y):
    x_rank = rankdata(x, axis=0)
    y_rank = rankdata(y)
    x_rank = x_rank - x_rank.mean(axis=0, keepdims=True)
    y_rank = y_rank - y_rank.mean()
    x_norm = np.sqrt((x_rank * x_rank).sum(axis=0))
    y_norm = np.sqrt((y_rank * y_rank).sum())
    denom = x_norm * y_norm
    corr = np.zeros(x.shape[1], dtype=float)
    ok = denom > 0
    corr[ok] = (x_rank[:, ok].T @ y_rank) / denom[ok]
    corr[~np.isfinite(corr)] = 0.0
    return np.abs(corr)


def build_baseline(df):
    col = image_col(df)
    work = df.copy()
    work["EXAMDATE.x"] = pd.to_datetime(work["EXAMDATE.x"], errors="coerce")

    visit_med = (
        work.dropna(subset=["RID", "EXAMDATE.x", "d_mod3"])
        .groupby(["RID", "EXAMDATE.x"], as_index=False)["d_mod3"]
        .median()
    )
    baseline_visit = (
        visit_med.sort_values(["RID", "EXAMDATE.x"])
        .groupby("RID", as_index=False)
        .first()
    )

    baseline_rows = work.merge(
        baseline_visit[["RID", "EXAMDATE.x", "d_mod3"]],
        on=["RID", "EXAMDATE.x"],
        suffixes=("", "_baseline_median"),
    )
    keep_cols = ["RID", "EXAMDATE.x", col, "d_mod3_baseline_median", "dx"]
    if "PTID" in baseline_rows.columns:
        keep_cols.append("PTID")
    baseline = baseline_rows.drop_duplicates("RID")[keep_cols].rename(
        columns={col: "Image Data ID", "d_mod3_baseline_median": "d_mod3"}
    )
    return baseline.dropna(subset=["Image Data ID", "d_mod3"])


def main():
    print("Step H: Train-only D-correlated Swin dim selection")
    X_train = np.load(OUT_DIR / "swin_combat_train.npy")
    X_test = np.load(OUT_DIR / "swin_combat_test.npy")
    train_ids = np.load(OUT_DIR / "swin_combat_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load(OUT_DIR / "swin_combat_test_ids.npy", allow_pickle=True).astype(str)
    matched_train = pd.read_csv(MASTER_DIR / "matched_TRAIN_with_batch.csv")
    matched_test = pd.read_csv(MASTER_DIR / "matched_TEST_with_batch.csv")

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"train_ids: {train_ids.shape}, test_ids: {test_ids.shape}")
    if X_train.shape[0] != len(train_ids) or X_test.shape[0] != len(test_ids):
        raise SystemExit("STOP: latent rows do not match id order")

    train_bl = build_baseline(matched_train)
    test_bl = build_baseline(matched_test)

    iid_to_train_idx = {iid: i for i, iid in enumerate(train_ids)}
    iid_to_test_idx = {iid: i for i, iid in enumerate(test_ids)}
    train_bl["idx"] = train_bl["Image Data ID"].astype(str).map(iid_to_train_idx)
    test_bl["idx"] = test_bl["Image Data ID"].astype(str).map(iid_to_test_idx)
    train_bl = train_bl.dropna(subset=["idx"]).copy()
    test_bl = test_bl.dropna(subset=["idx"]).copy()
    train_bl["idx"] = train_bl["idx"].astype(int)
    test_bl["idx"] = test_bl["idx"].astype(int)

    Z_train_bl = X_train[train_bl["idx"].values]
    y_train_d = train_bl["d_mod3"].astype(float).values
    Z_test_bl = X_test[test_bl["idx"].values]
    y_test_d = test_bl["d_mod3"].astype(float).values

    print("\nStep H1: Baseline subsets")
    print(f"Train baseline: {len(train_bl)} RIDs, Z shape {Z_train_bl.shape}")
    print(f"Test baseline:  {len(test_bl)} RIDs, Z shape {Z_test_bl.shape}")
    if len(train_bl) < 100 or len(test_bl) < 50:
        raise SystemExit("STOP: baseline sample unexpectedly small")

    print("\nStep H2: Per-dim D correlation on train baseline")
    d_corr_train = abs_spearman_all_dims(Z_train_bl, y_train_d)
    print("Train per-dim |rho with D| distribution:")
    print(f"  Max: {d_corr_train.max():.3f}")
    print(f"  Top 5: {np.sort(d_corr_train)[-5:][::-1]}")
    print(f"  Top 20: {np.sort(d_corr_train)[-20:][::-1]}")
    print(f"  >0.15 / >0.20: {(d_corr_train > 0.15).sum()} / {(d_corr_train > 0.20).sum()}")

    print("\nStep H3: Bootstrap stability of top-K selection")
    n_boot = 100
    rng = np.random.default_rng(42)
    stability_k = {}
    pick_count_by_k = {k_value: np.zeros(X_train.shape[1], dtype=float) for k_value in K_VALUES}
    max_k = max(K_VALUES)
    for _ in range(n_boot):
        idx = rng.choice(len(Z_train_bl), len(Z_train_bl), replace=True)
        corr_b = abs_spearman_all_dims(Z_train_bl[idx], y_train_d[idx])
        top = np.argsort(-corr_b)[:max_k]
        for k_value in K_VALUES:
            pick_count_by_k[k_value][top[:k_value]] += 1
    for k_value in K_VALUES:
        stability_k[k_value] = pick_count_by_k[k_value] / n_boot

    for k_value in K_VALUES:
        stable_count = int((stability_k[k_value] >= 0.50).sum())
        high_stable = int((stability_k[k_value] >= 0.80).sum())
        top_stable = sorted(
            [(i, p) for i, p in enumerate(stability_k[k_value]) if p > 0],
            key=lambda x: -x[1],
        )[:20]
        top_stable_str = ", ".join(f"{i}:{p:.2f}" for i, p in top_stable)
        print(f"  K={k_value}: >=50% appearance: {stable_count}, >=80%: {high_stable}")
        print(f"       Top stability dims: {top_stable_str}")

    print("\nStep H4: Select top-K dim by train point estimate")
    selected_by_k = {}
    top_order = np.argsort(-d_corr_train)
    for k_value in K_VALUES:
        selected = top_order[:k_value]
        selected_by_k[k_value] = selected
        np.save(OUT_DIR / f"dim_select_K{k_value}.npy", selected)
        print(f"\nK={k_value} selected dim and their |rho| on train:")
        print(f"  dim indices: {selected}")
        print(f"  |rho| values: {d_corr_train[selected]}")

    print("\nStep H5: Train baseline diagnostic (5-fold CV)")
    print(f"{'K':>4} {'Ridge R^2':>15} {'CN-vs-AD AUC':>15}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cn_ad_mask = train_bl["dx"].isin(["NORMAL", "AD"]).values
    y_cnad = (train_bl.loc[cn_ad_mask, "dx"] == "AD").astype(int).values
    auc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k_value in K_EVAL:
        selected = top_order[:k_value]
        x_sel = Z_train_bl[:, selected]
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        r2 = cross_val_score(ridge, x_sel, y_train_d, cv=cv, scoring="r2")
        auc = cross_val_score(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            x_sel[cn_ad_mask],
            y_cnad,
            cv=auc_cv,
            scoring="roc_auc",
        )
        print(f"{k_value:>4} {r2.mean():>8.3f} +/- {r2.std():<5.3f} {auc.mean():>8.3f} +/- {auc.std():<5.3f}")

    print("\nReference:")
    print("  Full 768-dim Ridge R^2 after ComBat: 0.202")
    print("  Full 768-dim CN-vs-AD AUC after ComBat: 0.849")

    print("\nStep H6: Generalization to test set")
    print(f"{'K':>4} {'Train avg |rho|':>20} {'Test avg |rho|':>20} {'Test Ridge R^2':>17}")
    for k_value in K_VALUES:
        selected = selected_by_k[k_value]
        train_rho_mean = float(d_corr_train[selected].mean())
        test_rho = np.array([safe_abs_spearman(Z_test_bl[:, dim], y_test_d) for dim in selected])
        test_rho_mean = float(test_rho.mean())
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(Z_train_bl[:, selected], y_train_d)
        test_r2 = model.score(Z_test_bl[:, selected], y_test_d)
        print(f"{k_value:>4} {train_rho_mean:>20.3f} {test_rho_mean:>20.3f} {test_r2:>17.3f}")

    print("\nSaved train-selected dim index arrays:")
    for k_value in K_VALUES:
        print(f"  {OUT_DIR / f'dim_select_K{k_value}.npy'}")
    print("\nSTOP. Did not build visit-level feature CSV.")
    print("Wait for user to decide K based on stability, train/test diagnostics, and Wang model parameter budget.")


if __name__ == "__main__":
    main()
