#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def image_col(df):
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    raise KeyError("No image id column found")


def safe_spearman(x, y):
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


def build_baseline(df, iid_to_idx):
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

    baseline_img = work.merge(
        baseline_visit[["RID", "EXAMDATE.x", "d_mod3"]],
        on=["RID", "EXAMDATE.x"],
        suffixes=("", "_baseline_median"),
    )
    keep_cols = ["RID", "EXAMDATE.x", col, "d_mod3_baseline_median", "dx"]
    baseline_img = baseline_img.drop_duplicates("RID")[keep_cols].rename(
        columns={col: "Image Data ID", "d_mod3_baseline_median": "d_mod3"}
    )
    baseline_img = baseline_img.dropna(subset=["Image Data ID", "d_mod3"]).copy()
    baseline_img["idx"] = baseline_img["Image Data ID"].astype(str).map(iid_to_idx)
    baseline_img = baseline_img.dropna(subset=["idx"]).copy()
    baseline_img["idx"] = baseline_img["idx"].astype(int)
    return baseline_img


def main():
    print("Step K: Detailed correlation diagnostic")

    z_pca_train = np.load("data/embeddings/swin_pca_z_train.npy")
    z_pca_test = np.load("data/embeddings/swin_pca_z_test.npy")
    x_combat_train = np.load("data/embeddings/swin_combat_train.npy")
    x_combat_test = np.load("data/embeddings/swin_combat_test.npy")
    train_ids = np.load("data/embeddings/swin_combat_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load("data/embeddings/swin_combat_test_ids.npy", allow_pickle=True).astype(str)
    matched_train = pd.read_csv("data/master_smri/matched_TRAIN_with_batch.csv")
    matched_test = pd.read_csv("data/master_smri/matched_TEST_with_batch.csv")

    iid_to_train = {iid: i for i, iid in enumerate(train_ids)}
    iid_to_test = {iid: i for i, iid in enumerate(test_ids)}
    bl_train = build_baseline(matched_train, iid_to_train)
    bl_test = build_baseline(matched_test, iid_to_test)

    print(f"\nTrain baseline RIDs: {len(bl_train)}")
    print(f"Test baseline RIDs:  {len(bl_test)}")

    y_train_d = bl_train["d_mod3"].astype(float).values
    y_test_d = bl_test["d_mod3"].astype(float).values
    train_idx = bl_train["idx"].astype(int).values
    test_idx = bl_test["idx"].astype(int).values

    z_pca_train_bl = z_pca_train[train_idx]
    z_pca_test_bl = z_pca_test[test_idx]

    print("\n" + "=" * 70)
    print("STANDARDIZED PCA: Per-PC Spearman with d_mod3 (top 20 PCs)")
    print("=" * 70)
    print(f"{'PC':>4} {'Train |rho|':>12} {'Test |rho|':>12} {'Train rho':>11} {'Test rho':>10}")
    for pc in range(20):
        train_rho = safe_spearman(z_pca_train_bl[:, pc], y_train_d)
        test_rho = safe_spearman(z_pca_test_bl[:, pc], y_test_d)
        print(
            f"{pc + 1:>4} {abs(train_rho):>12.3f} {abs(test_rho):>12.3f} "
            f"{train_rho:>+10.3f} {test_rho:>+10.3f}"
        )

    print("\nTop 15 PCs ranked by |Spearman with D| (train, first 50 PCs):")
    train_pc_rho = [
        (pc, abs(safe_spearman(z_pca_train_bl[:, pc], y_train_d)))
        for pc in range(min(50, z_pca_train_bl.shape[1]))
    ]
    train_pc_rho.sort(key=lambda x: -x[1])
    print(f"{'PC':>5} {'Train |rho|':>12} {'Test |rho|':>12}")
    for pc, train_rho_abs in train_pc_rho[:15]:
        test_rho = safe_spearman(z_pca_test_bl[:, pc], y_test_d)
        print(f"{pc + 1:>5} {train_rho_abs:>12.3f} {abs(test_rho):>12.3f}")

    print("\n" + "=" * 70)
    print("STANDARDIZED PCA: Ridge R^2 train-fit on top-K PCs, eval train/test")
    print("=" * 70)
    print(f"{'K':>4} {'Train CV R^2':>15} {'Test R^2 (fit-on-train)':>25}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        x_tr = z_pca_train_bl[:, :k_value]
        x_te = z_pca_test_bl[:, :k_value]
        model_cv = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        train_r2 = cross_val_score(model_cv, x_tr, y_train_d, cv=cv, scoring="r2").mean()
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_tr, y_train_d)
        test_r2 = model.score(x_te, y_test_d)
        print(f"{k_value:>4} {train_r2:>15.3f} {test_r2:>25.3f}")

    print("\n" + "=" * 70)
    print("D-SELECTED DIM: Same diagnostic for comparison")
    print("=" * 70)
    x_combat_train_bl = x_combat_train[train_idx]
    x_combat_test_bl = x_combat_test[test_idx]
    for k_value in [5, 10, 15, 20, 30]:
        selected_dim = np.load(f"data/embeddings/dim_select_K{k_value}.npy")
        x_tr = x_combat_train_bl[:, selected_dim]
        x_te = x_combat_test_bl[:, selected_dim]
        model_cv = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        train_r2 = cross_val_score(model_cv, x_tr, y_train_d, cv=cv, scoring="r2").mean()
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_tr, y_train_d)
        test_r2 = model.score(x_te, y_test_d)
        train_rhos = [abs(safe_spearman(x_tr[:, k], y_train_d)) for k in range(k_value)]
        test_rhos = [abs(safe_spearman(x_te[:, k], y_test_d)) for k in range(k_value)]
        print(
            f"K={k_value:>3}: "
            f"Train CV R^2 = {train_r2:>+.3f}, "
            f"Test R^2 = {test_r2:>+.3f}, "
            f"avg train |rho| = {np.mean(train_rhos):.3f}, "
            f"avg test |rho| = {np.mean(test_rhos):.3f}"
        )

    print("\n" + "=" * 70)
    print("Important note about d_mod3:")
    print("d_mod3 is Wang's scalar-only D (no image input).")
    print("R^2 above is image features vs scalar-derived D.")
    print("If joint Wang fit includes image, D will re-estimate;")
    print("PC correlation with re-estimated D may differ.")
    print("=" * 70)

    dx_train = bl_train["dx"].map({"NORMAL": 0, "MCI": 1, "AD": 2}).astype(float).values
    dx_test = bl_test["dx"].map({"NORMAL": 0, "MCI": 1, "AD": 2}).astype(float).values
    train_mask = np.isfinite(dx_train)
    test_mask = np.isfinite(dx_test)
    dx_d_train_rho = safe_spearman(dx_train[train_mask], y_train_d[train_mask])
    dx_d_test_rho = safe_spearman(dx_test[test_mask], y_test_d[test_mask])
    print("\nReference: dx (CN/MCI/AD ordinal) vs d_mod3:")
    print(f"  Train Spearman: {dx_d_train_rho:.3f}")
    print(f"  Test Spearman:  {dx_d_test_rho:.3f}")

    print("\nBest single PC discriminating CN vs AD among top 20 PCs:")
    for pc in range(20):
        cn_pc = z_pca_train_bl[bl_train["dx"].values == "NORMAL", pc]
        ad_pc = z_pca_train_bl[bl_train["dx"].values == "AD", pc]
        if len(cn_pc) > 5 and len(ad_pc) > 5:
            t_val, p_val = ttest_ind(cn_pc, ad_pc, equal_var=False)
            if abs(t_val) > 2.5:
                print(f"  PC{pc + 1}: t={t_val:+.2f}, p={p_val:.1e}")

    print("\nVariance explained by each of top 15 standardized PCs:")
    pca_var = np.load("data/embeddings/swin_pca_z_var.npy")
    for k in range(15):
        print(f"  PC{k + 1}: {pca_var[k] * 100:.2f}%")

    print("\nSTOP: detailed correlation diagnostic complete. No files written.")


if __name__ == "__main__":
    main()
