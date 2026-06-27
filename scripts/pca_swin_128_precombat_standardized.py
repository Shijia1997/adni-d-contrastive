#!/usr/bin/env python
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("data/embeddings_128")


def image_col(df):
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    raise KeyError("No image id column found")


def safe_spearman(x, y):
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


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
    baseline_img = work.merge(
        baseline_visit[["RID", "EXAMDATE.x", "d_mod3"]],
        on=["RID", "EXAMDATE.x"],
        suffixes=("", "_baseline_median"),
    )
    keep = ["RID", "EXAMDATE.x", col, "d_mod3_baseline_median", "dx"]
    baseline_img = baseline_img.drop_duplicates("RID")[keep].rename(
        columns={col: "Image Data ID", "d_mod3_baseline_median": "d_mod3"}
    )
    return baseline_img.dropna(subset=["Image Data ID", "d_mod3"]).copy()


def main():
    print("Step T: 128^3 pre-ComBat standardized PCA sensitivity")

    swin_128 = np.load(OUT_DIR / "swin_latent.npy")
    image_ids_128 = np.load(OUT_DIR / "image_id_order.npy", allow_pickle=True).astype(str)
    print(f"128^3 latent: {swin_128.shape}")
    if swin_128.shape[0] != len(image_ids_128):
        raise SystemExit("STOP: latent rows do not match image ids")

    matched_train = pd.read_csv("data/master_smri/matched_TRAIN_with_batch.csv")
    matched_test = pd.read_csv("data/master_smri/matched_TEST_with_batch.csv")
    train_col = image_col(matched_train)
    test_col = image_col(matched_test)

    iid_to_idx_128 = {iid: i for i, iid in enumerate(image_ids_128)}
    train_iids = matched_train[train_col].astype(str).drop_duplicates().tolist()
    test_iids = matched_test[test_col].astype(str).drop_duplicates().tolist()
    train_iids_in = [iid for iid in train_iids if iid in iid_to_idx_128]
    test_iids_in = [iid for iid in test_iids if iid in iid_to_idx_128]
    train_idx = [iid_to_idx_128[iid] for iid in train_iids_in]
    test_idx = [iid_to_idx_128[iid] for iid in test_iids_in]

    x_train_128 = swin_128[train_idx]
    x_test_128 = swin_128[test_idx]
    print(f"Train latent: {x_train_128.shape}, Test latent: {x_test_128.shape}")
    if x_train_128.shape[0] != 1709 or x_test_128.shape[0] != 390:
        print("WARNING: train/test unique image counts differ from expected 1709/390")

    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train_128)
    x_test_z = scaler.transform(x_test_128)

    pca_z = PCA(n_components=min(x_train_z.shape) - 1, random_state=42)
    z_train_z = pca_z.fit_transform(x_train_z).astype(np.float32)
    z_test_z = pca_z.transform(x_test_z).astype(np.float32)
    print(f"PCA output: Z_train {z_train_z.shape}, Z_test {z_test_z.shape}")

    cum_var = np.cumsum(pca_z.explained_variance_ratio_)
    print("\nCumulative variance 128^3 standardized PCA:")
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        if k_value <= len(cum_var):
            print(f"  K={k_value}: {cum_var[k_value - 1] * 100:.1f}%")

    bl_train = build_baseline(matched_train)
    bl_test = build_baseline(matched_test)
    train_iid_to_z_row = {iid: i for i, iid in enumerate(train_iids_in)}
    test_iid_to_z_row = {iid: i for i, iid in enumerate(test_iids_in)}
    bl_train["z_row"] = bl_train["Image Data ID"].astype(str).map(train_iid_to_z_row)
    bl_test["z_row"] = bl_test["Image Data ID"].astype(str).map(test_iid_to_z_row)
    bl_train = bl_train.dropna(subset=["z_row"]).copy()
    bl_test = bl_test.dropna(subset=["z_row"]).copy()
    bl_train["z_row"] = bl_train["z_row"].astype(int)
    bl_test["z_row"] = bl_test["z_row"].astype(int)

    z_train_bl = z_train_z[bl_train["z_row"].values]
    z_test_bl = z_test_z[bl_test["z_row"].values]
    y_train_d = bl_train["d_mod3"].astype(float).values
    y_test_d = bl_test["d_mod3"].astype(float).values
    print(f"\nBaseline: train {len(bl_train)} RIDs, test {len(bl_test)} RIDs")

    print("\n128^3 Standardized PCA: Per-PC Spearman with d_mod3 (top 20 PCs)")
    print(f"{'PC':>4} {'Train |rho|':>12} {'Test |rho|':>12} {'Train rho':>11} {'Test rho':>10} {'var%':>7}")
    for pc in range(20):
        train_rho = safe_spearman(z_train_bl[:, pc], y_train_d)
        test_rho = safe_spearman(z_test_bl[:, pc], y_test_d)
        var_pct = pca_z.explained_variance_ratio_[pc] * 100
        print(
            f"{pc + 1:>4} {abs(train_rho):>12.3f} {abs(test_rho):>12.3f} "
            f"{train_rho:>+10.3f} {test_rho:>+10.3f} {var_pct:>6.2f}%"
        )

    print("\nTop 15 PCs ranked by |Spearman with D| (train, top 50 PCs):")
    pc_rho_list = [
        (pc, abs(safe_spearman(z_train_bl[:, pc], y_train_d)))
        for pc in range(min(50, z_train_bl.shape[1]))
    ]
    pc_rho_list.sort(key=lambda x: -x[1])
    print(f"{'PC':>5} {'Train |rho|':>12} {'Test |rho|':>12} {'var%':>7}")
    for pc, train_rho_abs in pc_rho_list[:15]:
        test_rho = safe_spearman(z_test_bl[:, pc], y_test_d)
        var_pct = pca_z.explained_variance_ratio_[pc] * 100
        print(f"{pc + 1:>5} {train_rho_abs:>12.3f} {abs(test_rho):>12.3f} {var_pct:>6.2f}%")

    print("\n128^3 Standardized PCA: Ridge R^2")
    print(f"{'K':>4} {'Train CV R^2':>15} {'Test R^2 (fit-on-train)':>25}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_128 = {}
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        if k_value > z_train_bl.shape[1]:
            continue
        x_tr = z_train_bl[:, :k_value]
        x_te = z_test_bl[:, :k_value]
        ridge_cv = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        train_r2 = cross_val_score(ridge_cv, x_tr, y_train_d, cv=cv, scoring="r2").mean()
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_tr, y_train_d)
        test_r2 = model.score(x_te, y_test_d)
        r2_128[k_value] = (train_r2, test_r2)
        print(f"{k_value:>4} {train_r2:>15.3f} {test_r2:>25.3f}")

    print("\n" + "=" * 75)
    print("96^3 vs 128^3 Standardized PCA Side-by-Side")
    print("=" * 75)
    print(f"{'K':>4} | {'96^3 Train R^2':>14} | {'96^3 Test R^2':>13} | {'128^3 Train R^2':>15} | {'128^3 Test R^2':>14}")
    print("-" * 75)
    prev_96 = {
        5: (-0.010, -0.016),
        10: (-0.007, 0.021),
        15: (0.000, 0.037),
        20: (0.001, 0.022),
        30: (0.078, -0.006),
        50: (0.119, 0.053),
        100: (0.034, -0.212),
    }
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        if k_value not in r2_128:
            continue
        train_96, test_96 = prev_96[k_value]
        train_128, test_128 = r2_128[k_value]
        print(
            f"{k_value:>4} | {train_96:>14.3f} | {test_96:>13.3f} | "
            f"{train_128:>15.3f} | {test_128:>14.3f}"
        )

    print("\n128^3 Standardized PCA: CN-vs-AD AUC")
    print(f"{'K':>4} {'Train CV AUC':>15} {'Test AUC (fit-on-train)':>25}")
    cn_ad_train = bl_train[bl_train["dx"].isin(["NORMAL", "AD"])].copy()
    cn_ad_test = bl_test[bl_test["dx"].isin(["NORMAL", "AD"])].copy()
    x_tr_cnad = z_train_z[cn_ad_train["z_row"].astype(int).values]
    x_te_cnad = z_test_z[cn_ad_test["z_row"].astype(int).values]
    y_tr_cnad = (cn_ad_train["dx"] == "AD").astype(int).values
    y_te_cnad = (cn_ad_test["dx"] == "AD").astype(int).values
    auc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k_value in [5, 10, 15, 20, 30, 50]:
        if k_value > z_train_bl.shape[1]:
            continue
        x_tr = x_tr_cnad[:, :k_value]
        x_te = x_te_cnad[:, :k_value]
        clf_cv = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear"))
        train_auc = cross_val_score(clf_cv, x_tr, y_tr_cnad, cv=auc_cv, scoring="roc_auc").mean()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear"))
        model.fit(x_tr, y_tr_cnad)
        test_auc = roc_auc_score(y_te_cnad, model.predict_proba(x_te)[:, 1])
        print(f"{k_value:>4} {train_auc:>15.3f} {test_auc:>25.3f}")

    np.save(OUT_DIR / "swin_pca_z_train.npy", z_train_z)
    np.save(OUT_DIR / "swin_pca_z_test.npy", z_test_z)
    np.save(OUT_DIR / "swin_pca_z_var.npy", pca_z.explained_variance_ratio_)
    np.save(OUT_DIR / "swin_pca_z_train_ids.npy", np.array(train_iids_in))
    np.save(OUT_DIR / "swin_pca_z_test_ids.npy", np.array(test_iids_in))
    with open(OUT_DIR / "swin_pca_z_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca_z}, f)

    print("\nSaved PCA output to data/embeddings_128/")
    print("STOP: 128^3 pre-ComBat PCA sensitivity complete. Did not run ComBat or dim selection.")


if __name__ == "__main__":
    main()
