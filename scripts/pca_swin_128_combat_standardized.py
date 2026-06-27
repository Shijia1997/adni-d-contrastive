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
    print("Step U2/U3: 128^3 post-ComBat standardized PCA sensitivity")
    x_train = np.load(OUT_DIR / "swin_combat_train.npy")
    x_test = np.load(OUT_DIR / "swin_combat_test.npy")
    train_ids = np.load(OUT_DIR / "swin_combat_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load(OUT_DIR / "swin_combat_test_ids.npy", allow_pickle=True).astype(str)
    print(f"Post-ComBat train latent: {x_train.shape}, test latent: {x_test.shape}")
    if x_train.shape[0] != len(train_ids) or x_test.shape[0] != len(test_ids):
        raise SystemExit("STOP: latent rows do not match ids")

    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)

    pca = PCA(n_components=min(x_train_z.shape) - 1, random_state=42)
    z_train = pca.fit_transform(x_train_z).astype(np.float32)
    z_test = pca.transform(x_test_z).astype(np.float32)
    print(f"PCA output: Z_train {z_train.shape}, Z_test {z_test.shape}")

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print("\nCumulative variance 128^3 post-ComBat standardized PCA:")
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        if k_value <= len(cum_var):
            print(f"  K={k_value}: {cum_var[k_value - 1] * 100:.1f}%")

    matched_train = pd.read_csv("data/master_smri/matched_TRAIN_with_batch.csv")
    matched_test = pd.read_csv("data/master_smri/matched_TEST_with_batch.csv")
    bl_train = build_baseline(matched_train)
    bl_test = build_baseline(matched_test)
    train_id_to_row = {iid: i for i, iid in enumerate(train_ids)}
    test_id_to_row = {iid: i for i, iid in enumerate(test_ids)}
    bl_train["z_row"] = bl_train["Image Data ID"].astype(str).map(train_id_to_row)
    bl_test["z_row"] = bl_test["Image Data ID"].astype(str).map(test_id_to_row)
    bl_train = bl_train.dropna(subset=["z_row"]).copy()
    bl_test = bl_test.dropna(subset=["z_row"]).copy()
    bl_train["z_row"] = bl_train["z_row"].astype(int)
    bl_test["z_row"] = bl_test["z_row"].astype(int)

    z_train_bl = z_train[bl_train["z_row"].values]
    z_test_bl = z_test[bl_test["z_row"].values]
    y_train_d = bl_train["d_mod3"].astype(float).values
    y_test_d = bl_test["d_mod3"].astype(float).values
    print(f"\nBaseline: train {len(bl_train)} RIDs, test {len(bl_test)} RIDs")

    print("\n128^3 Post-ComBat standardized PCA: Per-PC Spearman with d_mod3 (top 20 PCs)")
    print(f"{'PC':>4} {'Train |rho|':>12} {'Test |rho|':>12} {'Train rho':>11} {'Test rho':>10} {'var%':>7}")
    for pc in range(20):
        train_rho = safe_spearman(z_train_bl[:, pc], y_train_d)
        test_rho = safe_spearman(z_test_bl[:, pc], y_test_d)
        var_pct = pca.explained_variance_ratio_[pc] * 100
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
        var_pct = pca.explained_variance_ratio_[pc] * 100
        print(f"{pc + 1:>5} {train_rho_abs:>12.3f} {abs(test_rho):>12.3f} {var_pct:>6.2f}%")

    print("\n128^3 Post-ComBat standardized PCA: Ridge R^2")
    print(f"{'K':>4} {'Train CV R^2':>15} {'Test R^2 (fit-on-train)':>25}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_post = {}
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
        r2_post[k_value] = (train_r2, test_r2)
        print(f"{k_value:>4} {train_r2:>15.3f} {test_r2:>25.3f}")

    print("\n" + "=" * 79)
    print("128^3 standardized PCA: Pre-ComBat vs Post-ComBat Ridge R^2")
    print("=" * 79)
    print(f"{'K':>4} | {'Pre Train R^2':>13} | {'Pre Test R^2':>12} | {'Post Train R^2':>14} | {'Post Test R^2':>13}")
    print("-" * 79)
    pre_128 = {
        5: (0.029, -0.009),
        10: (0.028, 0.035),
        15: (0.088, 0.053),
        20: (0.129, 0.131),
        30: (0.191, 0.164),
        50: (0.190, 0.171),
        100: (0.142, 0.182),
    }
    for k_value in [5, 10, 15, 20, 30, 50, 100]:
        if k_value not in r2_post:
            continue
        pre_train, pre_test = pre_128[k_value]
        post_train, post_test = r2_post[k_value]
        print(
            f"{k_value:>4} | {pre_train:>13.3f} | {pre_test:>12.3f} | "
            f"{post_train:>14.3f} | {post_test:>13.3f}"
        )

    print("\n128^3 Post-ComBat standardized PCA: CN-vs-AD AUC")
    print(f"{'K':>4} {'Train CV AUC':>15} {'Test AUC (fit-on-train)':>25}")
    cn_ad_train = bl_train[bl_train["dx"].isin(["NORMAL", "AD"])].copy()
    cn_ad_test = bl_test[bl_test["dx"].isin(["NORMAL", "AD"])].copy()
    x_tr_cnad = z_train[cn_ad_train["z_row"].astype(int).values]
    x_te_cnad = z_test[cn_ad_test["z_row"].astype(int).values]
    y_tr_cnad = (cn_ad_train["dx"] == "AD").astype(int).values
    y_te_cnad = (cn_ad_test["dx"] == "AD").astype(int).values
    auc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_post = {}
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
        auc_post[k_value] = (train_auc, test_auc)
        print(f"{k_value:>4} {train_auc:>15.3f} {test_auc:>25.3f}")

    print("\n" + "=" * 77)
    print("128^3 standardized PCA: Pre-ComBat vs Post-ComBat CN-vs-AD AUC")
    print("=" * 77)
    print(f"{'K':>4} | {'Pre Train AUC':>13} | {'Pre Test AUC':>12} | {'Post Train AUC':>14} | {'Post Test AUC':>13}")
    print("-" * 77)
    pre_auc = {
        5: (0.659, 0.581),
        10: (0.722, 0.650),
        15: (0.776, 0.737),
        20: (0.839, 0.813),
        30: (0.858, 0.855),
        50: (0.851, 0.842),
    }
    for k_value in [5, 10, 15, 20, 30, 50]:
        if k_value not in auc_post:
            continue
        pre_train, pre_test = pre_auc[k_value]
        post_train, post_test = auc_post[k_value]
        print(
            f"{k_value:>4} | {pre_train:>13.3f} | {pre_test:>12.3f} | "
            f"{post_train:>14.3f} | {post_test:>13.3f}"
        )

    np.save(OUT_DIR / "swin_combat_pca_z_train.npy", z_train)
    np.save(OUT_DIR / "swin_combat_pca_z_test.npy", z_test)
    np.save(OUT_DIR / "swin_combat_pca_z_var.npy", pca.explained_variance_ratio_)
    np.save(OUT_DIR / "swin_combat_pca_z_train_ids.npy", train_ids)
    np.save(OUT_DIR / "swin_combat_pca_z_test_ids.npy", test_ids)
    with open(OUT_DIR / "swin_combat_pca_z_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)

    print("\nSaved post-ComBat PCA output to data/embeddings_128/")
    print("STOP: 128^3 post-ComBat PCA diagnostic complete. Waiting for K/path decision.")


if __name__ == "__main__":
    main()
