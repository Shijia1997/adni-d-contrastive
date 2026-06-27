#!/usr/bin/env python
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("data/embeddings")


def image_col(df):
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    raise KeyError("No image id column found")


def cohen_d(ad, cn):
    pooled_std = np.sqrt((cn.var(ddof=1) + ad.var(ddof=1)) / 2.0)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return np.nan
    return (ad.mean() - cn.mean()) / pooled_std


def main():
    X_train = np.load(OUT_DIR / "swin_combat_train.npy")
    X_test = np.load(OUT_DIR / "swin_combat_test.npy")
    train_ids = np.load(OUT_DIR / "swin_combat_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load(OUT_DIR / "swin_combat_test_ids.npy", allow_pickle=True).astype(str)

    print("Step E: PCA on harmonized Swin latent")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"train_ids: {train_ids.shape}, test_ids: {test_ids.shape}")
    if X_train.shape[0] != len(train_ids) or X_test.shape[0] != len(test_ids):
        raise SystemExit("STOP: latent rows do not match id order")

    n_components = min(X_train.shape) - 1
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    print("\nCumulative variance explained:")
    for k in [5, 10, 15, 20, 30, 50, 100, 200]:
        if k <= len(cum_var):
            print(f"  K={k}: {cum_var[k - 1] * 100:.1f}%")
    k_50 = int((cum_var >= 0.50).argmax() + 1)
    k_80 = int((cum_var >= 0.80).argmax() + 1)
    k_90 = int((cum_var >= 0.90).argmax() + 1)
    print(f"\nK for 50% var: {k_50}")
    print(f"K for 80% var: {k_80}")
    print(f"K for 90% var: {k_90}")

    Z_train = pca.transform(X_train).astype(np.float32)
    Z_test = pca.transform(X_test).astype(np.float32)
    print(f"\nZ_train: {Z_train.shape}, Z_test: {Z_test.shape}")

    np.save(OUT_DIR / "swin_pca_train.npy", Z_train)
    np.save(OUT_DIR / "swin_pca_test.npy", Z_test)
    np.save(OUT_DIR / "swin_pca_explained_variance.npy", pca.explained_variance_ratio_)
    with open(OUT_DIR / "swin_pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    matched = pd.read_csv("data/master_smri/matched_TRAIN_with_batch.csv")
    col = image_col(matched)
    matched["_date"] = pd.to_datetime(matched["EXAMDATE.x"], errors="coerce")
    baseline = (
        matched.dropna(subset=["RID", "_date", col, "d_mod3"])
        .sort_values(["RID", "_date"])
        .groupby("RID", as_index=False)
        .first()
    )
    iid_to_idx = {iid: i for i, iid in enumerate(train_ids)}
    baseline["idx"] = baseline[col].astype(str).map(iid_to_idx)
    baseline = baseline.dropna(subset=["idx", "d_mod3"]).copy()
    baseline["idx"] = baseline["idx"].astype(int)
    Z_bl = Z_train[baseline["idx"].values]
    y_d = baseline["d_mod3"].astype(float).values

    print(f"\nStep F: PCA sanity checks")
    print(f"Baseline aligned RIDs: {len(baseline)}")
    print("\nPer-PC |Spearman| with D (top 20 PCs by variance):")
    pc_corrs = []
    for k in range(min(50, Z_bl.shape[1])):
        rho, p = spearmanr(Z_bl[:, k], y_d)
        if not np.isfinite(rho):
            rho, p = 0.0, 1.0
        pc_corrs.append((k, abs(rho), rho, p))

    print(f"{'PC':>4} {'|rho|':>7} {'rho':>7} {'p':>10} {'var%':>7}")
    for idx, abs_rho, rho, p in pc_corrs[:20]:
        var_pct = pca.explained_variance_ratio_[idx] * 100
        print(f"{idx + 1:>4} {abs_rho:>7.3f} {rho:>+7.3f} {p:>10.2e} {var_pct:>6.2f}%")

    sorted_by_corr = sorted(pc_corrs, key=lambda x: -x[1])
    print("\nTop 10 PCs ranked by |Spearman with D| among first 50 PCs:")
    print(f"{'PC':>4} {'|rho|':>7} {'rho':>7} {'p':>10} {'var%':>7}")
    for idx, abs_rho, rho, p in sorted_by_corr[:10]:
        var_pct = pca.explained_variance_ratio_[idx] * 100
        print(f"{idx + 1:>4} {abs_rho:>7.3f} {rho:>+7.3f} {p:>10.2e} {var_pct:>6.2f}%")

    baseline_dx = baseline[baseline["dx"].isin(["NORMAL", "AD"])].copy()
    Z_dx = Z_train[baseline_dx["idx"].astype(int).values]
    y_dx = (baseline_dx["dx"].values == "AD").astype(int)
    print("\nTop 5 PC: CN vs AD t-test (Cohen's d):")
    for k in range(5):
        cn = Z_dx[y_dx == 0, k]
        ad = Z_dx[y_dx == 1, k]
        t, p = ttest_ind(cn, ad, equal_var=False)
        d = cohen_d(ad, cn)
        print(f"  PC{k + 1}: t={t:>+6.2f}, p={p:.2e}, Cohen's d={d:>+5.2f}")

    print("\nRidge D R^2 with PCA features:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for k_test in [5, 10, 15, 20, 30, 50]:
        if k_test > Z_bl.shape[1]:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        scores = cross_val_score(model, Z_bl[:, :k_test], y_d, cv=cv, scoring="r2")
        print(f"  top {k_test:>2} PCs: {scores.mean():.3f} +/- {scores.std():.3f}  folds: {', '.join(f'{s:.3f}' for s in scores)}")

    print("\nFinal summary")
    print(f"  PCA fit on train only: {X_train.shape[0]} images, {X_train.shape[1]} features")
    print(f"  PCA outputs: Z_train {Z_train.shape}, Z_test {Z_test.shape}")
    print("  Cumulative variance:")
    for k in [5, 10, 15, 20, 30, 50]:
        if k <= len(cum_var):
            print(f"    K={k}: {cum_var[k - 1] * 100:.1f}%")
    print(f"  K for 50/80/90% variance: {k_50}/{k_80}/{k_90}")
    print("  Saved PCA model and transformed arrays in data/embeddings/")
    print("STOP: PCA diagnostics complete. Waiting for final K decision.")


if __name__ == "__main__":
    main()
