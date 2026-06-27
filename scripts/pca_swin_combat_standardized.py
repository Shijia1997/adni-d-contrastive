#!/usr/bin/env python
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


def main():
    print("Step G: Standardized PCA on harmonized Swin latent")
    X_train = np.load(OUT_DIR / "swin_combat_train.npy")
    X_test = np.load(OUT_DIR / "swin_combat_test.npy")
    train_ids = np.load(OUT_DIR / "swin_combat_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load(OUT_DIR / "swin_combat_test_ids.npy", allow_pickle=True).astype(str)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"train_ids: {train_ids.shape}, test_ids: {test_ids.shape}")
    if X_train.shape[0] != len(train_ids) or X_test.shape[0] != len(test_ids):
        raise SystemExit("STOP: latent rows do not match id order")

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    n_components = min(X_train_z.shape) - 1
    pca_z = PCA(n_components=n_components, random_state=42)
    Z_train_z = pca_z.fit_transform(X_train_z).astype(np.float32)
    Z_test_z = pca_z.transform(X_test_z).astype(np.float32)

    print(f"Standardized PCA shapes: Z_train {Z_train_z.shape}, Z_test {Z_test_z.shape}")

    cum_var = np.cumsum(pca_z.explained_variance_ratio_)
    print("\nCumulative variance (standardized PCA):")
    for k in [5, 10, 15, 20, 30, 50, 100]:
        if k <= len(cum_var):
            print(f"  K={k}: {cum_var[k - 1] * 100:.1f}%")

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

    Z_bl = Z_train_z[baseline["idx"].values]
    y_d = baseline["d_mod3"].astype(float).values

    print(f"\nBaseline aligned RIDs: {len(baseline)}")
    print("\nStandardized PCA: Per-PC |Spearman| with D")
    print(f"{'PC':>4} {'|rho|':>7} {'rho':>7} {'p':>10} {'var%':>7}")

    pc_corrs = []
    for k in range(min(50, Z_bl.shape[1])):
        rho, p = spearmanr(Z_bl[:, k], y_d)
        if not np.isfinite(rho):
            rho, p = 0.0, 1.0
        var_pct = pca_z.explained_variance_ratio_[k] * 100
        pc_corrs.append((k, abs(rho), rho, p, var_pct))
        if k < 20:
            print(f"{k + 1:>4} {abs(rho):>7.3f} {rho:>+7.3f} {p:>10.2e} {var_pct:>6.2f}%")

    sorted_by_corr = sorted(pc_corrs, key=lambda x: -x[1])
    print("\nTop 10 PCs ranked by |Spearman with D| among first 50 PCs (standardized):")
    print(f"{'PC':>4} {'|rho|':>7} {'rho':>7} {'p':>10} {'var%':>7}")
    for idx, abs_rho, rho, p, var_pct in sorted_by_corr[:10]:
        print(f"{idx + 1:>4} {abs_rho:>7.3f} {rho:>+7.3f} {p:>10.2e} {var_pct:>6.2f}%")

    print("\nStandardized PCA Ridge D R^2:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_summary = []
    for k_test in [5, 10, 15, 20, 30, 50]:
        if k_test > Z_bl.shape[1]:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        scores = cross_val_score(model, Z_bl[:, :k_test], y_d, cv=cv, scoring="r2")
        ridge_summary.append((k_test, scores.mean(), scores.std()))
        print(f"  K={k_test:>2}: R^2={scores.mean():.3f} +/- {scores.std():.3f}  folds: {', '.join(f'{s:.3f}' for s in scores)}")

    np.save(OUT_DIR / "swin_pca_z_train.npy", Z_train_z)
    np.save(OUT_DIR / "swin_pca_z_test.npy", Z_test_z)
    np.save(OUT_DIR / "swin_pca_z_var.npy", pca_z.explained_variance_ratio_)
    with open(OUT_DIR / "swin_pca_z_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca_z}, f)

    print("\nFinal summary")
    print(f"  Standardized PCA outputs: Z_train {Z_train_z.shape}, Z_test {Z_test_z.shape}")
    print("  Cumulative variance:")
    for k in [5, 10, 15, 20, 30, 50, 100]:
        if k <= len(cum_var):
            print(f"    K={k}: {cum_var[k - 1] * 100:.1f}%")
    print("  Top D-correlated PCs among first 50:")
    for idx, abs_rho, rho, p, var_pct in sorted_by_corr[:5]:
        print(f"    PC{idx + 1}: |rho|={abs_rho:.3f}, rho={rho:+.3f}, p={p:.2e}, var={var_pct:.2f}%")
    print("  Ridge R^2 summary:")
    for k_test, mean, std in ridge_summary:
        print(f"    K={k_test}: {mean:.3f} +/- {std:.3f}")
    print("  Saved standardized PCA arrays/model in data/embeddings/")
    print("STOP: standardized PCA sensitivity complete. Waiting for method decision.")


if __name__ == "__main__":
    main()
