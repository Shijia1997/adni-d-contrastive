#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


K = 30
PC_COLS = [f"pc_{i + 1}" for i in range(K)]


def image_col(df):
    if "Image Data ID" in df.columns:
        return "Image Data ID"
    if "Image_Data_ID" in df.columns:
        return "Image_Data_ID"
    raise KeyError("No image id column found")


def add_pc_features(df, lookup):
    col = image_col(df)
    out = df.copy()
    pc_array = np.full((len(out), K), np.nan, dtype=np.float32)
    for i, iid in enumerate(out[col].astype(str)):
        if iid in lookup:
            pc_array[i] = lookup[iid]
    for k, pc_col in enumerate(PC_COLS):
        out[pc_col] = pc_array[:, k]
    return out.dropna(subset=PC_COLS).copy()


def baseline_per_rid(df):
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
    baseline = work.merge(
        baseline_visit[["RID", "EXAMDATE.x", "d_mod3"]],
        on=["RID", "EXAMDATE.x"],
        suffixes=("", "_baseline_median"),
    )
    baseline = baseline.sort_values(["RID", "EXAMDATE.x"]).drop_duplicates("RID").copy()
    baseline["d_mod3"] = baseline["d_mod3_baseline_median"]
    return baseline.drop(columns=["d_mod3_baseline_median"])


def regression_metrics(y_true, pred):
    pearson = pearsonr(y_true, pred)[0] if len(np.unique(pred)) > 1 else np.nan
    spearman = spearmanr(y_true, pred)[0] if len(np.unique(pred)) > 1 else np.nan
    return {
        "r2": r2_score(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": mean_absolute_error(y_true, pred),
        "pearson": pearson,
        "spearman": spearman,
    }


def task1_regression(train_bl, test_bl):
    print("\n=== Task 1: Image-only -> d_mod3 regression ===")
    train_bl_d = train_bl.dropna(subset=["d_mod3"]).copy()
    test_bl_d = test_bl.dropna(subset=["d_mod3"]).copy()
    x_tr = train_bl_d[PC_COLS].values
    y_tr = train_bl_d["d_mod3"].astype(float).values
    x_te = test_bl_d[PC_COLS].values
    y_te = test_bl_d["d_mod3"].astype(float).values

    print(f"Train subjects: {len(y_tr)}, Test subjects: {len(y_te)}")
    print(f"y_train range: [{y_tr.min():.3f}, {y_tr.max():.3f}], std {y_tr.std():.3f}")
    print(f"y_test range:  [{y_te.min():.3f}, {y_te.max():.3f}], std {y_te.std():.3f}")

    models = [
        ("Ridge alpha=1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge alpha=10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("RF n=200", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ("GBM n=300", GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42)),
        (
            "MLP 128,64",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=3000,
                    alpha=0.01,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ),
    ]
    for name, model in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        m = regression_metrics(y_te, pred)
        print(
            f"  {name:16s} R2={m['r2']:+.3f} RMSE={m['rmse']:.3f} "
            f"MAE={m['mae']:.3f} Pearson={m['pearson']:.3f} Spearman={m['spearman']:.3f}"
        )


def task2_classification(train_bl, test_bl):
    print("\n=== Task 2: Image-only classification ===")
    mask_tr = train_bl["dx"].isin(["NORMAL", "AD"])
    mask_te = test_bl["dx"].isin(["NORMAL", "AD"])
    x_tr = train_bl.loc[mask_tr, PC_COLS].values
    y_tr = (train_bl.loc[mask_tr, "dx"] == "AD").astype(int).values
    x_te = test_bl.loc[mask_te, PC_COLS].values
    y_te = (test_bl.loc[mask_te, "dx"] == "AD").astype(int).values
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, solver="liblinear"))
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    print("\nCN vs AD binary:")
    print(f"  Train: {len(y_tr)} subjects ({y_tr.sum()} AD, {(y_tr == 0).sum()} NORMAL)")
    print(f"  Test:  {len(y_te)} subjects ({y_te.sum()} AD, {(y_te == 0).sum()} NORMAL)")
    print(f"  Test AUC: {roc_auc_score(y_te, prob):.3f}")
    print(f"  Test accuracy @0.5: {accuracy_score(y_te, pred):.3f}")

    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    mask_tr_all = train_bl["dx"].isin(dx_map)
    mask_te_all = test_bl["dx"].isin(dx_map)
    x_tr3 = train_bl.loc[mask_tr_all, PC_COLS].values
    y_tr3 = train_bl.loc[mask_tr_all, "dx"].map(dx_map).astype(int).values
    x_te3 = test_bl.loc[mask_te_all, PC_COLS].values
    y_te3 = test_bl.loc[mask_te_all, "dx"].map(dx_map).astype(int).values
    model3 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, multi_class="ovr", solver="liblinear"),
    )
    model3.fit(x_tr3, y_tr3)
    prob3 = model3.predict_proba(x_te3)
    pred3 = model3.predict(x_te3)
    print("\nCN/MCI/AD 3-class:")
    print(f"  Test counts: {pd.Series(y_te3).map({0:'NORMAL',1:'MCI',2:'AD'}).value_counts().to_dict()}")
    print(f"  Accuracy: {accuracy_score(y_te3, pred3):.3f}")
    print(f"  NORMAL vs rest AUC: {roc_auc_score((y_te3 == 0).astype(int), prob3[:, 0]):.3f}")
    print(f"  MCI vs rest AUC:    {roc_auc_score((y_te3 == 1).astype(int), prob3[:, 1]):.3f}")
    print(f"  AD vs rest AUC:     {roc_auc_score((y_te3 == 2).astype(int), prob3[:, 2]):.3f}")
    print("  Classification report:")
    print(classification_report(y_te3, pred3, target_names=["NORMAL", "MCI", "AD"], digits=3))


def label_conversion(df, baseline_dx="MCI", target_dx="AD", horizon_years=2):
    work = df.copy()
    work["EXAMDATE.x"] = pd.to_datetime(work["EXAMDATE.x"], errors="coerce")
    work = work.sort_values(["RID", "EXAMDATE.x"])
    rows = []
    for rid, group in work.groupby("RID"):
        group = group.dropna(subset=["EXAMDATE.x", "dx"])
        if group.empty:
            continue
        first_date = group["EXAMDATE.x"].iloc[0]
        baseline_rows = group[group["EXAMDATE.x"] == first_date]
        baseline_dx_values = baseline_rows["dx"].dropna().unique()
        if baseline_dx not in baseline_dx_values:
            continue
        target_date = first_date + pd.DateOffset(years=horizon_years)
        followup = group[(group["EXAMDATE.x"] > first_date) & (group["EXAMDATE.x"] <= target_date)]
        converted = int((followup["dx"] == target_dx).any())
        rows.append({"RID": rid, "converted_2y": converted, "n_followup_2y": followup["EXAMDATE.x"].nunique()})
    return pd.DataFrame(rows)


def task3_conversion(matched_train, matched_test, train_bl, test_bl):
    print("\n=== Task 3: Image-only MCI->AD 2-year conversion ===")
    train_conv = label_conversion(matched_train)
    test_conv = label_conversion(matched_test)
    print(f"Train MCI baseline subjects: {len(train_conv)}; converters: {int(train_conv['converted_2y'].sum()) if len(train_conv) else 0}")
    print(f"Test MCI baseline subjects:  {len(test_conv)}; converters: {int(test_conv['converted_2y'].sum()) if len(test_conv) else 0}")
    if len(train_conv) == 0 or len(test_conv) == 0:
        print("  skipped: no MCI baseline conversion labels")
        return
    train_mci = train_bl[train_bl["dx"] == "MCI"].merge(train_conv, on="RID", how="inner")
    test_mci = test_bl[test_bl["dx"] == "MCI"].merge(test_conv, on="RID", how="inner")
    y_tr = train_mci["converted_2y"].astype(int).values
    y_te = test_mci["converted_2y"].astype(int).values
    print(f"Feature-aligned train/test: {len(y_tr)} / {len(y_te)}")
    print(f"Aligned converters train/test: {int(y_tr.sum())}/{int(y_te.sum())}")
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or y_te.sum() < 3:
        print("  skipped AUC: not enough positive/negative conversion examples in train or test")
        return
    x_tr = train_mci[PC_COLS].values
    x_te = test_mci[PC_COLS].values
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, solver="liblinear"))
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]
    print(f"  Image-only conversion AUC: {roc_auc_score(y_te, prob):.3f}")


def task4_incremental(train_bl, test_bl):
    print("\n=== Task 4: Image vs scalar incremental value for d_mod3 ===")
    scalar_candidates = [
        "MMSCORE",
        "logmem",
        "DSST",
        "biec.thik",
        "bihippo",
        "biec.vol",
        "MTL1",
        "TAU",
        "PTAU",
        "ABETA",
    ]
    scalar_cols = [c for c in scalar_candidates if c in train_bl.columns]
    print(f"Available scalar columns: {scalar_cols}")
    if not scalar_cols:
        print("  skipped: no scalar columns")
        return
    print("Scalar non-missing counts in train baseline:")
    print(train_bl[scalar_cols].notna().sum().to_string())
    print("Scalar non-missing counts in test baseline:")
    print(test_bl[scalar_cols].notna().sum().to_string())

    needed = scalar_cols + PC_COLS + ["d_mod3"]
    tr = train_bl.dropna(subset=needed).copy()
    te = test_bl.dropna(subset=needed).copy()
    print(f"Complete-case subset with all scalar + PC + d: train {len(tr)}, test {len(te)}")
    if len(tr) <= 50 or len(te) <= 20:
        print("  skipped full scalar comparison: complete-case subset too small")
        return

    y_tr = tr["d_mod3"].astype(float).values
    y_te = te["d_mod3"].astype(float).values
    feature_sets = [
        ("scalar only", scalar_cols),
        ("image only", PC_COLS),
        ("scalar + image", scalar_cols + PC_COLS),
    ]
    scores = {}
    for label, cols in feature_sets:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(tr[cols].values, y_tr)
        pred = model.predict(te[cols].values)
        m = regression_metrics(y_te, pred)
        scores[label] = m
        print(
            f"  {label:14s} R2={m['r2']:+.3f} RMSE={m['rmse']:.3f} "
            f"MAE={m['mae']:.3f} Pearson={m['pearson']:.3f} Spearman={m['spearman']:.3f}"
        )
    print(f"  Incremental R2 image over scalar: {scores['scalar + image']['r2'] - scores['scalar only']['r2']:+.3f}")


def main():
    print("Path A: Evaluate 128^3 post-ComBat PCA K=30 image embedding tasks")
    matched_train = pd.read_csv("data/master_smri/matched_TRAIN_with_batch.csv")
    matched_test = pd.read_csv("data/master_smri/matched_TEST_with_batch.csv")
    z_train = np.load("data/embeddings_128/swin_combat_pca_z_train.npy")
    z_test = np.load("data/embeddings_128/swin_combat_pca_z_test.npy")
    train_ids = np.load("data/embeddings_128/swin_combat_pca_z_train_ids.npy", allow_pickle=True).astype(str)
    test_ids = np.load("data/embeddings_128/swin_combat_pca_z_test_ids.npy", allow_pickle=True).astype(str)
    print(f"Z_train: {z_train.shape}, Z_test: {z_test.shape}, K={K}")

    train_lookup = {iid: z_train[i, :K] for i, iid in enumerate(train_ids)}
    test_lookup = {iid: z_test[i, :K] for i, iid in enumerate(test_ids)}
    train_full = add_pc_features(matched_train, train_lookup)
    test_full = add_pc_features(matched_test, test_lookup)
    print(f"Visit-level rows with PC features: train {len(train_full)} / {len(matched_train)}, test {len(test_full)} / {len(matched_test)}")

    train_bl = baseline_per_rid(train_full)
    test_bl = baseline_per_rid(test_full)
    print(f"Baseline-per-RID: train {len(train_bl)}, test {len(test_bl)}")
    print("Baseline dx counts train:")
    print(train_bl["dx"].value_counts(dropna=False).to_string())
    print("Baseline dx counts test:")
    print(test_bl["dx"].value_counts(dropna=False).to_string())

    task1_regression(train_bl, test_bl)
    task2_classification(train_bl, test_bl)
    task3_conversion(matched_train, matched_test, train_bl, test_bl)
    task4_incremental(train_bl, test_bl)
    print("\nSTOP: Path A K=30 task evaluation complete. No Wang fit run.")


if __name__ == "__main__":
    main()
