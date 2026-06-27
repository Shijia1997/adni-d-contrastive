#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/master_smri/matched_TRAIN.csv")
    parser.add_argument("--latent", default="data/embeddings/swin_latent.npy")
    parser.add_argument("--image_ids", default="data/embeddings/image_id_order.npy")
    parser.add_argument("--out_txt", default="data/embeddings/swin_cn_ad_auc.txt")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    latent = np.load(args.latent)
    image_ids = np.load(args.image_ids, allow_pickle=True).astype(str)
    if latent.shape[0] != len(image_ids):
        raise SystemExit("STOP: latent rows do not match image_id_order")
    emb_index = pd.DataFrame({"Image_Data_ID": image_ids, "latent_row": np.arange(len(image_ids))})

    train = train.copy()
    train["_scan_date"] = pd.to_datetime(train["scan_date"], errors="coerce")
    baseline = (
        train.dropna(subset=["Image_Data_ID", "dx", "_scan_date"])
        .sort_values(["RID", "_scan_date", "EXAMDATE.x"])
        .drop_duplicates("RID", keep="first")
    )
    baseline = baseline[baseline["dx"].isin(["NORMAL", "AD"])].copy()
    baseline = baseline.merge(emb_index, on="Image_Data_ID", how="inner")
    if baseline["dx"].nunique() != 2:
        raise SystemExit("STOP: baseline train subset does not contain both NORMAL and AD")

    X = latent[baseline["latent_row"].to_numpy()]
    y = (baseline["dx"].to_numpy() == "AD").astype(int)
    counts = pd.Series(baseline["dx"]).value_counts()
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise SystemExit(f"STOP: too few CN/AD samples for CV: {counts.to_dict()}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    lines = [
        "STEP 5 CN vs AD baseline disease-signal sanity",
        f"baseline train RIDs/images: {len(baseline)}",
        f"label counts: {counts.to_dict()}",
        f"cv folds: {n_splits}",
        f"AUC mean: {aucs.mean():.6f}",
        f"AUC std: {aucs.std():.6f}",
        f"AUC folds: {', '.join(f'{x:.6f}' for x in aucs)}",
    ]
    text = "\n".join(lines)
    print(text)
    Path(args.out_txt).write_text(text + "\n")

    if aucs.mean() < 0.65:
        raise SystemExit("STOP: AUC < 0.65, encoder/preprocessing likely problematic")
    if aucs.mean() < 0.80:
        print("CAUTION: AUC is between 0.65 and 0.80; report before downstream use")
    else:
        print("PASS: AUC >= 0.80")


if __name__ == "__main__":
    main()
