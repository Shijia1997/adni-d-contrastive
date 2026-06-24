import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

from experiment_utils import (
    DCON,
    build_all_censored_cohorts,
    evaluate_current_classification,
    get_valid_cohort,
    load_raw_combat_data,
    regression_metrics,
    scaled_features,
)


class DxPretrainNet(nn.Module):
    def __init__(self, in_dim=768, hidden=384, latent=256, n_out=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.BatchNorm1d(latent),
        )
        self.head = nn.Linear(latent, n_out)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


class FinetuneNet(nn.Module):
    def __init__(self, encoder, latent=256, n_out=1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, n_out)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dx_pretrain(x_train, train_meta, seed, device, epochs=150, batch_size=128):
    set_seed(seed)
    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    y = train_meta["dx"].map(dx_map).fillna(-1).astype(int).values
    valid = y >= 0
    x_t = torch.FloatTensor(x_train[valid]).to(device)
    y_t = torch.LongTensor(y[valid]).to(device)
    model = DxPretrainNet(in_dim=x_train.shape[1], hidden=384, latent=256, n_out=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(x_t)
    losses = []
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            out, _ = model(x_t[b])
            loss = F.cross_entropy(out, y_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        losses.append(total / nb)
    return model, losses


def eval_setup3(z_train, z_test, train_meta, test_meta, cohorts):
    metrics = {}
    metrics.update(evaluate_current_classification(z_train, train_meta, z_test, test_meta))
    y_tr_d = train_meta["d_mod3"].values
    y_te_d = test_meta["d_mod3"].values
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_train, y_tr_d)
    pred = ridge.predict(z_test)
    metrics.update(regression_metrics(y_te_d, pred))
    z_dim_std = z_train.std(axis=0)
    metrics.update({
        "z_std_test": float(z_test.std()),
        "z_active_dims_005": int((z_dim_std > 0.05).sum()),
    })
    for task in ["MCI_to_AD_2y", "CN_to_MCI_2y"]:
        train_c = get_valid_cohort(cohorts, "train", task)
        test_c = get_valid_cohort(cohorts, "test", task)
        tr_idx = train_c["feature_idx"].astype(int).values
        te_idx = test_c["feature_idx"].astype(int).values
        y_train = train_c["converted"].astype(int).values
        y_test = test_c["converted"].astype(int).values
        if len(y_train) >= 10 and y_train.sum() >= 3 and y_test.sum() >= 3:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(z_train[tr_idx], y_train)
            auc = roc_auc_score(y_test, clf.predict_proba(z_test[te_idx])[:, 1])
        else:
            auc = np.nan
        metrics["mci_conv_auc" if task == "MCI_to_AD_2y" else "cn_mci_conv_auc"] = auc
    return metrics


def train_eval_finetune(x_train, y_train, x_test, y_test, encoder_state, task, device,
                        epochs=60, batch_size=64, n_class=2):
    n_out = 1 if task in ("binary", "regression") else n_class
    base = DxPretrainNet(in_dim=x_train.shape[1], hidden=384, latent=256, n_out=3)
    base.encoder.load_state_dict(encoder_state)
    model = FinetuneNet(base.encoder, latent=256, n_out=n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_t = torch.FloatTensor(x_train).to(device)
    x_te_t = torch.FloatTensor(x_test).to(device)
    if task == "multiclass":
        y_t = torch.LongTensor(y_train).to(device)
    else:
        y_t = torch.FloatTensor(y_train).to(device)
    n = len(x_t)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            out, _ = model(x_t[b])
            if task == "binary":
                loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), y_t[b])
            elif task == "regression":
                loss = F.mse_loss(out.squeeze(-1), y_t[b])
            else:
                loss = F.cross_entropy(out, y_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        out, z = model(x_te_t)
        if task == "binary":
            return roc_auc_score(y_test, torch.sigmoid(out.squeeze(-1)).cpu().numpy())
        if task == "multiclass":
            p = F.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(1)
            aucs = []
            for c in range(n_class):
                try:
                    aucs.append(roc_auc_score((y_test == c).astype(int), p[:, c]))
                except Exception:
                    pass
            return accuracy_score(y_test, pred), float(np.mean(aucs)) if aucs else np.nan
        pred = out.squeeze(-1).cpu().numpy()
        m = regression_metrics(y_test, pred)
        m["alignment_spearman"] = spearmanr(np.linalg.norm(z.cpu().numpy() - z.cpu().numpy().mean(axis=0), axis=1), y_test)[0]
        return m


def eval_setup4(x_train, x_test, train_meta, test_meta, cohorts, encoder_state, device, epochs=60):
    metrics = {}
    # Binary current dx tasks.
    for key, neg, pos in [
        ("cn_ad_auc", "NORMAL", "AD"),
        ("cn_mci_cls_auc", "NORMAL", "MCI"),
        ("mci_ad_cls_auc", "MCI", "AD"),
    ]:
        tr = train_meta["dx"].isin([neg, pos]).values
        te = test_meta["dx"].isin([neg, pos]).values
        y_tr = (train_meta.loc[tr, "dx"] == pos).astype(int).values
        y_te = (test_meta.loc[te, "dx"] == pos).astype(int).values
        metrics[key] = train_eval_finetune(x_train[tr], y_tr, x_test[te], y_te, encoder_state, "binary", device, epochs=epochs)
    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    y_tr = train_meta["dx"].map(dx_map).fillna(-1).astype(int).values
    y_te = test_meta["dx"].map(dx_map).fillna(-1).astype(int).values
    tr = y_tr >= 0
    te = y_te >= 0
    acc, macro_auc = train_eval_finetune(x_train[tr], y_tr[tr], x_test[te], y_te[te], encoder_state, "multiclass", device, epochs=epochs, n_class=3)
    metrics["3class_acc"] = acc
    metrics["3class_macro_auc"] = macro_auc
    for task in ["MCI_to_AD_2y", "CN_to_MCI_2y"]:
        train_c = get_valid_cohort(cohorts, "train", task)
        test_c = get_valid_cohort(cohorts, "test", task)
        tr_idx = train_c["feature_idx"].astype(int).values
        te_idx = test_c["feature_idx"].astype(int).values
        y_train = train_c["converted"].astype(int).values
        y_test = test_c["converted"].astype(int).values
        metrics["mci_conv_auc" if task == "MCI_to_AD_2y" else "cn_mci_conv_auc"] = train_eval_finetune(
            x_train[tr_idx], y_train, x_test[te_idx], y_test, encoder_state, "binary", device, epochs=epochs
        )
    y_tr_d = train_meta["d_mod3"].values
    y_te_d = test_meta["d_mod3"].values
    metrics.update(train_eval_finetune(x_train, y_tr_d, x_test, y_te_d, encoder_state, "regression", device, epochs=epochs))
    return metrics


def summarize(df):
    metric_cols = [c for c in df.columns if c not in ["seed", "input_version", "setup", "pretrain_final_loss"]]
    rows = []
    for (version, setup), group in df.groupby(["input_version", "setup"]):
        row = {"input_version": version, "setup": setup, "n_seeds": len(group)}
        for c in metric_cols:
            row[f"{c}_mean"] = group[c].mean()
            row[f"{c}_std"] = group[c].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--finetune_epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="results_exp1_dx_pretrain")
    args = ap.parse_args()

    out_dir = DCON / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    data_by_version = load_raw_combat_data()
    cohorts = build_all_censored_cohorts(data_by_version, horizon_years=2)

    rows = []
    for version, data in data_by_version.items():
        x_train, x_test = scaled_features(data)
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== EXP1 version={version} seed={seed} ===", flush=True)
            model, losses = train_dx_pretrain(x_train, data["train_meta"], seed, device, epochs=args.epochs)
            model.eval()
            with torch.no_grad():
                z_train = model.encoder(torch.FloatTensor(x_train).to(device)).cpu().numpy()
                z_test = model.encoder(torch.FloatTensor(x_test).to(device)).cpu().numpy()
            encoder_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            s3 = eval_setup3(z_train, z_test, data["train_meta"], data["test_meta"], cohorts)
            s3.update({"input_version": version, "seed": seed, "setup": "dx_pretrain_setup3", "pretrain_final_loss": losses[-1]})
            rows.append(s3)
            s4 = eval_setup4(x_train, x_test, data["train_meta"], data["test_meta"], cohorts, encoder_state, device, epochs=args.finetune_epochs)
            s4.update({"input_version": version, "seed": seed, "setup": "dx_pretrain_setup4", "pretrain_final_loss": losses[-1]})
            rows.append(s4)
            pd.DataFrame(rows).to_csv(out_dir / "exp1_method2_dx_pretrain_per_seed.csv", index=False)
            print(f"done version={version} seed={seed} in {time.time()-t0:.1f}s", flush=True)

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(out_dir / "exp1_method2_dx_pretrain_per_seed.csv", index=False)
    summary.to_csv(out_dir / "exp1_method2_dx_pretrain_summary.csv", index=False)

    # Add direct comparison anchors from existing files when available.
    anchors = []
    baseline_path = DCON / "rank_sweep_with_ml_baselines.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        anchors.append(base)
    summary.to_csv(DCON / "exp1_method2_dx_pretrain.csv", index=False)
    print("\n=== EXP1 summary ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(out_dir / "exp1_method2_dx_pretrain_per_seed.csv")
    print(out_dir / "exp1_method2_dx_pretrain_summary.csv")
    print(DCON / "exp1_method2_dx_pretrain.csv")


if __name__ == "__main__":
    main()
