"""
minimal_v0_contrastive.py
==========================

Minimal proof-of-concept: validate contrastive learning with d_mod3 supervisor
adds value over supervised baseline on AD downstream tasks.

4 setups × 2 input versions × 4 tasks = 32 evaluations
Decision criterion: contrastive setups > Setup 2 (supervised) on majority of tasks.

Usage:
  python minimal_v0_contrastive.py \\
    --features_dir data/embeddings_128 \\
    --d_csv D_with_image_paths.csv \\
    --output_dir results_v0

Output:
  results_v0/
    setup1_logistic_metrics.csv
    setup2_supervised_metrics.csv  
    setup3_contrastive_metrics.csv
    setup4_contrastive_supervised_metrics.csv
    summary_table.csv
    summary_figure.png
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true, pred):
    """Consistent d_mod3 regression diagnostics."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    ss_res = ((y_true - pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson = pearsonr(y_true, pred)[0] if len(y_true) > 1 else np.nan
    spearman = spearmanr(y_true, pred)[0] if len(y_true) > 1 else np.nan
    return {
        'r2_d_mod3': r2,
        'pearson_d_mod3': pearson,
        'spearman_d_mod3': spearman,
    }


def alignment_spearman(z, y, n_pairs=5000, seed=42):
    """Spearman rho between pairwise latent distances and pairwise d_mod3 gaps."""
    z = np.asarray(z)
    y = np.asarray(y)
    if len(z) < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, len(z) * max(len(z) - 1, 1))
    ia = rng.integers(0, len(z), size=n_pairs)
    ib = rng.integers(0, len(z), size=n_pairs)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    if len(ia) < 3:
        return np.nan
    latent_dists = np.linalg.norm(z[ia] - z[ib], axis=1)
    d_dists = np.abs(y[ia] - y[ib])
    rho, _ = spearmanr(latent_dists, d_dists)
    return rho


def centroid_alignment_spearman(z, y):
    """Spearman rho between distance-to-centroid and d_mod3."""
    z = np.asarray(z)
    y = np.asarray(y)
    if len(z) < 3:
        return np.nan
    centroid = z.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(z - centroid, axis=1)
    rho, _ = spearmanr(dist, y)
    return rho


def binary_dx_auc(X_train, train_meta, X_test, test_meta, neg_dx, pos_dx):
    """Current-diagnosis binary AUC for two dx labels."""
    train_mask = train_meta['dx'].isin([neg_dx, pos_dx]).values
    test_mask = test_meta['dx'].isin([neg_dx, pos_dx]).values
    if train_mask.sum() < 10 or test_mask.sum() < 10:
        return np.nan
    y_train = (train_meta.loc[train_mask, 'dx'] == pos_dx).astype(int).values
    y_test = (test_meta.loc[test_mask, 'dx'] == pos_dx).astype(int).values
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return np.nan
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train[train_mask], y_train)
    p = clf.predict_proba(X_test[test_mask])[:, 1]
    return roc_auc_score(y_test, p)


def conversion_auc_from_score(score, meta, baseline_dx='MCI', target_dx='AD'):
    """AUC using a precomputed one-dimensional score for the conversion task."""
    conv = build_conversion_labels(meta, baseline_dx=baseline_dx, target_dx=target_dx)
    idx, y = first_dx_image_indices(meta, conv, baseline_dx=baseline_dx)
    y = np.asarray(y, dtype=int)
    if len(idx) < 5 or y.sum() < 3 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, np.asarray(score)[idx])


# ============================================================
# Data preparation
# ============================================================

def load_features_and_labels(features_dir, d_csv, version="raw"):
    """
    Load Swin 768-dim features and metadata.
    
    Args:
      features_dir: path to data/embeddings_128/
      d_csv: path to D_with_image_paths.csv
      version: "raw" or "combat"
    
    Returns dict with keys:
      train_features, test_features  (N, 768)
      train_ids, test_ids            (N,) image IDs
      train_meta, test_meta          DataFrame with RID, d_mod3, dx, MMSCORE, etc.
    """
    features_dir = Path(features_dir)
    
    if version == "raw":
        # Pre-ComBat raw 768-dim. The raw file stores all unique images;
        # use the established train/test image-id lists to split without leakage.
        Z_all = np.load(features_dir / "swin_latent.npy")
        all_ids = np.load(features_dir / "image_id_order.npy", allow_pickle=True)
        train_ids = np.load(features_dir / "swin_combat_train_ids.npy", allow_pickle=True)
        test_ids = np.load(features_dir / "swin_combat_test_ids.npy", allow_pickle=True)
        id_to_all_idx = {iid: i for i, iid in enumerate(all_ids)}
        train_keep = [iid for iid in train_ids if iid in id_to_all_idx]
        test_keep = [iid for iid in test_ids if iid in id_to_all_idx]
        Z_train = Z_all[[id_to_all_idx[iid] for iid in train_keep]]
        Z_test = Z_all[[id_to_all_idx[iid] for iid in test_keep]]
        train_ids = np.array(train_keep)
        test_ids = np.array(test_keep)
    elif version == "combat":
        # Post-ComBat 768-dim
        Z_train = np.load(features_dir / "swin_combat_train.npy")
        Z_test = np.load(features_dir / "swin_combat_test.npy")
        train_ids = np.load(features_dir / "swin_combat_train_ids.npy", allow_pickle=True)
        test_ids = np.load(features_dir / "swin_combat_test_ids.npy", allow_pickle=True)
    else:
        raise ValueError(f"Unknown version: {version}")
    
    print(f"[{version}] Train: {Z_train.shape}, Test: {Z_test.shape}")
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
    
    # Load metadata
    df = pd.read_csv(d_csv)
    df['EXAMDATE.x'] = pd.to_datetime(df['EXAMDATE.x'])
    
    # Match features with metadata. There can be multiple D visits per image;
    # keep one row per image to avoid assigning conflicting d_mod3 labels to the
    # same frozen embedding. Prefer the closest D visit to the scan date.
    image_col = "Image_Data_ID"
    if image_col not in df.columns and "Image Data ID" in df.columns:
        image_col = "Image Data ID"

    def one_row_per_image(ids):
        sub = df[df[image_col].isin(ids)].copy()
        sub["_abs_gap"] = sub["gap_days"].abs() if "gap_days" in sub.columns else 0
        sub = sub.sort_values([image_col, "_abs_gap", "EXAMDATE.x"])
        sub = sub.drop_duplicates(image_col, keep="first")
        return sub

    train_meta = one_row_per_image(train_ids)
    test_meta = one_row_per_image(test_ids)
    
    # Reorder to match feature array order
    train_id_to_idx = {iid: i for i, iid in enumerate(train_ids)}
    test_id_to_idx = {iid: i for i, iid in enumerate(test_ids)}
    
    train_meta['_idx'] = train_meta[image_col].map(train_id_to_idx)
    test_meta['_idx'] = test_meta[image_col].map(test_id_to_idx)
    
    train_meta = train_meta.dropna(subset=['_idx']).sort_values('_idx').reset_index(drop=True)
    test_meta = test_meta.dropna(subset=['_idx']).sort_values('_idx').reset_index(drop=True)
    
    # Subset features to matched metadata
    train_indices = train_meta['_idx'].astype(int).values
    test_indices = test_meta['_idx'].astype(int).values
    
    Z_train = Z_train[train_indices]
    Z_test = Z_test[test_indices]
    
    print(f"  After matching: train {Z_train.shape}, test {Z_test.shape}")
    print(f"  Unique train RIDs: {train_meta['RID'].nunique()}, test RIDs: {test_meta['RID'].nunique()}")
    print(f"  Train d_mod3 range: [{train_meta['d_mod3'].min():.2f}, {train_meta['d_mod3'].max():.2f}]")
    
    return {
        'train_features': Z_train,
        'test_features': Z_test,
        'train_meta': train_meta,
        'test_meta': test_meta,
        'full_meta': df,
    }


# ============================================================
# Setup 1: Logistic / Ridge baseline (no MLP train)
# ============================================================

def run_setup1(data, results_dict):
    """Logistic / Ridge directly on 768-dim features."""
    print("\n" + "=" * 60)
    print("SETUP 1: Logistic / Ridge baseline (no MLP)")
    print("=" * 60)
    
    X_train = data['train_features']
    X_test = data['test_features']
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    metrics = {}
    
    # Task A: CN vs AD binary
    print("\n--- Task A: CN vs AD ---")
    train_cnad = data['train_meta']['dx'].isin(['NORMAL', 'AD'])
    test_cnad = data['test_meta']['dx'].isin(['NORMAL', 'AD'])
    
    y_tr = (data['train_meta']['dx'] == 'AD').astype(int).values
    y_te = (data['test_meta']['dx'] == 'AD').astype(int).values
    
    if train_cnad.sum() > 10 and test_cnad.sum() > 10:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train_s[train_cnad], y_tr[train_cnad])
        p = clf.predict_proba(X_test_s[test_cnad])[:, 1]
        auc = roc_auc_score(y_te[test_cnad], p)
        metrics['cn_ad_auc'] = auc
        print(f"  CN/AD AUC: {auc:.3f}")
    
    print("\n--- Task A2: CN vs MCI current dx ---")
    auc_cn_mci_cls = binary_dx_auc(
        X_train_s, data['train_meta'], X_test_s, data['test_meta'],
        neg_dx='NORMAL', pos_dx='MCI'
    )
    metrics['cn_mci_cls_auc'] = auc_cn_mci_cls
    print(f"  CN/MCI current-dx AUC: {auc_cn_mci_cls:.3f}")
    
    print("\n--- Task A3: MCI vs AD current dx ---")
    auc_mci_ad_cls = binary_dx_auc(
        X_train_s, data['train_meta'], X_test_s, data['test_meta'],
        neg_dx='MCI', pos_dx='AD'
    )
    metrics['mci_ad_cls_auc'] = auc_mci_ad_cls
    print(f"  MCI/AD current-dx AUC: {auc_mci_ad_cls:.3f}")
    
    # Task B: 3-class
    print("\n--- Task B: 3-class CN/MCI/AD ---")
    dx_map = {'NORMAL': 0, 'MCI': 1, 'AD': 2}
    y_tr_3c = data['train_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    y_te_3c = data['test_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    
    tr_valid = y_tr_3c >= 0
    te_valid = y_te_3c >= 0
    
    clf3 = LogisticRegression(max_iter=2000, C=1.0, multi_class='ovr')
    clf3.fit(X_train_s[tr_valid], y_tr_3c[tr_valid])
    
    p3 = clf3.predict_proba(X_test_s[te_valid])
    pred = clf3.predict(X_test_s[te_valid])
    acc = accuracy_score(y_te_3c[te_valid], pred)
    
    # Macro AUC (one-vs-rest)
    aucs = []
    for c in range(3):
        try:
            a = roc_auc_score((y_te_3c[te_valid] == c).astype(int), p3[:, c])
            aucs.append(a)
        except Exception:
            pass
    macro_auc = np.mean(aucs) if aucs else np.nan
    metrics['3class_acc'] = acc
    metrics['3class_macro_auc'] = macro_auc
    print(f"  3-class accuracy: {acc:.3f}, macro AUC: {macro_auc:.3f}")
    
    # Task C: MCI conversion 2y
    print("\n--- Task C: MCI -> AD 2y conversion ---")
    auc_conv = compute_mci_conversion_auc(
        X_train_s, data['train_meta'], 
        X_test_s, data['test_meta'],
        method='logistic'
    )
    metrics['mci_conv_auc'] = auc_conv
    print(f"  MCI conversion AUC: {auc_conv:.3f}")
    
    # Task C2: CN -> MCI conversion 2y
    print("\n--- Task C2: CN -> MCI 2y conversion ---")
    auc_cn_mci = compute_conversion_auc(
        X_train_s, data['train_meta'],
        X_test_s, data['test_meta'],
        baseline_dx='NORMAL', target_dx='MCI',
        method='logistic'
    )
    metrics['cn_mci_conv_auc'] = auc_cn_mci
    print(f"  CN->MCI conversion AUC: {auc_cn_mci:.3f}")
    
    # Task D: R² on d_mod3
    print("\n--- Task D: R² on d_mod3 ---")
    y_tr_d = data['train_meta']['d_mod3'].values
    y_te_d = data['test_meta']['d_mod3'].values
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_tr_d)
    pred_d = ridge.predict(X_test_s)
    
    metrics.update(regression_metrics(y_te_d, pred_d))
    metrics['alignment_spearman'] = alignment_spearman(X_test_s, y_te_d)
    print(
        f"  R² d_mod3: {metrics['r2_d_mod3']:.3f}, "
        f"Pearson: {metrics['pearson_d_mod3']:.3f}, "
        f"Spearman: {metrics['spearman_d_mod3']:.3f}"
    )
    print(f"  Input distance alignment Spearman ρ: {metrics['alignment_spearman']:.3f}")
    
    results_dict['setup1'] = metrics
    return metrics


def compute_mci_conversion_auc(X_train, train_meta, X_test, test_meta, method='logistic'):
    return compute_conversion_auc(
        X_train, train_meta, X_test, test_meta,
        baseline_dx='MCI', target_dx='AD', method=method
    )


def compute_conversion_auc(X_train, train_meta, X_test, test_meta,
                           baseline_dx='MCI', target_dx='AD', method='logistic'):
    """Build conversion labels and compute AUC for first baseline-dx image per RID."""
    train_conv = build_conversion_labels(
        train_meta, baseline_dx=baseline_dx, target_dx=target_dx
    )
    test_conv = build_conversion_labels(
        test_meta, baseline_dx=baseline_dx, target_dx=target_dx
    )
    
    # Filter to the first available baseline-dx image per RID to avoid repeated
    # subject-level conversion labels.
    train_idx, train_y_conv = first_dx_image_indices(train_meta, train_conv, baseline_dx)
    test_idx, test_y_conv = first_dx_image_indices(test_meta, test_conv, baseline_dx)
    
    if len(train_idx) < 10 or len(test_idx) < 5:
        print(f"  Sample too small: train {baseline_dx} {len(train_idx)}, test {baseline_dx} {len(test_idx)}")
        return np.nan
    
    train_y = np.array(train_y_conv)
    test_y = np.array(test_y_conv)
    
    if train_y.sum() < 3 or test_y.sum() < 3:
        print(f"  Too few converters: train {train_y.sum()}, test {test_y.sum()}")
        return np.nan
    
    X_tr = X_train[train_idx]
    X_te = X_test[test_idx]
    
    if method == 'logistic':
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_tr, train_y)
        p = clf.predict_proba(X_te)[:, 1]
        return roc_auc_score(test_y, p)


def first_mci_image_indices(meta, conv_labels):
    return first_dx_image_indices(meta, conv_labels, baseline_dx='MCI')


def first_dx_image_indices(meta, conv_labels, baseline_dx):
    idxs, ys = [], []
    meta_sorted = meta.sort_values(['RID', 'EXAMDATE.x'])
    for rid, group in meta_sorted.groupby('RID'):
        if rid not in conv_labels:
            continue
        baseline_rows = group[group['dx'] == baseline_dx]
        if len(baseline_rows) == 0:
            continue
        idxs.append(baseline_rows.index[0])
        ys.append(conv_labels[rid])
    return idxs, ys


def build_conversion_labels(meta, horizon_years=2, baseline_dx='MCI', target_dx='AD'):
    """Per-subject: did baseline_dx convert to target_dx within horizon?"""
    meta = meta.sort_values(['RID', 'EXAMDATE.x'])
    conv = {}
    for rid, group in meta.groupby('RID'):
        first = group.iloc[0]
        if first['dx'] != baseline_dx:
            continue
        target_date = first['EXAMDATE.x'] + pd.DateOffset(years=horizon_years)
        followup = group[
            (group['EXAMDATE.x'] > first['EXAMDATE.x']) & 
            (group['EXAMDATE.x'] <= target_date)
        ]
        conv[rid] = int((followup['dx'] == target_dx).any())
    return conv


# ============================================================
# Setup 2: Supervised MLP (end-to-end per task)
# ============================================================

class SupervisedMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, latent=128, n_out=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, n_out)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


def run_setup2(data, results_dict, device='cpu', epochs=80, hidden=256, latent=128):
    """Supervised MLP end-to-end on each task."""
    print("\n" + "=" * 60)
    print("SETUP 2: Supervised MLP")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['train_features'])
    X_test = scaler.transform(data['test_features'])
    
    metrics = {}
    
    # Task A: CN/AD binary
    print("\n--- Task A: CN vs AD ---")
    train_cnad_mask = data['train_meta']['dx'].isin(['NORMAL', 'AD'])
    test_cnad_mask = data['test_meta']['dx'].isin(['NORMAL', 'AD'])
    
    y_tr = (data['train_meta']['dx'] == 'AD').astype(int).values
    y_te = (data['test_meta']['dx'] == 'AD').astype(int).values
    
    if train_cnad_mask.sum() > 10:
        auc = train_supervised_mlp(
            X_train[train_cnad_mask], y_tr[train_cnad_mask],
            X_test[test_cnad_mask], y_te[test_cnad_mask],
            task='binary', device=device, epochs=epochs,
            hidden=hidden, latent=latent
        )
        metrics['cn_ad_auc'] = auc
        print(f"  CN/AD AUC: {auc:.3f}")
    
    print("\n--- Task A2: CN vs MCI current dx ---")
    train_mask = data['train_meta']['dx'].isin(['NORMAL', 'MCI']).values
    test_mask = data['test_meta']['dx'].isin(['NORMAL', 'MCI']).values
    if train_mask.sum() > 10 and test_mask.sum() > 10:
        y_train = (data['train_meta'].loc[train_mask, 'dx'] == 'MCI').astype(int).values
        y_test = (data['test_meta'].loc[test_mask, 'dx'] == 'MCI').astype(int).values
        auc = train_supervised_mlp(
            X_train[train_mask], y_train,
            X_test[test_mask], y_test,
            task='binary', device=device, epochs=epochs,
            hidden=hidden, latent=latent
        )
        metrics['cn_mci_cls_auc'] = auc
        print(f"  CN/MCI current-dx AUC: {auc:.3f}")
    else:
        metrics['cn_mci_cls_auc'] = np.nan
    
    print("\n--- Task A3: MCI vs AD current dx ---")
    train_mask = data['train_meta']['dx'].isin(['MCI', 'AD']).values
    test_mask = data['test_meta']['dx'].isin(['MCI', 'AD']).values
    if train_mask.sum() > 10 and test_mask.sum() > 10:
        y_train = (data['train_meta'].loc[train_mask, 'dx'] == 'AD').astype(int).values
        y_test = (data['test_meta'].loc[test_mask, 'dx'] == 'AD').astype(int).values
        auc = train_supervised_mlp(
            X_train[train_mask], y_train,
            X_test[test_mask], y_test,
            task='binary', device=device, epochs=epochs,
            hidden=hidden, latent=latent
        )
        metrics['mci_ad_cls_auc'] = auc
        print(f"  MCI/AD current-dx AUC: {auc:.3f}")
    else:
        metrics['mci_ad_cls_auc'] = np.nan
    
    # Task B: 3-class
    print("\n--- Task B: 3-class ---")
    dx_map = {'NORMAL': 0, 'MCI': 1, 'AD': 2}
    y_tr_3c = data['train_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    y_te_3c = data['test_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    tr_v = y_tr_3c >= 0
    te_v = y_te_3c >= 0
    
    acc, macro_auc = train_supervised_mlp(
        X_train[tr_v], y_tr_3c[tr_v],
        X_test[te_v], y_te_3c[te_v],
        task='multiclass', n_class=3, device=device, epochs=epochs,
        hidden=hidden, latent=latent
    )
    metrics['3class_acc'] = acc
    metrics['3class_macro_auc'] = macro_auc
    print(f"  3-class acc: {acc:.3f}, macro AUC: {macro_auc:.3f}")
    
    # Task C: MCI conversion
    print("\n--- Task C: MCI conversion ---")
    train_conv = build_conversion_labels(data['train_meta'])
    test_conv = build_conversion_labels(data['test_meta'])
    
    tr_mci, tr_y = first_mci_image_indices(data['train_meta'], train_conv)
    te_mci, te_y = first_mci_image_indices(data['test_meta'], test_conv)
    
    if len(tr_mci) > 10 and sum(tr_y) > 3 and len(te_mci) > 5 and sum(te_y) > 3:
        auc = train_supervised_mlp(
            X_train[tr_mci], np.array(tr_y),
            X_test[te_mci], np.array(te_y),
            task='binary', device=device, epochs=epochs,
            hidden=hidden, latent=latent
        )
        metrics['mci_conv_auc'] = auc
        print(f"  MCI conversion AUC: {auc:.3f}")
    else:
        metrics['mci_conv_auc'] = np.nan
        print(f"  Insufficient sample")
    
    print("\n--- Task C2: CN -> MCI conversion ---")
    train_conv = build_conversion_labels(data['train_meta'], baseline_dx='NORMAL', target_dx='MCI')
    test_conv = build_conversion_labels(data['test_meta'], baseline_dx='NORMAL', target_dx='MCI')
    tr_cn, tr_y = first_dx_image_indices(data['train_meta'], train_conv, baseline_dx='NORMAL')
    te_cn, te_y = first_dx_image_indices(data['test_meta'], test_conv, baseline_dx='NORMAL')
    
    if len(tr_cn) > 10 and sum(tr_y) > 3 and len(te_cn) > 5 and sum(te_y) > 3:
        auc = train_supervised_mlp(
            X_train[tr_cn], np.array(tr_y),
            X_test[te_cn], np.array(te_y),
            task='binary', device=device, epochs=epochs,
            hidden=hidden, latent=latent
        )
        metrics['cn_mci_conv_auc'] = auc
        print(f"  CN->MCI conversion AUC: {auc:.3f}")
    else:
        metrics['cn_mci_conv_auc'] = np.nan
        print(f"  Insufficient sample")
    
    # Task D: R² d_mod3
    print("\n--- Task D: R² d_mod3 ---")
    y_tr_d = data['train_meta']['d_mod3'].values
    y_te_d = data['test_meta']['d_mod3'].values
    r2, pred_d, z_test_d = train_supervised_mlp(
        X_train, y_tr_d, X_test, y_te_d, task='regression', device=device,
        epochs=epochs, hidden=hidden, latent=latent, return_details=True
    )
    metrics['r2_d_mod3'] = r2
    metrics['pearson_d_mod3'] = pearsonr(y_te_d, pred_d)[0]
    metrics['spearman_d_mod3'] = spearmanr(y_te_d, pred_d)[0]
    metrics['alignment_spearman'] = alignment_spearman(z_test_d, y_te_d)
    print(
        f"  R² d_mod3: {r2:.3f}, "
        f"Pearson: {metrics['pearson_d_mod3']:.3f}, "
        f"Spearman: {metrics['spearman_d_mod3']:.3f}"
    )
    print(f"  Regression-MLP embedding alignment Spearman ρ: {metrics['alignment_spearman']:.3f}")
    
    results_dict['setup2'] = metrics
    return metrics


def train_supervised_mlp(X_tr, y_tr, X_te, y_te, task='binary', n_class=2, device='cpu',
                         epochs=80, batch_size=64, lr=1e-3, init_encoder_state=None,
                         hidden=256, latent=128, return_details=False):
    """Train supervised MLP, return metric."""
    n_out = 1 if task in ('binary', 'regression') else n_class
    
    model = SupervisedMLP(in_dim=X_tr.shape[1], hidden=hidden, latent=latent, n_out=n_out).to(device)
    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device) if task != 'multiclass' else torch.LongTensor(y_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    
    n = len(X_tr_t)
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            opt.zero_grad()
            out, _ = model(X_tr_t[batch_idx])
            
            if task == 'binary':
                loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), y_tr_t[batch_idx])
            elif task == 'multiclass':
                loss = F.cross_entropy(out, y_tr_t[batch_idx])
            elif task == 'regression':
                loss = F.mse_loss(out.squeeze(-1), y_tr_t[batch_idx])
            
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        out, z_te = model(X_te_t)
        if task == 'binary':
            p = torch.sigmoid(out.squeeze(-1)).cpu().numpy()
            return roc_auc_score(y_te, p)
        elif task == 'multiclass':
            p = F.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(1)
            acc = accuracy_score(y_te, pred)
            aucs = []
            for c in range(n_class):
                try:
                    a = roc_auc_score((y_te == c).astype(int), p[:, c])
                    aucs.append(a)
                except Exception:
                    pass
            return acc, np.mean(aucs) if aucs else np.nan
        elif task == 'regression':
            pred = out.squeeze(-1).cpu().numpy()
            ss_res = ((y_te - pred) ** 2).sum()
            ss_tot = ((y_te - y_te.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            if return_details:
                return r2, pred, z_te.cpu().numpy()
            return r2


def train_contrastive_finetune_mlp(X_tr, y_tr, X_te, y_te, encoder_state, task='binary',
                                   n_class=2, device='cpu', epochs=80, batch_size=64,
                                   lr=1e-3, hidden=256, latent=128, return_details=False):
    """Supervised finetune with an encoder initialized from ContrastiveMLP_v2."""
    n_out = 1 if task in ('binary', 'regression') else n_class
    encoder = ContrastiveMLP_v2(in_dim=X_tr.shape[1], hidden=hidden, latent=latent).encoder.to(device)
    encoder.load_state_dict(encoder_state)
    model = ContrastiveFinetuneMLP(encoder, latent=latent, n_out=n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device) if task != 'multiclass' else torch.LongTensor(y_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)

    n = len(X_tr_t)
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            opt.zero_grad()
            out, _ = model(X_tr_t[batch_idx])
            if task == 'binary':
                loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), y_tr_t[batch_idx])
            elif task == 'multiclass':
                loss = F.cross_entropy(out, y_tr_t[batch_idx])
            elif task == 'regression':
                loss = F.mse_loss(out.squeeze(-1), y_tr_t[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        out, z_te = model(X_te_t)
        if task == 'binary':
            p = torch.sigmoid(out.squeeze(-1)).cpu().numpy()
            return roc_auc_score(y_te, p)
        elif task == 'multiclass':
            p = F.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(1)
            acc = accuracy_score(y_te, pred)
            aucs = []
            for c in range(n_class):
                try:
                    aucs.append(roc_auc_score((y_te == c).astype(int), p[:, c]))
                except Exception:
                    pass
            return acc, np.mean(aucs) if aucs else np.nan
        elif task == 'regression':
            pred = out.squeeze(-1).cpu().numpy()
            ss_res = ((y_te - pred) ** 2).sum()
            ss_tot = ((y_te - y_te.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            if return_details:
                return r2, pred, z_te.cpu().numpy()
            return r2


# ============================================================
# Setup 3: Contrastive pretrain MLP + linear probe (your method)
# ============================================================

class ContrastiveMLP_v2(nn.Module):
    def __init__(self, in_dim=768, hidden=256, latent=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.BatchNorm1d(latent),
        )
        self.progression_head = nn.Linear(latent, 1)
    
    def forward(self, x):
        return self.encoder(x)

    def score(self, x):
        z = self.encoder(x)
        return z, self.progression_head(z).squeeze(-1)


class ContrastiveFinetuneMLP(nn.Module):
    def __init__(self, pretrained_encoder, latent=128, n_out=1):
        super().__init__()
        self.encoder = pretrained_encoder
        self.head = nn.Linear(latent, n_out)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


def y_aware_euclidean_loss(z, d, tau=1.0, beta=1.0):
    """
    Y-Aware-style soft continuous-label contrastive loss.
    
    z: (B, D) aligned latent
    d: (B,) continuous supervisor (d_mod3)
    
    Loss = cross_entropy(softmax(-latent_l2_sq / tau), softmax(-d_dist_sq * beta)).
    """
    B = z.shape[0]
    
    # Latent Euclidean similarity. Keep magnitude information for downstream
    # linear/Ridge probes instead of projecting onto cosine unit sphere.
    z_dist_sq = torch.cdist(z, z, p=2).pow(2)
    sim = -z_dist_sq / tau
    
    # Mask diagonal. Do not leave -inf in tensors that will later be multiplied
    # by zeros: in PyTorch, 0 * -inf becomes NaN.
    mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))
    
    # Predicted distribution
    log_p = F.log_softmax(sim, dim=1)
    log_p = log_p.masked_fill(mask, 0.0)
    
    # Target distribution from d distance
    d_dist_sq = (d.unsqueeze(0) - d.unsqueeze(1)) ** 2
    target_logits = -beta * d_dist_sq
    target_logits.masked_fill_(mask, float('-inf'))
    w = F.softmax(target_logits, dim=1)
    w = w.masked_fill(mask, 0.0)
    
    return -(w * log_p).sum(dim=1).mean()


def soft_kendall_loss(s, d, alpha=10.0):
    """Differentiable soft-Kendall rank loss between 1D score and d_mod3.

    NOTE: this is the *original* surrogate. It replaces the order indicator on
    the score side with tanh(alpha * (s_i - s_j)) using a hand-picked sharpness
    alpha=10. Because alpha is finite and fixed, tanh also responds to the
    *magnitude* of the score gap (not just its order), so this loss is not a
    purely ordinal/rank loss -- it leaks margin/magnitude information. Kept for
    backward-compatibility and as the ablation baseline against
    `kendall_loss_basic`.
    """
    s = s.view(-1)
    d = d.view(-1).detach()
    s_diff = s.unsqueeze(0) - s.unsqueeze(1)
    d_sign = torch.sign(d.unsqueeze(0) - d.unsqueeze(1))
    concord = torch.tanh(alpha * s_diff) * d_sign
    mask = ~torch.eye(s.numel(), dtype=torch.bool, device=s.device)
    return -concord[mask].mean()


def kendall_loss_basic(s, d, nu=None):
    """Most-basic smoothed Kendall's tau rank loss (Henderson 2026, eq. 11).

    Concordance to MAXIMIZE:
        mean_{i!=j}  g_nu(s_i - s_j) * ( I(d_i > d_j) - 1/2 ),
    where g_nu(x) = 1 / (1 + exp(-x / nu)) is the canonical sigmoid surrogate
    for the order indicator I(x > 0). Returned negated so it can be minimized.

    Key difference vs `soft_kendall_loss`: the smoothing scale `nu` is set by the
    paper's principled rule  nu = 0.1 * ||score scale||  so that g_nu closely
    approximates the indicator (g_nu(+0.25 sd) >= 0.99, g_nu(-0.25 sd) <= 0.01).
    This keeps the loss as close to *pure rank* as a differentiable surrogate
    allows, instead of using an arbitrary sharpness alpha that mixes in
    score magnitude. d enters only through its order (I(d_i > d_j)); for the
    continuous d_mod3 exact ties are negligible.
    """
    s = s.view(-1)
    d = d.view(-1).detach()
    s_diff = s.unsqueeze(0) - s.unsqueeze(1)
    if nu is None:
        # Principled default: nu = 0.1 * scale of the 1-D score
        # (paper's nu = 0.1 * ||beta|| adapted to the projected score s = x^T beta).
        nu = (0.1 * s.detach().std()).clamp_min(1e-6)
    g = torch.sigmoid(s_diff / nu)
    d_gt = (d.unsqueeze(0) > d.unsqueeze(1)).float() - 0.5
    mask = ~torch.eye(s.numel(), dtype=torch.bool, device=s.device)
    return -(g * d_gt)[mask].mean()


def train_contrastive_encoder(X_tr, d_tr, *, device='cpu', tau=0.1, beta=1.0,
                              epochs=150, batch_size=128, hidden=256, latent=128,
                              loss_mode='euclidean', rank_alpha=10.0,
                              lambda_rank=1.0, rank_nu=None, lr=1e-3,
                              weight_decay=1e-4, seed=42, verbose=True):
    """Stage-1 contrastive pretraining loop, factored out of run_setup3.

    Trains a ContrastiveMLP_v2 on already-standardized features X_tr with the
    continuous supervisor d_tr (d_mod3). The training math is identical to
    run_setup3 Stage 1 -- this is the single trainer reused by the 3-way
    conversion driver so the contrastive method does not diverge.

    X_tr: (N, P) standardized features (NaN d already filtered out by caller).
    d_tr: (N,) d_mod3 values.

    loss_mode:
      euclidean / rank_kendall[_basic] / hybrid[_basic] -- pairwise geometries
        that only constrain the RELATIVE structure of (z, s) w.r.t. d.
      regress_d -- Workstream-2 arm (c): direct MSE(s, d). A per-sample loss that
        trains the head to predict the absolute d value; exact under gradient
        accumulation (unlike the pairwise geometries).
    Returns (model, loss_history).
    """
    torch.manual_seed(seed)
    model = ContrastiveMLP_v2(in_dim=X_tr.shape[1], hidden=hidden, latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    d_tr_t = torch.FloatTensor(d_tr).to(device)
    n = len(X_tr_t)
    loss_history = []
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        total_loss = total_euclidean = total_rank = total_reg = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i + batch_size]
            if len(batch_idx) < 4:
                continue
            opt.zero_grad()
            z, s = model.score(X_tr_t[batch_idx])
            d_batch = d_tr_t[batch_idx]
            loss_euclidean = y_aware_euclidean_loss(z, d_batch, tau=tau, beta=beta)
            if loss_mode in ('rank_kendall_basic', 'hybrid_basic'):
                loss_rank = kendall_loss_basic(s, d_batch, nu=rank_nu)
            else:
                loss_rank = soft_kendall_loss(s, d_batch, alpha=rank_alpha)
            # Direct-regression supervision (Workstream 2, arm c): the 1-D head is
            # trained to PREDICT d (MSE), not just to order/space by it. This is a
            # per-sample loss, so unlike the pairwise geometries it is exact under
            # gradient accumulation -- the cleaner "internalize d" objective.
            loss_reg = nn.functional.mse_loss(s, d_batch)
            if loss_mode == 'euclidean':
                loss = loss_euclidean
            elif loss_mode in ('rank_kendall', 'rank_kendall_basic'):
                loss = loss_rank
            elif loss_mode in ('hybrid', 'hybrid_basic'):
                loss = loss_euclidean + lambda_rank * loss_rank
            elif loss_mode == 'regress_d':
                loss = loss_reg
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            total_euclidean += loss_euclidean.item()
            total_rank += loss_rank.item()
            total_reg += loss_reg.item()
            n_batches += 1
        loss_history.append({
            "epoch": epoch + 1,
            "loss": total_loss / n_batches if n_batches else np.nan,
            "loss_euclidean": total_euclidean / n_batches if n_batches else np.nan,
            "loss_rank": total_rank / n_batches if n_batches else np.nan,
            "loss_reg": total_reg / n_batches if n_batches else np.nan,
            "loss_mode": loss_mode, "tau": tau, "beta": beta,
            "rank_alpha": rank_alpha, "rank_nu": rank_nu, "lambda_rank": lambda_rank,
        })
        if verbose and ((epoch + 1) == 1 or (epoch + 1) % 25 == 0 or (epoch + 1) == epochs):
            h = loss_history[-1]
            print(f"    [{loss_mode}] epoch {epoch+1}/{epochs} loss={h['loss']:.4f} "
                  f"euclidean={h['loss_euclidean']:.4f} rank={h['loss_rank']:.4f}")
    return model, loss_history


def encode_features(model, X, device='cpu'):
    """Return (z, s): frozen 128-d latent and 1-D progression score as numpy."""
    model.eval()
    with torch.no_grad():
        z, s = model.score(torch.FloatTensor(X).to(device))
    return z.cpu().numpy(), s.cpu().numpy()


def finetune_encoder_classifier(pretrained_model, X_tr, y_tr, X_te, *, device='cpu',
                                latent=128, epochs=60, batch_size=64, lr=5e-4,
                                weight_decay=1e-4, seed=42):
    """Fine-tune the WHOLE pretrained encoder + a fresh binary head on (X_tr, y_tr).

    Counterpart to the frozen linear probe: instead of freezing the contrastive
    encoder, this unfreezes it and trains end-to-end with a conversion (BCE) head.
    A deep copy of the encoder is taken so every call restarts from the
    pretrained weights. Returns predicted P(convert) on X_te.
    """
    torch.manual_seed(seed)
    enc = copy.deepcopy(pretrained_model.encoder)
    clf = ContrastiveFinetuneMLP(enc, latent=latent, n_out=1).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    Xtr = torch.FloatTensor(X_tr).to(device)
    ytr = torch.FloatTensor(np.asarray(y_tr, dtype=float)).to(device)
    n = len(Xtr)
    for _ in range(epochs):
        clf.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            bi = idx[i:i + batch_size]
            if len(bi) < 2:               # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            logit, _ = clf(Xtr[bi])
            loss = lossfn(logit.squeeze(-1), ytr[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            opt.step()
    clf.eval()
    with torch.no_grad():
        logit, _ = clf(torch.FloatTensor(X_te).to(device))
        return torch.sigmoid(logit.squeeze(-1)).cpu().numpy()


def _build_global_feature_lut(features_dir, version):
    """image_id -> row index into a global feature matrix Z (returns Z, lut)."""
    features_dir = Path(features_dir)
    if version == "raw":
        Z = np.load(features_dir / "swin_latent.npy")
        ids = np.load(features_dir / "image_id_order.npy", allow_pickle=True)
    elif version == "combat":
        # ComBat features were stored under the OLD train/test partition; the new
        # 3-way split is a re-partition of the SAME images, so union them into one
        # lookup and slice per new split.
        Ztr = np.load(features_dir / "swin_combat_train.npy")
        Zte = np.load(features_dir / "swin_combat_test.npy")
        itr = np.load(features_dir / "swin_combat_train_ids.npy", allow_pickle=True)
        ite = np.load(features_dir / "swin_combat_test_ids.npy", allow_pickle=True)
        Z = np.concatenate([Ztr, Zte], axis=0)
        ids = np.concatenate([itr, ite], axis=0)
    else:
        raise ValueError(f"Unknown version: {version}")
    lut = {}
    for i, iid in enumerate(ids):
        lut.setdefault(iid, i)
    return Z, lut


def load_features_3way(features_dir, split_dir, d_csv, version="raw",
                       split_names=("contrastive", "finetune", "test")):
    """Load Swin features + metadata for a 3-way contrastive/finetune/test split.

    Each split is selected by its `{split}_image_ids.npy` list under split_dir.
    `meta` is one row per image and is aligned 1:1 with `features`, so d_mod3
    supervision can index directly into the returned feature array.

    Conversion labels need all longitudinal matched rows for a RID, not just one
    row per image. If split-specific matched CSVs are present, `long_meta` keeps
    those complete rows and adds `_feature_idx`, the image's row in `features`.
    Downstream conversion code should use `long_meta` for follow-up labels and
    `_feature_idx` to select the baseline image feature.

    Returns {split: {'features': (N,768), 'meta': DataFrame, 'image_ids': (N,)}}.
    """
    split_dir = Path(split_dir)
    Z, lut = _build_global_feature_lut(features_dir, version)
    df = pd.read_csv(d_csv)
    df["EXAMDATE.x"] = pd.to_datetime(df["EXAMDATE.x"])
    image_col = "Image_Data_ID" if "Image_Data_ID" in df.columns else "Image Data ID"

    out = {}
    for name in split_names:
        split_ids = np.load(split_dir / f"{name}_image_ids.npy", allow_pickle=True)
        keep = [iid for iid in split_ids if iid in lut]
        sub = df[df[image_col].isin(keep)].copy()
        sub["_abs_gap"] = sub["gap_days"].abs() if "gap_days" in sub.columns else 0
        sub = (sub.sort_values([image_col, "_abs_gap", "EXAMDATE.x"])
                  .drop_duplicates(image_col, keep="first")
                  .reset_index(drop=True))
        feat = Z[[lut[iid] for iid in sub[image_col].values]]
        image_to_feature_idx = {iid: i for i, iid in enumerate(sub[image_col].values)}

        split_csv = split_dir / f"matched_{name.upper()}.csv"
        if split_csv.exists():
            long_meta = pd.read_csv(split_csv)
            long_meta["EXAMDATE.x"] = pd.to_datetime(long_meta["EXAMDATE.x"])
            long_image_col = (
                "Image Data ID" if "Image Data ID" in long_meta.columns else "Image_Data_ID"
            )
            long_meta = long_meta[long_meta[long_image_col].isin(image_to_feature_idx)].copy()
            long_meta["_feature_idx"] = long_meta[long_image_col].map(image_to_feature_idx)
            long_meta = long_meta.dropna(subset=["_feature_idx"]).reset_index(drop=True)
            long_meta["_feature_idx"] = long_meta["_feature_idx"].astype(int)
        else:
            long_meta = sub.copy()
            long_meta["_feature_idx"] = np.arange(len(long_meta), dtype=int)

        out[name] = {"features": feat, "meta": sub, "long_meta": long_meta,
                     "image_ids": sub[image_col].values}
        n_conv_visits = len(sub)
        print(f"[{version}] {name:11s}: features {feat.shape}, RIDs "
              f"{sub['RID'].nunique()}, image rows {n_conv_visits}, "
              f"long rows {len(long_meta)}, d_mod3 valid "
              f"{int(sub['d_mod3'].notna().sum())}")

    # leakage tripwire: no RID may be shared across splits
    for a in split_names:
        for b in split_names:
            if a < b:
                ov = set(out[a]['meta']['RID']) & set(out[b]['meta']['RID'])
                if ov:
                    raise RuntimeError(
                        f"RID leakage between '{a}' and '{b}': {len(ov)} shared RIDs")
    print(f"[{version}] leakage check OK: contrastive/finetune/test RID-disjoint")
    return out


def load_features_2way(features_dir, master_dir, d_csv, version="raw"):
    """Load Swin features for the PREVIOUS 2-way train/test partition.

    Unlike the 3-way split, the encoder pretrain (d) AND the downstream head both
    use `train`; only `test` is held out. Same per-split schema as
    load_features_3way (features / meta / long_meta / image_ids), keyed by
    train/test. For drop-in compatibility with the 3-way drivers it ALSO aliases
    `contrastive` and `finetune` -> `train`, and sets `_mode='2way'` so those
    drivers collapse the two ridge d_hat configs into one (see the drivers).

    The train/test membership comes from `matched_TRAIN.csv` / `matched_TEST.csv`
    under master_dir (the same longitudinal tables used to build the embeddings).
    """
    master_dir = Path(master_dir)
    Z, lut = _build_global_feature_lut(features_dir, version)
    df = pd.read_csv(d_csv)
    df["EXAMDATE.x"] = pd.to_datetime(df["EXAMDATE.x"])
    image_col = "Image_Data_ID" if "Image_Data_ID" in df.columns else "Image Data ID"

    out = {}
    for name, fname in (("train", "matched_TRAIN.csv"), ("test", "matched_TEST.csv")):
        long_meta = pd.read_csv(master_dir / fname)
        long_meta["EXAMDATE.x"] = pd.to_datetime(long_meta["EXAMDATE.x"])
        long_image_col = (
            "Image Data ID" if "Image Data ID" in long_meta.columns else "Image_Data_ID"
        )
        keep = [iid for iid in long_meta[long_image_col].unique() if iid in lut]
        sub = df[df[image_col].isin(keep)].copy()
        sub["_abs_gap"] = sub["gap_days"].abs() if "gap_days" in sub.columns else 0
        sub = (sub.sort_values([image_col, "_abs_gap", "EXAMDATE.x"])
                  .drop_duplicates(image_col, keep="first")
                  .reset_index(drop=True))
        feat = Z[[lut[iid] for iid in sub[image_col].values]]
        image_to_feature_idx = {iid: i for i, iid in enumerate(sub[image_col].values)}

        long_meta = long_meta[long_meta[long_image_col].isin(image_to_feature_idx)].copy()
        long_meta["_feature_idx"] = long_meta[long_image_col].map(image_to_feature_idx)
        long_meta = long_meta.dropna(subset=["_feature_idx"]).reset_index(drop=True)
        long_meta["_feature_idx"] = long_meta["_feature_idx"].astype(int)

        out[name] = {"features": feat, "meta": sub, "long_meta": long_meta,
                     "image_ids": sub[image_col].values}
        print(f"[{version}] {name:5s}: features {feat.shape}, RIDs "
              f"{sub['RID'].nunique()}, image rows {len(sub)}, "
              f"long rows {len(long_meta)}, d_mod3 valid "
              f"{int(sub['d_mod3'].notna().sum())}")

    overlap = set(out["train"]["meta"]["RID"]) & set(out["test"]["meta"]["RID"])
    if overlap:
        raise RuntimeError(f"RID leakage between train and test: {len(overlap)} shared RIDs")
    print(f"[{version}] leakage check OK: train/test RID-disjoint")

    # aliases so the 3-way drivers run unchanged; _mode tells them to collapse ridge.
    out["contrastive"] = out["train"]
    out["finetune"] = out["train"]
    out["_mode"] = "2way"
    return out


def save_setup3_diagnostic_plots(z_test, y_te_d, output_dir, version):
    """Save PCA/t-SNE/pairwise diagnostics for the contrastive test embedding."""
    paths = {}
    output_dir = Path(output_dir)
    tag = f"_{version}" if version else ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z2 = PCA(n_components=2, random_state=42).fit_transform(z_test)
        p = output_dir / f"setup3_pca_d_mod3{tag}.png"
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(z2[:, 0], z2[:, 1], c=y_te_d, s=12, cmap="viridis", alpha=0.85)
        plt.colorbar(sc, label="d_mod3")
        plt.title(f"Setup3 PCA colored by d_mod3 ({version})")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["pca_plot"] = str(p)

        n_pairs = min(5000, len(z_test) * max(len(z_test) - 1, 1))
        rng = np.random.default_rng(42)
        ia = rng.integers(0, len(z_test), size=n_pairs)
        ib = rng.integers(0, len(z_test), size=n_pairs)
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]
        latent_d = np.linalg.norm(z_test[ia] - z_test[ib], axis=1)
        d_d = np.abs(y_te_d[ia] - y_te_d[ib])
        p = output_dir / f"setup3_pairwise_distance{tag}.png"
        plt.figure(figsize=(6, 5))
        plt.scatter(d_d, latent_d, s=6, alpha=0.25)
        plt.xlabel("|d_mod3_i - d_mod3_j|")
        plt.ylabel("||z_i - z_j||")
        plt.title(f"Pairwise distance diagnostic ({version})")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["pairwise_plot"] = str(p)

        try:
            from sklearn.manifold import TSNE
            z_tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(z_test)
            p = output_dir / f"setup3_tsne_d_mod3{tag}.png"
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_te_d, s=12, cmap="viridis", alpha=0.85)
            plt.colorbar(sc, label="d_mod3")
            plt.title(f"Setup3 t-SNE colored by d_mod3 ({version})")
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            paths["tsne_plot"] = str(p)
        except Exception as e:
            paths["tsne_plot_error"] = str(e)
    except Exception as e:
        paths["plot_error"] = str(e)
    return paths


def run_setup3(data, results_dict, device='cpu', tau=0.1, beta=1.0, epochs=150,
               batch_size=128, output_dir=None, version="", hidden=256, latent=128,
               uniformity_weight=0.0, loss_mode='euclidean', rank_alpha=10.0,
               lambda_rank=1.0, rank_nu=None):
    """Contrastive pretrain MLP + frozen linear probe per task."""
    print("\n" + "=" * 60)
    print("SETUP 3: Contrastive MLP + Linear probe (YOUR METHOD)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['train_features'])
    X_test = scaler.transform(data['test_features'])
    
    # Filter train with valid d_mod3
    train_meta = data['train_meta']
    valid_d = train_meta['d_mod3'].notna()
    X_tr = X_train[valid_d.values]
    d_tr = train_meta.loc[valid_d, 'd_mod3'].values
    
    print(f"  Contrastive training: {len(X_tr)} images with d_mod3")
    print(
        f"  loss_mode={loss_mode}, tau={tau}, beta={beta}, rank_alpha={rank_alpha}, "
        f"rank_nu={rank_nu}, lambda_rank={lambda_rank}, epochs={epochs}, "
        f"batch_size={batch_size}"
    )
    
    # Stage 1: Contrastive pretrain
    model = ContrastiveMLP_v2(in_dim=X_train.shape[1], hidden=hidden, latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    d_tr_t = torch.FloatTensor(d_tr).to(device)
    
    n = len(X_tr_t)
    print("\n  Stage 1: Contrastive pretrain...")
    t0 = time.time()
    loss_history = []
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        total_loss = 0
        total_euclidean = 0
        total_rank = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            if len(batch_idx) < 4:
                continue
            
            opt.zero_grad()
            z, s = model.score(X_tr_t[batch_idx])
            d_batch = d_tr_t[batch_idx]
            loss_euclidean = y_aware_euclidean_loss(z, d_batch, tau=tau, beta=beta)
            # Pick the rank surrogate: the principled basic Kendall (sigmoid + nu)
            # for the *_basic modes, otherwise the original tanh+alpha surrogate.
            if loss_mode in ('rank_kendall_basic', 'hybrid_basic'):
                loss_rank = kendall_loss_basic(s, d_batch, nu=rank_nu)
            else:
                loss_rank = soft_kendall_loss(s, d_batch, alpha=rank_alpha)
            if loss_mode == 'euclidean':
                loss = loss_euclidean
            elif loss_mode in ('rank_kendall', 'rank_kendall_basic'):
                loss = loss_rank
            elif loss_mode in ('hybrid', 'hybrid_basic'):
                loss = loss_euclidean + lambda_rank * loss_rank
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item()
            total_euclidean += loss_euclidean.item()
            total_rank += loss_rank.item()
            n_batches += 1
        
        epoch_loss = total_loss / n_batches if n_batches else np.nan
        epoch_euclidean = total_euclidean / n_batches if n_batches else np.nan
        epoch_rank = total_rank / n_batches if n_batches else np.nan
        loss_history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "loss_euclidean": epoch_euclidean,
            "loss_rank": epoch_rank,
            "loss_mode": loss_mode,
            "tau": tau,
            "beta": beta,
            "rank_alpha": rank_alpha,
            "rank_nu": rank_nu,
            "lambda_rank": lambda_rank,
        })
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(
                f"    Epoch {epoch+1}/{epochs}, loss: {epoch_loss:.4f}, "
                f"euclidean: {epoch_euclidean:.4f}, rank: {epoch_rank:.4f}"
            )
    
    t_train = time.time() - t0
    print(f"  Pretrain done in {t_train:.1f}s")
    
    # Stage 2: Extract frozen latent, linear probe per task
    print("\n  Stage 2: Frozen MLP + linear probe...")
    model.eval()
    with torch.no_grad():
        z_train_t, s_train_t = model.score(torch.FloatTensor(X_train).to(device))
        z_test_t, s_test_t = model.score(torch.FloatTensor(X_test).to(device))
        z_train = z_train_t.cpu().numpy()
        z_test = z_test_t.cpu().numpy()
        s_train = s_train_t.cpu().numpy()
        s_test = s_test_t.cpu().numpy()
    z_std_train = float(z_train.std())
    z_std_test = float(z_test.std())
    z_dim_std = z_train.std(axis=0)
    z_active_dims_005 = int((z_dim_std > 0.05).sum())
    z_active_dims_010 = int((z_dim_std > 0.10).sum())
    z_active_dims_050 = int((z_dim_std > 0.50).sum())
    print(f"  z.std train/test: {z_std_train:.4f} / {z_std_test:.4f}")
    print(
        f"  z dim std mean/max, active >0.05/>0.10/>0.50: "
        f"{float(z_dim_std.mean()):.4f}/{float(z_dim_std.max()):.4f}, "
        f"{z_active_dims_005}/{z_active_dims_010}/{z_active_dims_050}"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        tag = f"_{version}" if version else ""
        np.save(output_dir / f"setup3_contrastive_z_train{tag}.npy", z_train)
        np.save(output_dir / f"setup3_contrastive_z_test{tag}.npy", z_test)
        np.save(output_dir / f"setup3_progression_s_train{tag}.npy", s_train)
        np.save(output_dir / f"setup3_progression_s_test{tag}.npy", s_test)
        pd.DataFrame(loss_history).to_csv(
            output_dir / f"setup3_contrastive_loss_history{tag}.csv", index=False
        )
        data['train_meta'].to_csv(output_dir / f"setup3_contrastive_train_meta{tag}.csv", index=False)
        data['test_meta'].to_csv(output_dir / f"setup3_contrastive_test_meta{tag}.csv", index=False)
    
    # Linear probe on each task
    metrics = {
        'loss_mode': loss_mode,
        'tau': tau,
        'beta': beta,
        'rank_alpha': rank_alpha,
        'rank_nu': rank_nu,
        'lambda_rank': lambda_rank,
        'final_soft_tau': tau,
        'z_std_train': z_std_train,
        'z_std_test': z_std_test,
        'z_dim_std_mean': float(z_dim_std.mean()),
        'z_dim_std_max': float(z_dim_std.max()),
        'z_active_dims_005': z_active_dims_005,
        'z_active_dims_010': z_active_dims_010,
        'z_active_dims_050': z_active_dims_050,
    }
    
    # Task A: CN/AD
    train_cnad_mask = data['train_meta']['dx'].isin(['NORMAL', 'AD']).values
    test_cnad_mask = data['test_meta']['dx'].isin(['NORMAL', 'AD']).values
    y_tr = (data['train_meta']['dx'] == 'AD').astype(int).values
    y_te = (data['test_meta']['dx'] == 'AD').astype(int).values
    
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(z_train[train_cnad_mask], y_tr[train_cnad_mask])
    p = clf.predict_proba(z_test[test_cnad_mask])[:, 1]
    metrics['cn_ad_auc'] = roc_auc_score(y_te[test_cnad_mask], p)
    print(f"  Task A CN/AD AUC: {metrics['cn_ad_auc']:.3f}")
    
    metrics['cn_mci_cls_auc'] = binary_dx_auc(
        z_train, data['train_meta'], z_test, data['test_meta'],
        neg_dx='NORMAL', pos_dx='MCI'
    )
    print(f"  Task A2 CN/MCI current-dx AUC: {metrics['cn_mci_cls_auc']:.3f}")
    
    metrics['mci_ad_cls_auc'] = binary_dx_auc(
        z_train, data['train_meta'], z_test, data['test_meta'],
        neg_dx='MCI', pos_dx='AD'
    )
    print(f"  Task A3 MCI/AD current-dx AUC: {metrics['mci_ad_cls_auc']:.3f}")
    
    # Task B: 3-class
    dx_map = {'NORMAL': 0, 'MCI': 1, 'AD': 2}
    y_tr_3c = data['train_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    y_te_3c = data['test_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    tr_v = y_tr_3c >= 0
    te_v = y_te_3c >= 0
    
    clf3 = LogisticRegression(max_iter=2000, C=1.0, multi_class='ovr')
    clf3.fit(z_train[tr_v], y_tr_3c[tr_v])
    p3 = clf3.predict_proba(z_test[te_v])
    pred3 = clf3.predict(z_test[te_v])
    metrics['3class_acc'] = accuracy_score(y_te_3c[te_v], pred3)
    aucs = []
    for c in range(3):
        try:
            aucs.append(roc_auc_score((y_te_3c[te_v] == c).astype(int), p3[:, c]))
        except Exception:
            pass
    metrics['3class_macro_auc'] = np.mean(aucs) if aucs else np.nan
    print(f"  Task B 3-class acc: {metrics['3class_acc']:.3f}, macro AUC: {metrics['3class_macro_auc']:.3f}")
    
    # Task C: MCI conv
    auc_conv = compute_mci_conversion_auc(
        z_train, data['train_meta'], z_test, data['test_meta']
    )
    metrics['mci_conv_auc'] = auc_conv
    print(f"  Task C MCI conv AUC: {auc_conv:.3f}")
    
    auc_cn_mci = compute_conversion_auc(
        z_train, data['train_meta'], z_test, data['test_meta'],
        baseline_dx='NORMAL', target_dx='MCI'
    )
    metrics['cn_mci_conv_auc'] = auc_cn_mci
    print(f"  Task C2 CN->MCI conv AUC: {auc_cn_mci:.3f}")
    
    # Task D: R² d_mod3
    y_tr_d = data['train_meta']['d_mod3'].values
    y_te_d = data['test_meta']['d_mod3'].values
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_train, y_tr_d)
    pred_d = ridge.predict(z_test)
    metrics.update(regression_metrics(y_te_d, pred_d))
    print(
        f"  Task D R² d_mod3: {metrics['r2_d_mod3']:.3f}, "
        f"Pearson: {metrics['pearson_d_mod3']:.3f}, "
        f"Spearman: {metrics['spearman_d_mod3']:.3f}"
    )
    
    # Alignment metric: Spearman ρ between latent dist and d dist
    rho = alignment_spearman(z_test, y_te_d)
    metrics['alignment_spearman'] = rho
    metrics['centroid_alignment_spearman'] = centroid_alignment_spearman(z_test, y_te_d)
    print(f"  Alignment Spearman ρ: {rho:.3f}")
    print(f"  Centroid alignment Spearman ρ: {metrics['centroid_alignment_spearman']:.3f}")

    s_metrics = {
        'loss_mode': loss_mode,
        'tau': tau,
        'beta': beta,
        'rank_alpha': rank_alpha,
        'rank_nu': rank_nu,
        'lambda_rank': lambda_rank,
        'final_soft_tau': tau,
        'mci_conv_auc': conversion_auc_from_score(
            s_test, data['test_meta'], baseline_dx='MCI', target_dx='AD'
        ),
        'cn_mci_conv_auc': conversion_auc_from_score(
            s_test, data['test_meta'], baseline_dx='NORMAL', target_dx='MCI'
        ),
        'spearman_d_mod3': spearmanr(y_te_d, s_test)[0],
        'pearson_d_mod3': pearsonr(y_te_d, s_test)[0],
    }
    print(
        f"  Setup3-s direct score: MCI conv AUC={s_metrics['mci_conv_auc']:.3f}, "
        f"CN->MCI conv AUC={s_metrics['cn_mci_conv_auc']:.3f}, "
        f"d Spearman={s_metrics['spearman_d_mod3']:.3f}"
    )

    if output_dir is not None:
        plot_paths = save_setup3_diagnostic_plots(z_test, y_te_d, output_dir, version)
        for k, v in plot_paths.items():
            metrics[k] = v
            print(f"  {k}: {v}")
    
    results_dict['setup3'] = metrics
    results_dict['setup3-s'] = s_metrics
    return metrics, model


# ============================================================
# Setup 4: Contrastive pretrain MLP + supervised finetune
# ============================================================

def run_setup4(data, results_dict, pretrained_model, device='cpu', epochs=60,
               hidden=256, latent=128):
    """Initialize supervised MLP from contrastive encoder, then finetune per task."""
    print("\n" + "=" * 60)
    print("SETUP 4: Contrastive MLP + Supervised finetune")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['train_features'])
    X_test = scaler.transform(data['test_features'])
    
    init_encoder_state = {
        k: v.detach().clone().to(device)
        for k, v in pretrained_model.encoder.state_dict().items()
    }
    
    metrics = {}
    
    # Task A: CN/AD binary
    print("\n--- Task A: CN vs AD ---")
    train_cnad_mask = data['train_meta']['dx'].isin(['NORMAL', 'AD'])
    test_cnad_mask = data['test_meta']['dx'].isin(['NORMAL', 'AD'])
    
    y_tr = (data['train_meta']['dx'] == 'AD').astype(int).values
    y_te = (data['test_meta']['dx'] == 'AD').astype(int).values
    
    if train_cnad_mask.sum() > 10:
        auc = train_contrastive_finetune_mlp(
            X_train[train_cnad_mask], y_tr[train_cnad_mask],
            X_test[test_cnad_mask], y_te[test_cnad_mask],
            task='binary', device=device, epochs=epochs,
            encoder_state=init_encoder_state,
            hidden=hidden, latent=latent
        )
        metrics['cn_ad_auc'] = auc
        print(f"  CN/AD AUC: {auc:.3f}")
    
    print("\n--- Task A2: CN vs MCI current dx ---")
    train_mask = data['train_meta']['dx'].isin(['NORMAL', 'MCI']).values
    test_mask = data['test_meta']['dx'].isin(['NORMAL', 'MCI']).values
    if train_mask.sum() > 10 and test_mask.sum() > 10:
        y_train = (data['train_meta'].loc[train_mask, 'dx'] == 'MCI').astype(int).values
        y_test = (data['test_meta'].loc[test_mask, 'dx'] == 'MCI').astype(int).values
        auc = train_contrastive_finetune_mlp(
            X_train[train_mask], y_train,
            X_test[test_mask], y_test,
            task='binary', device=device, epochs=epochs,
            encoder_state=init_encoder_state,
            hidden=hidden, latent=latent
        )
        metrics['cn_mci_cls_auc'] = auc
        print(f"  CN/MCI current-dx AUC: {auc:.3f}")
    else:
        metrics['cn_mci_cls_auc'] = np.nan
    
    print("\n--- Task A3: MCI vs AD current dx ---")
    train_mask = data['train_meta']['dx'].isin(['MCI', 'AD']).values
    test_mask = data['test_meta']['dx'].isin(['MCI', 'AD']).values
    if train_mask.sum() > 10 and test_mask.sum() > 10:
        y_train = (data['train_meta'].loc[train_mask, 'dx'] == 'AD').astype(int).values
        y_test = (data['test_meta'].loc[test_mask, 'dx'] == 'AD').astype(int).values
        auc = train_contrastive_finetune_mlp(
            X_train[train_mask], y_train,
            X_test[test_mask], y_test,
            task='binary', device=device, epochs=epochs,
            encoder_state=init_encoder_state,
            hidden=hidden, latent=latent
        )
        metrics['mci_ad_cls_auc'] = auc
        print(f"  MCI/AD current-dx AUC: {auc:.3f}")
    else:
        metrics['mci_ad_cls_auc'] = np.nan
    
    # Task B: 3-class
    print("\n--- Task B: 3-class ---")
    dx_map = {'NORMAL': 0, 'MCI': 1, 'AD': 2}
    y_tr_3c = data['train_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    y_te_3c = data['test_meta']['dx'].map(dx_map).fillna(-1).astype(int).values
    tr_v = y_tr_3c >= 0
    te_v = y_te_3c >= 0
    
    acc, macro_auc = train_contrastive_finetune_mlp(
        X_train[tr_v], y_tr_3c[tr_v],
        X_test[te_v], y_te_3c[te_v],
        task='multiclass', n_class=3, device=device, epochs=epochs,
        encoder_state=init_encoder_state,
        hidden=hidden, latent=latent
    )
    metrics['3class_acc'] = acc
    metrics['3class_macro_auc'] = macro_auc
    print(f"  3-class acc: {acc:.3f}, macro AUC: {macro_auc:.3f}")
    
    # Task C: MCI conversion
    print("\n--- Task C: MCI conversion ---")
    train_conv = build_conversion_labels(data['train_meta'])
    test_conv = build_conversion_labels(data['test_meta'])
    
    tr_mci, tr_y = first_mci_image_indices(data['train_meta'], train_conv)
    te_mci, te_y = first_mci_image_indices(data['test_meta'], test_conv)
    
    if len(tr_mci) > 10 and sum(tr_y) > 3 and len(te_mci) > 5 and sum(te_y) > 3:
        auc = train_contrastive_finetune_mlp(
            X_train[tr_mci], np.array(tr_y),
            X_test[te_mci], np.array(te_y),
            task='binary', device=device, epochs=epochs,
            encoder_state=init_encoder_state,
            hidden=hidden, latent=latent
        )
        metrics['mci_conv_auc'] = auc
        print(f"  MCI conversion AUC: {auc:.3f}")
    else:
        metrics['mci_conv_auc'] = np.nan
        print(f"  Insufficient sample")
    
    print("\n--- Task C2: CN -> MCI conversion ---")
    train_conv = build_conversion_labels(data['train_meta'], baseline_dx='NORMAL', target_dx='MCI')
    test_conv = build_conversion_labels(data['test_meta'], baseline_dx='NORMAL', target_dx='MCI')
    tr_cn, tr_y = first_dx_image_indices(data['train_meta'], train_conv, baseline_dx='NORMAL')
    te_cn, te_y = first_dx_image_indices(data['test_meta'], test_conv, baseline_dx='NORMAL')
    
    if len(tr_cn) > 10 and sum(tr_y) > 3 and len(te_cn) > 5 and sum(te_y) > 3:
        auc = train_contrastive_finetune_mlp(
            X_train[tr_cn], np.array(tr_y),
            X_test[te_cn], np.array(te_y),
            task='binary', device=device, epochs=epochs,
            encoder_state=init_encoder_state,
            hidden=hidden, latent=latent
        )
        metrics['cn_mci_conv_auc'] = auc
        print(f"  CN->MCI conversion AUC: {auc:.3f}")
    else:
        metrics['cn_mci_conv_auc'] = np.nan
        print(f"  Insufficient sample")
    
    # Task D: R² d_mod3
    print("\n--- Task D: R² d_mod3 ---")
    y_tr_d = data['train_meta']['d_mod3'].values
    y_te_d = data['test_meta']['d_mod3'].values
    r2, pred_d, z_test_d = train_contrastive_finetune_mlp(
        X_train, y_tr_d, X_test, y_te_d,
        task='regression', device=device, epochs=epochs,
        encoder_state=init_encoder_state,
        hidden=hidden, latent=latent, return_details=True
    )
    metrics['r2_d_mod3'] = r2
    metrics['pearson_d_mod3'] = pearsonr(y_te_d, pred_d)[0]
    metrics['spearman_d_mod3'] = spearmanr(y_te_d, pred_d)[0]
    metrics['alignment_spearman'] = alignment_spearman(z_test_d, y_te_d)
    print(
        f"  R² d_mod3: {r2:.3f}, "
        f"Pearson: {metrics['pearson_d_mod3']:.3f}, "
        f"Spearman: {metrics['spearman_d_mod3']:.3f}"
    )
    print(f"  Finetuned regression embedding alignment Spearman ρ: {metrics['alignment_spearman']:.3f}")
    
    results_dict['setup4'] = metrics
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="../data/embeddings_128")
    parser.add_argument("--d_csv", default="../data/master_smri/D_with_image_paths_full.csv")
    parser.add_argument("--output_dir", default="results_v0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--versions", nargs='+', default=["raw", "combat"])
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--loss_mode",
                        choices=["euclidean", "rank_kendall", "hybrid",
                                 "rank_kendall_basic", "hybrid_basic"],
                        default="euclidean",
                        help="*_basic use the principled smoothed Kendall "
                             "(sigmoid + nu); others use the original tanh+alpha.")
    parser.add_argument("--rank_alpha", type=float, default=10.0,
                        help="Sharpness for the original soft_kendall_loss "
                             "(rank_kendall / hybrid only).")
    parser.add_argument("--rank_nu", type=float, default=None,
                        help="Smoothing scale for kendall_loss_basic. None => "
                             "principled default nu = 0.1 * std(score).")
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--supervised_epochs", type=int, default=80)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_latent", type=int, default=128)
    parser.add_argument("--uniformity_weight", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny CPU-friendly smoke test with few epochs.")
    args = parser.parse_args()
    
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.supervised_epochs = min(args.supervised_epochs, 2)
        args.output_dir = args.output_dir if args.output_dir != "results_v0" else "results_smoke"
        print("SMOKE MODE: contrastive epochs <= 2, supervised epochs <= 2")

    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"Device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for version in args.versions:
        print(f"\n{'='*70}")
        print(f"INPUT VERSION: {version}")
        print(f"{'='*70}")
        
        try:
            data = load_features_and_labels(args.features_dir, args.d_csv, version=version)
        except FileNotFoundError as e:
            print(f"  Cannot load {version} features: {e}")
            continue
        
        version_results = {}
        run_setup1(data, version_results)
        run_setup2(data, version_results, device=device, epochs=args.supervised_epochs,
                   hidden=args.mlp_hidden, latent=args.mlp_latent)
        _, contrastive_model = run_setup3(data, version_results, device=device,
                                          tau=args.tau, beta=args.beta, epochs=args.epochs,
                                          output_dir=args.output_dir, version=version,
                                          hidden=args.mlp_hidden, latent=args.mlp_latent,
                                          uniformity_weight=args.uniformity_weight,
                                          loss_mode=args.loss_mode,
                                          rank_alpha=args.rank_alpha,
                                          lambda_rank=args.lambda_rank,
                                          rank_nu=args.rank_nu)
        run_setup4(data, version_results, pretrained_model=contrastive_model,
                   device=device, epochs=args.supervised_epochs,
                   hidden=args.mlp_hidden, latent=args.mlp_latent)
        
        all_results[version] = version_results
        
        # Save intermediate
        with open(f"{args.output_dir}/results_{version}.json", "w") as f:
            json.dump(version_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Final summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    
    rows = []
    for version, vres in all_results.items():
        for setup, metrics in vres.items():
            row = {'input_version': version, 'setup': setup}
            row.update(metrics)
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{args.output_dir}/summary_table.csv", index=False)
    print(summary_df.to_string())
    
    # Decision criterion
    print(f"\n{'='*70}")
    print(f"DECISION: Method useful?")
    print(f"{'='*70}")
    
    for version in args.versions:
        if version not in all_results:
            continue
        s2 = all_results[version].get('setup2', {})
        s3 = all_results[version].get('setup3', {})
        s4 = all_results[version].get('setup4', {})
        
        print(f"\nInput: {version}")
        for metric in ['cn_ad_auc', 'cn_mci_cls_auc', 'mci_ad_cls_auc',
                       '3class_macro_auc', 'mci_conv_auc', 'cn_mci_conv_auc',
                       'r2_d_mod3']:
            v2 = s2.get(metric, np.nan)
            v3 = s3.get(metric, np.nan)
            if not (np.isnan(v2) or np.isnan(v3)):
                delta = v3 - v2
                mark = "✓" if delta > 0.03 else ("⚠" if delta > -0.03 else "✗")
                print(f"  {metric:25s}: setup2={v2:.3f}  setup3={v3:.3f}  Δ={delta:+.3f}  {mark}")
            v4 = s4.get(metric, np.nan)
            if not (np.isnan(v2) or np.isnan(v4)):
                delta = v4 - v2
                mark = "✓" if delta > 0.03 else ("⚠" if delta > -0.03 else "✗")
                print(f"  {metric:25s}: setup2={v2:.3f}  setup4={v4:.3f}  Δ={delta:+.3f}  {mark}")
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
