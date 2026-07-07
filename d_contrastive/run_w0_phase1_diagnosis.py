"""Workstream 2, Phase 1 (frozen encoder): DIAGNOSIS task, image-only.

Companion to run_w0_conversion_3way.py (which already covers the conversion task).
Same 3-way RID-disjoint split, same frozen 768-d Swin features. Compares the
three supervision arms for learning the encoder, then a downstream diagnosis
classifier on the FINETUNE split, evaluated on TEST:

  arm (a) baseline        : logistic directly on raw 768-d features (no d)
  arm (b) contrastive     : encoder pretrained with a pairwise d-geometry
                            (euclidean / rank_kendall_basic / hybrid_basic)
  arm (c) regress_d       : encoder pretrained with direct MSE(s, d)   [NEW]

For (b)/(c) we report both a logistic probe on the frozen 128-d latent z
(`contrastive_<mode>_probe`) and, for binaries, the 1-D score s as a direct
ordinal score (`contrastive_<mode>_s`). Scalar borrow-score references
`ridge_dhat_finetune` / `ridge_dhat_all` ride along on the binaries.

HARD CONSTRAINT (same as the conversion driver): the TEST split's d_mod3 is NEVER
used for any fit or selection. Test contributes only imaging features + dx labels.
Encoder d comes from the contrastive split; probe labels (dx) from finetune.

CPU-only (small MLP on frozen features).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from minimal_v0_contrastive import (
    load_features_3way,
    load_features_2way,
    train_contrastive_encoder,
    encode_features,
)
from run_w0_conversion_3way import ridge_dhat, auc_ci

DX3 = ["NORMAL", "MCI", "AD"]
DX_BINARIES = [("CN_vs_AD", "NORMAL", "AD"),
               ("CN_vs_MCI", "NORMAL", "MCI"),
               ("MCI_vs_AD", "MCI", "AD")]


def binary_probe_auc(Xtr, ytr, Xte, yte):
    """Logistic probe on the given representation; AUC + bootstrap CI on test."""
    if len(Xtr) < 10 or ytr.sum() < 3 or len(np.unique(ytr)) < 2:
        return np.nan, np.nan, np.nan
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return auc_ci(yte, p)


def macro_auc_3class(Xtr, ytr, Xte, yte):
    """OvR macro-AUC for the 3-class CN/MCI/AD task (point estimate)."""
    if len(np.unique(ytr)) < 3 or len(np.unique(yte)) < 3:
        return np.nan
    clf = LogisticRegression(max_iter=5000, C=1.0,
                             multi_class="multinomial").fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    try:
        return float(roc_auc_score(yte, proba, multi_class="ovr",
                                   average="macro", labels=clf.classes_))
    except ValueError:
        return np.nan


def run_version(version, data, args, audit):
    rows = []
    Xc, Xf, Xt = (data["contrastive"]["features"], data["finetune"]["features"],
                  data["test"]["features"])
    mc, mf, mt = (data["contrastive"]["meta"], data["finetune"]["meta"],
                  data["test"]["meta"])
    dc = mc["d_mod3"].values.astype(float)
    df_ = mf["d_mod3"].values.astype(float)
    # NOTE: mt["d_mod3"] is deliberately NEVER read.

    scaler = StandardScaler().fit(np.vstack([Xc, Xf]))
    Xc_s, Xf_s, Xt_s = scaler.transform(Xc), scaler.transform(Xf), scaler.transform(Xt)

    # scalar borrow-score references
    split_mode = data.get("_mode", "3way")
    dhat_ft = ridge_dhat(Xf_s, df_, Xt_s, args.ridge_alpha)
    if split_mode == "2way":
        dhat_all = dhat_ft                       # encoder+head pools == train; ridge collapses
        audit["ridge_dhat_finetune"] = "fit d on train split (2way)"
        audit["ridge_dhat_all"] = "fit d on train split (2way; == finetune)"
    else:
        X_all = np.vstack([Xc_s, Xf_s])
        d_all = np.concatenate([dc, df_])
        dhat_all = ridge_dhat(X_all, d_all, Xt_s, args.ridge_alpha)
        audit["ridge_dhat_finetune"] = "fit d on finetune split"
        audit["ridge_dhat_all"] = "fit d on contrastive+finetune splits"

    # supervision arms (b) + (c): one encoder per loss mode, pretrained on contrastive d
    enc = {}
    validc = np.isfinite(dc)
    for mode in args.loss_modes:
        print(f"  [{version}] training encoder: loss_mode={mode}")
        model, _ = train_contrastive_encoder(
            Xc_s[validc], dc[validc], device=args.device, tau=args.tau,
            beta=args.beta, epochs=args.epochs, batch_size=args.batch_size,
            hidden=args.hidden, latent=args.latent, loss_mode=mode,
            rank_alpha=args.rank_alpha, lambda_rank=args.lambda_rank,
            rank_nu=args.rank_nu, seed=args.seed, verbose=True)
        zf, _ = encode_features(model, Xf_s, args.device)
        zt, st = encode_features(model, Xt_s, args.device)
        enc[mode] = (zf, zt, st)
        audit[f"contrastive_{mode}"] = (
            "fit d on train split (2way, encoder)" if split_mode == "2way"
            else "fit d on contrastive split (encoder)")

    def add(task_kind, task, n, n_pos, method, auc, lo=np.nan, hi=np.nan):
        rows.append({"version": version, "task_kind": task_kind, "task": task,
                     "n": n, "n_positive": n_pos, "method": method,
                     "auc": auc, "ci_lo": lo, "ci_hi": hi})

    # ---- 3-class CN/MCI/AD (macro OvR AUC) -----------------------------------
    ftr = mf["dx"].isin(DX3).values
    fte = mt["dx"].isin(DX3).values
    ytr3, yte3 = mf.loc[ftr, "dx"].values, mt.loc[fte, "dx"].values
    n3 = int(fte.sum())
    add("dx3", "CN_MCI_AD", n3, np.nan, "baseline_raw",
        macro_auc_3class(Xf_s[ftr], ytr3, Xt_s[fte], yte3))
    for mode, (zf, zt, st) in enc.items():
        add("dx3", "CN_MCI_AD", n3, np.nan, f"contrastive_{mode}_probe",
            macro_auc_3class(zf[ftr], ytr3, zt[fte], yte3))
    print(f"  dx3 CN/MCI/AD ({version}): test n={n3}")

    # ---- binaries ------------------------------------------------------------
    for name, neg, pos in DX_BINARIES:
        ftr = mf["dx"].isin([neg, pos]).values
        fte = mt["dx"].isin([neg, pos]).values
        if ftr.sum() < 10 or fte.sum() < 10:
            print(f"  [skip] {name} ({version}): finetune={int(ftr.sum())} test={int(fte.sum())}")
            continue
        ytr = (mf.loc[ftr, "dx"] == pos).astype(int).values
        yte = (mt.loc[fte, "dx"] == pos).astype(int).values
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        n, n_pos = int(fte.sum()), int(yte.sum())
        print(f"  {name} ({version}): finetune n={int(ftr.sum())} pos={int(ytr.sum())} "
              f"| test n={n} pos={n_pos}")

        add("dx_binary", name, n, n_pos, "baseline_raw",
            *binary_probe_auc(Xf_s[ftr], ytr, Xt_s[fte], yte))
        add("dx_binary", name, n, n_pos, "ridge_dhat_finetune", *auc_ci(yte, dhat_ft[fte]))
        add("dx_binary", name, n, n_pos, "ridge_dhat_all", *auc_ci(yte, dhat_all[fte]))
        for mode, (zf, zt, st) in enc.items():
            add("dx_binary", name, n, n_pos, f"contrastive_{mode}_probe",
                *binary_probe_auc(zf[ftr], ytr, zt[fte], yte))
            add("dx_binary", name, n, n_pos, f"contrastive_{mode}_s",
                *auc_ci(yte, st[fte]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--mode", choices=["2way", "3way"], default="3way")
    ap.add_argument("--split_dir", default="../data/splits_3way_20260627_v2")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w2_phase1_dx")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--loss_modes", nargs="+",
                    default=["euclidean", "rank_kendall_basic", "hybrid_basic", "regress_d"],
                    help="supervision arms (b: geometries, c: regress_d)")
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rank_alpha", type=float, default=10.0)
    ap.add_argument("--rank_nu", type=float, default=None)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, audit = [], {}
    for v in args.versions:
        print(f"\n===== VERSION: {v} (mode={args.mode}) =====")
        if args.mode == "2way":
            data = load_features_2way(args.features_dir, args.master_dir, args.d_csv, version=v)
        else:
            data = load_features_3way(args.features_dir, args.split_dir, args.d_csv, version=v)
        all_rows += run_version(v, data, args, audit)

    print("\n===== d_mod3 fit-source audit =====")
    for k, src in audit.items():
        print(f"  {k:28s} <- {src}")
    assert not any("test" in s.lower() for s in audit.values()), \
        "LEAKAGE: a method used the test split's d_mod3"
    print("  TEST split d_mod3 used in any fit/selection: NO  (held out)")

    df = pd.DataFrame(all_rows)
    csv_path = out / "w2_phase1_diagnosis.csv"
    df.to_csv(csv_path, index=False)
    with open(out / "w2_phase1_diagnosis.md", "w") as f:
        f.write("# Workstream 2, Phase 1 (frozen): diagnosis\n\n")
        f.write("Supervision arms: `baseline_raw` (no d), contrastive geometries "
                "(euclidean/rank_kendall_basic/hybrid_basic), and `regress_d` "
                "(direct MSE(s,d)). `_probe` = logistic on frozen z; `_s` = 1-D "
                "score s as ordinal score. `dx3` AUC is OvR macro (point estimate).\n\n")
        if len(df):
            f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out / 'w2_phase1_diagnosis.md'}")


if __name__ == "__main__":
    main()
