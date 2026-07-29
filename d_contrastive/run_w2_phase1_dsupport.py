"""Workstream 2, Phase 1: d-as-support (regime R2 ceiling), frozen, image-only base.

Answers the "把 d 加进去和 image 一起,会更好" question: does giving the downstream
classifier an EXPLICIT d feature (on top of the imaging representation) help?

Three input regimes per task, for diagnosis + conversion:
  image_only   -- logistic on the 768-d features                     [R1 baseline]
  image+dhat   -- logistic on [features, d_hat], d_hat = frozen Ridge [R1, deployable]
  image+dtrue  -- logistic on [features, true d]                      [R2, uses test d]

plus scalar references dhat_only / dtrue_only.

Leakage discipline:
  * scaler fit on TRAIN pool (contrastive+finetune) only;
  * logistic params fit on the FINETUNE split only;
  * `image+dtrue`/`dtrue_only` feed the TEST split's true d as an *input feature at
    test time* (params still fit on train). This is regime R2 -- labelled
    non-deployable (needs Wang's d at test) and never mixed into an R1 headline.
    The d feature is z-scored by FINETUNE stats (no test stats in the transform).

CPU-only (frozen 768-d features).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from minimal_v0_contrastive import load_features_3way, load_features_2way
from run_w0_conversion_3way import cohort, auc_ci, TASKS

DX_BINARIES = [("CN_vs_AD", "NORMAL", "AD"),
               ("CN_vs_MCI", "NORMAL", "MCI"),
               ("MCI_vs_AD", "MCI", "AD")]


def zscore(fit_vals, *apply_vals):
    """z-score by the fit-split stats (no eval stats leak into the transform)."""
    fit_vals = np.asarray(fit_vals, float)
    m = np.nanmean(fit_vals)
    s = np.nanstd(fit_vals)
    s = s if s > 1e-8 else 1.0
    return [np.nan_to_num((np.asarray(v, float) - m) / s) for v in (fit_vals, *apply_vals)]


def probe_auc(Xtr, ytr, Xte, yte):
    if len(Xtr) < 10 or ytr.sum() < 3 or len(np.unique(ytr)) < 2:
        return np.nan, np.nan, np.nan
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    return auc_ci(yte, clf.predict_proba(Xte)[:, 1])


def run_version(version, data, args, audit):
    rows = []
    Xc, Xf, Xt = (data["contrastive"]["features"], data["finetune"]["features"],
                  data["test"]["features"])
    mc, mf, mt = (data["contrastive"]["meta"], data["finetune"]["meta"],
                  data["test"]["meta"])
    lf = data["finetune"].get("long_meta", mf)
    lt = data["test"].get("long_meta", mt)
    dc = mc["d_mod3"].values.astype(float)
    df_ = mf["d_mod3"].values.astype(float)
    dt_true = mt["d_mod3"].values.astype(float)   # R2 only, labelled below

    scaler = StandardScaler().fit(np.vstack([Xc, Xf]))
    Xc_s, Xf_s, Xt_s = scaler.transform(Xc), scaler.transform(Xf), scaler.transform(Xt)

    # frozen d_hat (deployable; no test d in the fit) on both finetune and test
    split_mode = data.get("_mode", "3way")
    if split_mode == "2way":
        fitX, fitd = Xf_s, df_                          # train only (== contrastive)
    else:
        fitX, fitd = np.vstack([Xc_s, Xf_s]), np.concatenate([dc, df_])
    valid = np.isfinite(fitd)
    ridge = Ridge(alpha=args.ridge_alpha).fit(fitX[valid], fitd[valid])
    dhat_f, dhat_t = ridge.predict(Xf_s), ridge.predict(Xt_s)
    audit["image+dhat"] = "R1: d_hat feature (Ridge on contrastive+finetune)"
    audit["image+dtrue"] = "R2: true d as TEST-time input feature (labelled non-deployable)"

    def add(kind, task, n, n_pos, method, triple):
        auc, lo, hi = triple
        rows.append({"version": version, "task_kind": kind, "task": task, "n": n,
                     "n_positive": n_pos, "method": method,
                     "auc": auc, "ci_lo": lo, "ci_hi": hi})

    def eval_block(kind, task, Xf_rep, Xt_rep, ytr, yte, df_fit, dt_hat_eval,
                   df_true, dt_true_eval):
        n, n_pos = len(yte), int(yte.sum())
        # image only
        add(kind, task, n, n_pos, "image_only", probe_auc(Xf_rep, ytr, Xt_rep, yte))
        # image + d_hat  (R1 deployable)
        hf, ht = zscore(df_fit, dt_hat_eval)
        add(kind, task, n, n_pos, "image+dhat",
            probe_auc(np.column_stack([Xf_rep, hf]), ytr,
                      np.column_stack([Xt_rep, ht]), yte))
        add(kind, task, n, n_pos, "dhat_only", auc_ci(yte, dt_hat_eval))
        # image + true d  (R2)
        tf, tt = zscore(df_true, dt_true_eval)
        add(kind, task, n, n_pos, "image+dtrue",
            probe_auc(np.column_stack([Xf_rep, tf]), ytr,
                      np.column_stack([Xt_rep, tt]), yte))
        add(kind, task, n, n_pos, "dtrue_only", auc_ci(yte, dt_true_eval))

    # ---- diagnosis binaries --------------------------------------------------
    for name, neg, pos in DX_BINARIES:
        ftr = mf["dx"].isin([neg, pos]).values
        fte = mt["dx"].isin([neg, pos]).values
        if ftr.sum() < 10 or fte.sum() < 10:
            continue
        ytr = (mf.loc[ftr, "dx"] == pos).astype(int).values
        yte = (mt.loc[fte, "dx"] == pos).astype(int).values
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        print(f"  dx {name} ({version}): test n={int(fte.sum())} pos={int(yte.sum())}")
        eval_block("dx_binary", name, Xf_s[ftr], Xt_s[fte], ytr, yte,
                   df_[ftr], dhat_t[fte], df_[ftr], dt_true[fte])

    # ---- conversion ----------------------------------------------------------
    for task, base, tgt in TASKS:
        for h in args.horizons:
            cf, ct = cohort(lf, "finetune", task, base, tgt, h), cohort(lt, "test", task, base, tgt, h)
            if len(cf) == 0 or len(ct) == 0:
                continue
            tri, ytr = cf["feature_idx"].astype(int).values, cf["converted"].astype(int).values
            tei, yte = ct["feature_idx"].astype(int).values, ct["converted"].astype(int).values
            if len(ytr) < 10 or ytr.sum() < 3 or int(yte.sum()) < 2 or len(np.unique(yte)) < 2:
                continue
            print(f"  conv {task} {h}y ({version}): finetune n={len(ytr)} | test n={len(yte)} conv={int(yte.sum())}")
            eval_block("conversion", f"{task}_{h}y", Xf_s[tri], Xt_s[tei], ytr, yte,
                       df_[tri], dhat_t[tei], df_[tri], dt_true[tei])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--mode", choices=["2way", "3way"], default="3way")
    ap.add_argument("--split_dir", default="../data/splits_3way_20260627_v2")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w2_phase1_dsupport")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows, audit = [], {}
    for v in args.versions:
        print(f"\n===== VERSION: {v} (mode={args.mode}) =====")
        if args.mode == "2way":
            data = load_features_2way(args.features_dir, args.master_dir, args.d_csv, version=v)
        else:
            data = load_features_3way(args.features_dir, args.split_dir, args.d_csv, version=v)
        all_rows += run_version(v, data, args, audit)

    print("\n===== d-source audit (R1 = deployable, R2 = uses test d as input) =====")
    for k, src in audit.items():
        print(f"  {k:12s} <- {src}")

    df = pd.DataFrame(all_rows)
    csv_path = out / "w2_phase1_dsupport.csv"
    df.to_csv(csv_path, index=False)
    with open(out / "w2_phase1_dsupport.md", "w") as f:
        f.write("# Workstream 2, Phase 1: d-as-support\n\n")
        f.write("`image+dhat` (R1, deployable) and `image+dtrue` (R2, feeds true d at "
                "test) vs `image_only`. Compare the lift over `image_only` to judge "
                "how much an explicit d feature adds. R2 rows are NOT deployable.\n\n")
        if len(df):
            f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out / 'w2_phase1_dsupport.md'}")


if __name__ == "__main__":
    main()
