"""Workstream 1: RASPER for AD conversion prediction.

Borrows an EXTERNAL RANKING (option A: Ridge d_hat trained to predict d_mod3 on
the large train split) to regularize a small internal conversion model, and
compares borrowing-ranking (RASPER) against borrowing-score (ridge_d_hat) and
against plain internal logistic.

Mapping to the paper (Henderson 2026):
  internal study Y      = conversion label (MCI->AD / CN->MCI within horizon)
  internal covariates x = low-dim imaging block (default: PCA of frozen Swin 768;
                          can be swapped for the W0 contrastive latent z)
  external ranker f_E   = Ridge predicting d_mod3 on the full train split
  external ranks r_ext  = ranks of d_hat on the internal cohort

Outputs (in --out_dir):
  w1_rasper_conversion.csv  -- one row per method x task x horizon x version
  w1_rasper_conversion.md   -- readable summary table

CPU-only, pure numpy/scipy/sklearn. No torch, no GPU.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    bootstrap_auc,
    build_censored_conversion_cohort,
    paired_bootstrap_delta,
)
from minimal_v0_contrastive import load_features_and_labels
from rasper import RASPER, select_lambda_alpha_cv

TASKS = [("MCI_to_AD", "MCI", "AD"), ("CN_to_MCI", "NORMAL", "MCI")]


def ranks(v):
    """Ascending ranks 1..n (ties broken by order); larger value -> larger rank."""
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float) + 1.0


def build_internal_features(train_feat, test_feat, n_pca):
    """Standardize the 768-d Swin features and PCA-reduce to n_pca components.

    Returns standardized + reduced (Xtr, Xte). Swap this out to load the W0
    contrastive latent z once W0 has been run.
    """
    sc = StandardScaler().fit(train_feat)
    Xtr, Xte = sc.transform(train_feat), sc.transform(test_feat)
    if n_pca and n_pca < Xtr.shape[1]:
        pca = PCA(n_components=n_pca, random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        sc2 = StandardScaler().fit(Xtr)        # rescale PCs for stable optimization
        Xtr, Xte = sc2.transform(Xtr), sc2.transform(Xte)
    return Xtr, Xte


def external_dhat(train_feat, test_feat, d_train, ridge_alpha=10.0):
    """Option-A external ranker: Ridge predicting d_mod3 on full train split."""
    sc = StandardScaler().fit(train_feat)
    valid = np.isfinite(d_train)
    rid = Ridge(alpha=ridge_alpha).fit(sc.transform(train_feat)[valid], d_train[valid])
    return rid.predict(sc.transform(train_feat)), rid.predict(sc.transform(test_feat))


def auc_ci(y, score, quick_no_ci=False):
    if len(np.unique(y)) < 2 or len(y) < 5:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, score)
    if quick_no_ci:
        return float(auc), np.nan, np.nan
    _, lo, hi = bootstrap_auc(y, score)
    return float(auc), lo, hi


def run_version(version, data, args):
    rows = []
    tr_feat, te_feat = data["train_features"], data["test_features"]
    d_tr = data["train_meta"]["d_mod3"].values.astype(float)

    Xtr_int, Xte_int = build_internal_features(tr_feat, te_feat, args.n_pca)
    dhat_tr, dhat_te = external_dhat(tr_feat, te_feat, d_tr, args.ridge_alpha)

    for task, base_dx, tgt_dx in TASKS:
        for h in args.horizons:
            name = f"{task}_{h}y"
            ctr = build_censored_conversion_cohort(
                data["train_meta"], "train", name, base_dx, tgt_dx, horizon_years=h)
            cte = build_censored_conversion_cohort(
                data["test_meta"], "test", name, base_dx, tgt_dx, horizon_years=h)
            ctr = ctr[ctr["valid_censored"].astype(bool)]
            cte = cte[cte["valid_censored"].astype(bool)]
            tri = ctr["feature_idx"].astype(int).values
            tei = cte["feature_idx"].astype(int).values
            ytr = ctr["converted"].astype(int).values
            yte = cte["converted"].astype(int).values

            n_te, n_conv_te = len(yte), int(yte.sum())
            if (len(ytr) < 10 or ytr.sum() < 3 or n_conv_te < 2
                    or len(np.unique(yte)) < 2):
                print(f"  [skip] {name} ({version}) too few: "
                      f"train_conv={int(ytr.sum())} test_conv={n_conv_te}")
                continue
            print(f"  {name} ({version}): train n={len(ytr)} conv={int(ytr.sum())} "
                  f"| test n={n_te} conv={n_conv_te}")

            r_ext_tr = ranks(dhat_tr[tri])      # external ranking on the internal cohort

            scores = {}

            # borrow score: external d_hat applied directly
            scores["ridge_d_hat"] = dhat_te[tei]

            # internal-only logistic on the imaging block
            try:
                lr = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr_int[tri], ytr)
                scores["direct_logistic"] = lr.predict_proba(Xte_int[tei])[:, 1]
            except Exception as e:
                scores["direct_logistic"] = np.full(n_te, np.nan)
                print(f"    direct_logistic failed: {e}")

            # oracle: true baseline d_mod3
            scores["oracle_true_d_mod3"] = cte["d_mod3"].values.astype(float)

            # RASPER: borrow ranking
            for pen in args.penalties:
                try:
                    if args.fixed_rasper:
                        lam = float(args.lambdas[0])
                        al = float(args.alphas[0])
                        est = RASPER(
                            family="binomial", penalty=pen, lam=lam, alpha=al,
                            nu=args.rank_nu
                        ).fit(Xtr_int[tri], ytr, r_ext_tr)
                    else:
                        lam, al, est = select_lambda_alpha_cv(
                            Xtr_int[tri], ytr, r_ext_tr, family="binomial", penalty=pen,
                            lambdas=np.asarray(args.lambdas, dtype=float),
                            alphas=np.asarray(args.alphas, dtype=float),
                            nu=args.rank_nu,
                            n_splits=args.cv_splits, seed=0)
                    scores[f"rasper_{pen}"] = est.decision_function(Xte_int[tei])
                    print(f"    rasper_{pen}: lambda={lam:.3g} alpha={al:.3g} "
                          f"nu={est.nu_:.3g}")
                except Exception as e:
                    scores[f"rasper_{pen}"] = np.full(n_te, np.nan)
                    print(f"    rasper_{pen} failed: {e}")

            base = scores["ridge_d_hat"]
            for method, sc in scores.items():
                if not np.all(np.isfinite(sc)):
                    auc = lo = hi = np.nan
                else:
                    auc, lo, hi = auc_ci(yte, sc, quick_no_ci=args.quick_no_ci)
                d, dlo, dhi, dp = (np.nan,) * 4
                if method != "ridge_d_hat" and np.all(np.isfinite(sc)) and np.all(np.isfinite(base)):
                    if args.quick_no_ci:
                        d = roc_auc_score(yte, sc) - roc_auc_score(yte, base)
                    else:
                        d, dlo, dhi, dp = paired_bootstrap_delta(yte, base, sc)
                rows.append({
                    "version": version, "task": task, "horizon_years": h,
                    "method": method, "n": n_te, "n_converters": n_conv_te,
                    "auc": auc, "ci_lo": lo, "ci_hi": hi,
                    "delta_vs_ridge_dhat": d, "delta_ci_lo": dlo,
                    "delta_ci_hi": dhi, "delta_p": dp,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w1_rasper")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--n_pca", type=int, default=16,
                    help="PCA dim of the internal imaging block (0 = use all 768).")
    ap.add_argument("--ridge_alpha", type=float, default=10.0,
                    help="L2 for the external d_hat ranker.")
    ap.add_argument("--penalties", nargs="+", default=["kendall"],
                    choices=["kendall", "spearman"])
    ap.add_argument("--cv_splits", type=int, default=5,
                    help="CV folds for (lambda,alpha) selection; <=0 => LOO.")
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 1.0, 3.1622776601683795, 10.0],
                    help="Lambda grid for RASPER CV.")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1.0, 10.0],
                    help="L2 alpha grid for RASPER CV.")
    ap.add_argument("--fixed_rasper", action="store_true",
                    help="Skip CV and fit RASPER with lambdas[0], alphas[0].")
    ap.add_argument("--rank_nu", type=float, default=None,
                    help="RASPER smoothing nu. None => 0.1 * ||beta_MLE||.")
    ap.add_argument("--quick_no_ci", action="store_true",
                    help="Skip bootstrap CIs/p-values for fast point estimates.")
    args = ap.parse_args()
    if args.cv_splits <= 0:
        args.cv_splits = None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for v in args.versions:
        print(f"\n===== VERSION: {v} =====")
        data = load_features_and_labels(args.features_dir, args.d_csv, version=v)
        all_rows += run_version(v, data, args)

    df = pd.DataFrame(all_rows)
    csv_path = out / "w1_rasper_conversion.csv"
    df.to_csv(csv_path, index=False)
    with open(out / "w1_rasper_conversion.md", "w") as f:
        f.write("# Workstream 1: RASPER conversion results\n\n")
        f.write("Headline contrast: `rasper_*` (borrow ranking) vs `ridge_d_hat` "
                "(borrow score). `delta_*` columns are paired bootstrap AUC "
                "differences vs `ridge_d_hat`.\n\n")
        if len(df):
            f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out / 'w1_rasper_conversion.md'}")


if __name__ == "__main__":
    main()
