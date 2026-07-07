"""Workstream 0 (3-way split): contrastive conversion prediction + ridge baselines.

Data is partitioned into three RID-DISJOINT splits:
  contrastive : pretrain the ContrastiveMLP encoder on d_mod3 (the MAIN method)
  finetune    : train the downstream conversion probe (logistic on the frozen z)
  test        : evaluation ONLY

HARD CONSTRAINT enforced here: the TEST split's d_mod3 is NEVER used for any
fit or model/hyper-parameter selection. Test contributes only (a) imaging
features and (b) conversion outcome labels for AUC. Every d_mod3 used for
fitting comes from the contrastive and/or finetune splits.

Methods compared per task x horizon x version:
  ridge_dhat_finetune  -- borrow score: Ridge(d_mod3) fit on FINETUNE only
  ridge_dhat_all       -- borrow score: Ridge(d_mod3) fit on CONTRASTIVE+FINETUNE
  direct_logistic      -- internal-only logistic on raw imaging (finetune labels)
  contrastive_<mode>_probe -- logistic probe on the frozen contrastive z (finetune labels)
  contrastive_<mode>_s     -- the 1-D progression score s as a direct conversion score

`delta_*` columns are paired bootstrap AUC differences vs `ridge_dhat_all`.

CPU-only by default (small MLP on frozen 768-d Swin features).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    bootstrap_auc,
    build_censored_conversion_cohort,
    paired_bootstrap_delta,
)
from minimal_v0_contrastive import (
    load_features_3way,
    load_features_2way,
    train_contrastive_encoder,
    encode_features,
    finetune_encoder_classifier,
)

TASKS = [("MCI_to_AD", "MCI", "AD"), ("CN_to_MCI", "NORMAL", "MCI")]


def cohort(meta, split, task, base, tgt, h):
    """Build censored conversion cohort.

    If meta contains `_feature_idx`, it is complete longitudinal metadata and
    `_feature_idx` points back to the image feature row. This is the correct
    path for 3-way conversion evaluation: follow-up labels use all matched
    visits, while baseline features still index into the one-image-per-row
    feature matrix.
    """
    if "_feature_idx" not in meta.columns:
        c = build_censored_conversion_cohort(meta, split, f"{task}_{h}y", base, tgt, h)
        return c[c["valid_censored"].astype(bool)].copy()

    rows = []
    meta = meta.sort_values(["RID", "EXAMDATE.x"]).copy()
    for rid, group in meta.groupby("RID"):
        first = group.iloc[0]
        if first["dx"] != base:
            continue
        baseline_date = first["EXAMDATE.x"]
        horizon_date = baseline_date + pd.DateOffset(years=h)
        follow = group[(group["EXAMDATE.x"] > baseline_date)
                       & (group["EXAMDATE.x"] <= horizon_date)]
        target_follow = follow[follow["dx"] == tgt]
        converted = int(len(target_follow) > 0)
        max_followup_days = np.nan
        later = group[group["EXAMDATE.x"] > baseline_date]
        if len(later):
            max_followup_days = int((later["EXAMDATE.x"].max() - baseline_date).days)
        valid = converted or (
            pd.notna(max_followup_days) and max_followup_days >= h * 365
        )
        rows.append({
            "split": split,
            "task": f"{task}_{h}y",
            "RID": rid,
            "feature_idx": int(first["_feature_idx"]),
            "converted": converted,
            "valid_censored": bool(valid),
            "max_followup_days": max_followup_days,
        })
    c = pd.DataFrame(rows)
    if len(c) == 0:
        return c
    return c[c["valid_censored"].astype(bool)].copy()


def ridge_dhat(X_fit, d_fit, X_test, alpha):
    """Fit Ridge(features -> d_mod3) on valid-d rows of (X_fit,d_fit); predict test."""
    valid = np.isfinite(d_fit)
    rid = Ridge(alpha=alpha).fit(X_fit[valid], d_fit[valid])
    return rid.predict(X_test)


def auc_ci(y, score):
    if len(np.unique(y)) < 2 or len(y) < 5 or not np.all(np.isfinite(score)):
        return np.nan, np.nan, np.nan
    return float(roc_auc_score(y, score)), *bootstrap_auc(y, score)[1:]


def logistic_score(Xtr, ytr, Xte):
    if len(ytr) < 10 or ytr.sum() < 3 or len(np.unique(ytr)) < 2:
        return np.full(len(Xte), np.nan)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def run_version(version, data, args, audit):
    rows = []
    Xc = data["contrastive"]["features"]
    Xf = data["finetune"]["features"]
    Xt = data["test"]["features"]
    mc, mf, mt = (data["contrastive"]["meta"], data["finetune"]["meta"],
                  data["test"]["meta"])
    lf = data["finetune"].get("long_meta", mf)
    lt = data["test"].get("long_meta", mt)
    dc = mc["d_mod3"].values.astype(float)
    df_ = mf["d_mod3"].values.astype(float)
    # NOTE: mt["d_mod3"] is deliberately NEVER read below.

    # Standardize on the TRAIN pool only (contrastive+finetune features). Test
    # features are transformed with train stats -- no test labels/d involved.
    scaler = StandardScaler().fit(np.vstack([Xc, Xf]))
    Xc_s, Xf_s, Xt_s = scaler.transform(Xc), scaler.transform(Xf), scaler.transform(Xt)

    # ---- borrow-score baselines: Ridge d_hat -------------------------------
    split_mode = data.get("_mode", "3way")
    dhat_ft = ridge_dhat(Xf_s, df_, Xt_s, args.ridge_alpha)             # finetune (2way: train)
    if split_mode == "2way":
        # encoder-pretrain and head pools are the same `train` split, so the two
        # ridge configs collapse to one fit on train.
        dhat_all = dhat_ft
        audit["ridge_dhat_finetune"] = "fit d on train split (2way)"
        audit["ridge_dhat_all"] = "fit d on train split (2way; == finetune)"
    else:
        X_all = np.vstack([Xc_s, Xf_s])
        d_all = np.concatenate([dc, df_])
        dhat_all = ridge_dhat(X_all, d_all, Xt_s, args.ridge_alpha)     # contrastive+finetune
        audit["ridge_dhat_finetune"] = "fit d on finetune split"
        audit["ridge_dhat_all"] = "fit d on contrastive+finetune splits"

    # ---- contrastive encoders (MAIN method), one per loss mode ---------------
    enc = {}
    validc = np.isfinite(dc)
    for mode in args.loss_modes:
        print(f"  [{version}] training contrastive encoder: loss_mode={mode}")
        model, _ = train_contrastive_encoder(
            Xc_s[validc], dc[validc], device=args.device, tau=args.tau,
            beta=args.beta, epochs=args.epochs, batch_size=args.batch_size,
            hidden=args.hidden, latent=args.latent, loss_mode=mode,
            rank_alpha=args.rank_alpha, lambda_rank=args.lambda_rank,
            rank_nu=args.rank_nu, seed=args.seed, verbose=True)
        zf, _ = encode_features(model, Xf_s, args.device)
        zt, st = encode_features(model, Xt_s, args.device)
        enc[mode] = (model, zf, zt, st)
        audit[f"contrastive_{mode}"] = (
            "fit d on train split (2way, encoder)" if split_mode == "2way"
            else "fit d on contrastive split (encoder)")

    # ---- per task x horizon evaluation on the TEST cohort --------------------
    for task, base, tgt in TASKS:
        for h in args.horizons:
            cf = cohort(lf, "finetune", task, base, tgt, h)
            ct = cohort(lt, "test", task, base, tgt, h)
            tri, ytr = cf["feature_idx"].astype(int).values, cf["converted"].astype(int).values
            tei, yte = ct["feature_idx"].astype(int).values, ct["converted"].astype(int).values
            n_te, n_conv = len(yte), int(yte.sum())
            if len(ytr) < 10 or ytr.sum() < 3 or n_conv < 2 or len(np.unique(yte)) < 2:
                print(f"  [skip] {task} {h}y ({version}): finetune_conv={int(ytr.sum())} "
                      f"test_conv={n_conv}")
                continue
            print(f"  {task} {h}y ({version}): finetune n={len(ytr)} conv={int(ytr.sum())} "
                  f"| test n={n_te} conv={n_conv}")

            scores = {
                "ridge_dhat_finetune": dhat_ft[tei],
                "ridge_dhat_all": dhat_all[tei],
                "direct_logistic": logistic_score(Xf_s[tri], ytr, Xt_s[tei]),
            }
            for mode, (model, zf, zt, st) in enc.items():
                # frozen-encoder linear probe
                scores[f"contrastive_{mode}_probe"] = logistic_score(zf[tri], ytr, zt[tei])
                # direct contrastive progression score (no downstream training)
                scores[f"contrastive_{mode}_s"] = st[tei]
                # end-to-end fine-tune of the encoder on the conversion labels
                if not args.no_finetune:
                    scores[f"contrastive_{mode}_finetune"] = finetune_encoder_classifier(
                        model, Xf_s[tri], ytr, Xt_s[tei], device=args.device,
                        latent=args.latent, epochs=args.finetune_epochs, seed=args.seed)

            base_score = scores["ridge_dhat_all"]
            for method, sc in scores.items():
                auc, lo, hi = auc_ci(yte, sc)
                d = dlo = dhi = dp = np.nan
                if (method != "ridge_dhat_all" and np.all(np.isfinite(sc))
                        and np.all(np.isfinite(base_score))):
                    d, dlo, dhi, dp = paired_bootstrap_delta(yte, base_score, sc)
                rows.append({
                    "version": version, "task": task, "horizon_years": h,
                    "method": method, "n": n_te, "n_converters": n_conv,
                    "auc": auc, "ci_lo": lo, "ci_hi": hi,
                    "delta_vs_ridge_all": d, "delta_ci_lo": dlo,
                    "delta_ci_hi": dhi, "delta_p": dp,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--mode", choices=["2way", "3way"], default="3way",
                    help="2way: encoder+head both on train, test held out")
    ap.add_argument("--split_dir", default="../data/splits_3way_20260627_v2")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016",
                    help="2way: dir with matched_TRAIN.csv / matched_TEST.csv")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w0_conv_3way")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--loss_modes", nargs="+",
                    default=["euclidean", "rank_kendall_basic", "hybrid_basic"],
                    help="contrastive geometries to evaluate")
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
    ap.add_argument("--finetune_epochs", type=int, default=60,
                    help="epochs for the end-to-end encoder fine-tune variant")
    ap.add_argument("--no_finetune", action="store_true",
                    help="skip the encoder fine-tune variant (frozen probe only)")
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

    # Audit: prove TEST d_mod3 was never a fit input.
    print("\n===== d_mod3 fit-source audit =====")
    for k, src in audit.items():
        print(f"  {k:28s} <- {src}")
    assert not any("test" in s.lower() for s in audit.values()), \
        "LEAKAGE: a method used the test split's d_mod3"
    print("  TEST split d_mod3 used in any fit/selection: NO  (held out)")

    df = pd.DataFrame(all_rows)
    csv_path = out / "w0_conversion_3way.csv"
    df.to_csv(csv_path, index=False)
    with open(out / "w0_conversion_3way.md", "w") as f:
        f.write("# Workstream 0 (3-way split): conversion results\n\n")
        f.write("Splits are RID-disjoint: **contrastive** pretrains the encoder, "
                "**finetune** trains the conversion probe, **test** is eval-only. "
                "The test split's `d_mod3` is never used for any fit.\n\n")
        f.write("Two ridge baselines: `ridge_dhat_finetune` (Ridge on d_mod3, "
                "finetune split only) vs `ridge_dhat_all` (contrastive+finetune). "
                "For each contrastive geometry both a frozen-encoder linear probe "
                "(`_probe`) and an end-to-end encoder fine-tune (`_finetune`) are "
                "reported. `delta_*` = paired bootstrap AUC vs `ridge_dhat_all`.\n\n")
        if len(df):
            f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out / 'w0_conversion_3way.md'}")


if __name__ == "__main__":
    main()
