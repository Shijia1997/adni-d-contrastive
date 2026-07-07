"""Workstream 2, Phase 0: how much diagnosis / conversion signal does d carry?

Pure characterization -- NO model training. Establishes the CEILING that every
later "internalize d into the encoder" method (Phase 1/2) is chasing.

Two references per task, on the TEST split:
  oracle_true_d  -- AUC using the TEST split's TRUE d_mod3 as the score.
                    ORACLE ONLY: this reads test d, so it is *not* a deployable
                    method -- it is the analysis upper bound on d's information.
                    It is labelled as oracle everywhere and never mixed into any
                    headline method comparison.
  ridge_dhat_all -- AUC of frozen d_hat = Ridge(features -> d) fit on
                    CONTRASTIVE+FINETUNE, predicted on test. This is the
                    DEPLOYABLE ceiling: no test d enters any fit.

Tasks:
  diagnosis binaries : CN_vs_AD, CN_vs_MCI, MCI_vs_AD (per-visit dx, score = d)
  conversion         : MCI_to_AD, CN_to_MCI at horizons 2/3/4y (baseline visit)

CPU-only; operates on precomputed frozen 768-d embeddings.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from minimal_v0_contrastive import load_features_3way, load_features_2way
# Reuse the exact conversion-cohort + ridge + AUC helpers so Phase 0 and the
# later phases score identical cohorts.
from run_w0_conversion_3way import cohort, ridge_dhat, auc_ci, TASKS

DX_BINARIES = [("CN_vs_AD", "NORMAL", "AD"),
               ("CN_vs_MCI", "NORMAL", "MCI"),
               ("MCI_vs_AD", "MCI", "AD")]
DX_ORDINAL = {"NORMAL": 0, "MCI": 1, "AD": 2}


def run_version(version, data, args, audit):
    rows = []
    Xc = data["contrastive"]["features"]
    Xf = data["finetune"]["features"]
    Xt = data["test"]["features"]
    mc, mf, mt = (data["contrastive"]["meta"], data["finetune"]["meta"],
                  data["test"]["meta"])
    lt = data["test"].get("long_meta", mt)
    dc = mc["d_mod3"].values.astype(float)
    df_ = mf["d_mod3"].values.astype(float)
    # true test d -- ONLY used for the clearly-labelled oracle reference.
    d_test_true = mt["d_mod3"].values.astype(float)

    # Standardize on the TRAIN pool only (no test features/labels in the scaler).
    scaler = StandardScaler().fit(np.vstack([Xc, Xf]))
    Xt_s = scaler.transform(Xt)
    split_mode = data.get("_mode", "3way")
    if split_mode == "2way":
        dhat_all = ridge_dhat(scaler.transform(Xf), df_, Xt_s, args.ridge_alpha)
        audit["ridge_dhat_all"] = "fit d on train split (2way)"
    else:
        X_all = np.vstack([scaler.transform(Xc), scaler.transform(Xf)])
        d_all = np.concatenate([dc, df_])
        dhat_all = ridge_dhat(X_all, d_all, Xt_s, args.ridge_alpha)   # deployable ceiling
        audit["ridge_dhat_all"] = "fit d on contrastive+finetune splits"
    audit["oracle_true_d"] = "ORACLE: reads test d_mod3 (analysis upper bound only)"

    def emit(task_kind, task, n, n_pos, y, sc_true, sc_hat):
        for method, sc in (("oracle_true_d", sc_true), ("ridge_dhat_all", sc_hat)):
            auc, lo, hi = auc_ci(y, sc)
            rows.append({"version": version, "task_kind": task_kind, "task": task,
                         "n": n, "n_positive": n_pos, "method": method,
                         "auc": auc, "ci_lo": lo, "ci_hi": hi})

    # ---- diagnosis binaries (per-visit dx; test meta aligned 1:1 with features)
    for name, neg, pos in DX_BINARIES:
        m = mt["dx"].isin([neg, pos]).values
        if m.sum() < 10:
            print(f"  [skip dx] {name} ({version}): n={int(m.sum())}")
            continue
        y = (mt.loc[m, "dx"] == pos).astype(int).values
        if len(np.unique(y)) < 2:
            continue
        emit("diagnosis", name, int(m.sum()), int(y.sum()),
             y, d_test_true[m], dhat_all[m])
        print(f"  dx {name} ({version}): n={int(m.sum())} pos={int(y.sum())}")

    # ---- ordinal summary: monotonicity of d along CN<MCI<AD (test, per visit)
    ord_mask = mt["dx"].isin(DX_ORDINAL).values
    if ord_mask.sum() > 10:
        yo = mt.loc[ord_mask, "dx"].map(DX_ORDINAL).values.astype(float)
        rho_true = spearmanr(d_test_true[ord_mask], yo).correlation
        rho_hat = spearmanr(dhat_all[ord_mask], yo).correlation
        rows.append({"version": version, "task_kind": "dx_ordinal",
                     "task": "CN<MCI<AD_spearman", "n": int(ord_mask.sum()),
                     "n_positive": np.nan, "method": "oracle_true_d",
                     "auc": rho_true, "ci_lo": np.nan, "ci_hi": np.nan})
        rows.append({"version": version, "task_kind": "dx_ordinal",
                     "task": "CN<MCI<AD_spearman", "n": int(ord_mask.sum()),
                     "n_positive": np.nan, "method": "ridge_dhat_all",
                     "auc": rho_hat, "ci_lo": np.nan, "ci_hi": np.nan})

    # ---- conversion (baseline visit; same censored cohorts as the main driver)
    for task, base, tgt in TASKS:
        for h in args.horizons:
            ct = cohort(lt, "test", task, base, tgt, h)
            if len(ct) == 0:
                continue
            tei = ct["feature_idx"].astype(int).values
            yte = ct["converted"].astype(int).values
            n_conv = int(yte.sum())
            if n_conv < 2 or len(np.unique(yte)) < 2:
                print(f"  [skip conv] {task} {h}y ({version}): conv={n_conv}")
                continue
            emit("conversion", f"{task}_{h}y", len(yte), n_conv,
                 yte, d_test_true[tei], dhat_all[tei])
            print(f"  conv {task} {h}y ({version}): n={len(yte)} conv={n_conv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="../data/embeddings_128_05152016")
    ap.add_argument("--mode", choices=["2way", "3way"], default="3way")
    ap.add_argument("--split_dir", default="../data/splits_3way_20260627_v2")
    ap.add_argument("--master_dir", default="../data/master_smri_05152016")
    ap.add_argument("--d_csv",
                    default="../data/master_smri_05152016/D_with_image_paths_full.csv")
    ap.add_argument("--out_dir", default="results_w2_phase0")
    ap.add_argument("--versions", nargs="+", default=["raw", "combat"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
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

    # The only method that touches test d is the clearly-labelled oracle.
    print("\n===== d-source audit =====")
    for k, src in audit.items():
        print(f"  {k:16s} <- {src}")
    assert audit.get("ridge_dhat_all") and "test" not in audit["ridge_dhat_all"].lower()
    print("  deployable ceiling (ridge_dhat_all) uses TEST d: NO")
    print("  oracle_true_d uses TEST d: YES (analysis upper bound, not a method)")

    df = pd.DataFrame(all_rows)
    csv_path = out / "w2_phase0_dvalue.csv"
    df.to_csv(csv_path, index=False)
    with open(out / "w2_phase0_dvalue.md", "w") as f:
        f.write("# Workstream 2, Phase 0: d-value ceiling\n\n")
        f.write("`oracle_true_d` uses the TEST split's true d_mod3 (upper bound, "
                "NOT deployable). `ridge_dhat_all` = frozen d_hat, no test d in any "
                "fit (deployable ceiling). For `dx_ordinal` the `auc` column is a "
                "Spearman rho, not an AUC.\n\n")
        if len(df):
            f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out / 'w2_phase0_dvalue.md'}")


if __name__ == "__main__":
    main()
