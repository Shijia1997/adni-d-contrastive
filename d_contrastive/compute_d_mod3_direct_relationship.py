from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path("/dcs07/zwang/data/adni_d")
OUT_DIR = ROOT / "d_contrastive/results_w0_conv_3way"


def boot_auc(y, s, n=1000, seed=42):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    vals = np.asarray(vals)
    return float(auc), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_baseline(df):
    df = df.copy()
    df["EXAMDATE.x"] = pd.to_datetime(df["EXAMDATE.x"])
    visit_med = df.groupby(["RID", "EXAMDATE.x"])["d_mod3"].median().reset_index()
    base = visit_med.sort_values("EXAMDATE.x").groupby("RID").first().reset_index()
    b = (
        df.merge(base[["RID", "EXAMDATE.x"]], on=["RID", "EXAMDATE.x"])
        .sort_values(["RID", "EXAMDATE.x"])
        .drop_duplicates("RID")
    )
    ptid_col = "PTID" if "PTID" in b.columns else "Subject"
    image_col = "Image Data ID" if "Image Data ID" in b.columns else "Image_Data_ID"
    b = b[["RID", ptid_col, image_col, "EXAMDATE.x", "dx"]].copy()
    b = b.rename(columns={ptid_col: "PTID", image_col: "Image Data ID"})
    b = b.merge(base[["RID", "d_mod3"]], on="RID", how="left")
    return b.dropna(subset=["d_mod3", "dx"])


def conversion_cohort(df, base_dx, target_dx, h):
    df = df.copy()
    df["EXAMDATE.x"] = pd.to_datetime(df["EXAMDATE.x"])
    rows = []
    for rid, g in df.sort_values(["RID", "EXAMDATE.x"]).groupby("RID"):
        first_date = g["EXAMDATE.x"].min()
        first_rows = g[g["EXAMDATE.x"] == first_date]
        first = first_rows.iloc[0]
        if first["dx"] != base_dx:
            continue
        d0 = first_rows["d_mod3"].median()
        if pd.isna(d0):
            continue
        horizon = first_date + pd.DateOffset(years=h)
        follow = g[(g["EXAMDATE.x"] > first_date) & (g["EXAMDATE.x"] <= horizon)]
        converted = int((follow["dx"] == target_dx).any())
        later = g[g["EXAMDATE.x"] > first_date]
        max_follow = np.nan if len(later) == 0 else int((later["EXAMDATE.x"].max() - first_date).days)
        valid = bool(converted or (pd.notna(max_follow) and max_follow >= h * 365))
        if valid:
            rows.append(
                {
                    "RID": rid,
                    "baseline_dx": base_dx,
                    "target_dx": target_dx,
                    "horizon_years": h,
                    "converted": converted,
                    "d_mod3": float(d0),
                    "max_followup_days": max_follow,
                }
            )
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    strict_test = pd.read_csv(ROOT / "data/splits_3way_20260627_v2/matched_TEST.csv")
    strict_bl = build_baseline(strict_test)

    print("\n=== STRICT 3-WAY TEST: baseline d_mod3 by dx ===")
    print(strict_bl.groupby("dx")["d_mod3"].describe()[["count", "mean", "std", "min", "50%", "max"]].to_string())

    dx_map = {"NORMAL": 0, "MCI": 1, "AD": 2}
    mask = strict_bl["dx"].isin(dx_map)
    rho, p = spearmanr(strict_bl.loc[mask, "dx"].map(dx_map), strict_bl.loc[mask, "d_mod3"])
    print(f"Ordinal dx Spearman with d_mod3: rho={rho:.3f}, p={p:.2e}, n={mask.sum()}")
    kw = kruskal(*[g["d_mod3"].values for _, g in strict_bl[strict_bl["dx"].isin(dx_map)].groupby("dx")])
    print(f"Kruskal dx group difference: H={kw.statistic:.2f}, p={kw.pvalue:.2e}")

    print("\n=== STRICT 3-WAY TEST: current dx AUC using d_mod3 directly ===")
    for neg, pos in [("NORMAL", "MCI"), ("MCI", "AD"), ("NORMAL", "AD")]:
        sub = strict_bl[strict_bl["dx"].isin([neg, pos])]
        y = (sub["dx"] == pos).astype(int).values
        s = sub["d_mod3"].values
        auc, lo, hi = boot_auc(y, s)
        rows.append(
            {
                "split": "strict_3way_test",
                "relation": "current_dx",
                "task": f"{neg}_vs_{pos}",
                "horizon_years": np.nan,
                "n": len(sub),
                "n_positive": int(y.sum()),
                "score": "d_mod3",
                "auc_or_rho": auc,
                "ci_lo": lo,
                "ci_hi": hi,
                "extra": "positive=higher_dx",
            }
        )
        print(f"{neg}_vs_{pos:6s} n={len(sub):3d} pos={int(y.sum()):3d} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]")

    strict_ft = pd.read_csv(ROOT / "data/splits_3way_20260627_v2/matched_FINETUNE.csv")
    ft_bl = build_baseline(strict_ft)
    train = ft_bl[ft_bl["dx"].isin(dx_map)].copy()
    test = strict_bl[strict_bl["dx"].isin(dx_map)].copy()
    clf = LogisticRegression(max_iter=2000, multi_class="ovr").fit(
        train[["d_mod3"]].values, train["dx"].map(dx_map).values
    )
    prob = clf.predict_proba(test[["d_mod3"]].values)
    macro = []
    for c, name in enumerate(["NORMAL", "MCI", "AD"]):
        auc = roc_auc_score((test["dx"].map(dx_map).values == c).astype(int), prob[:, c])
        macro.append(auc)
        rows.append(
            {
                "split": "strict_3way_test",
                "relation": "current_dx_multinomial_fit_on_finetune",
                "task": f"{name}_vs_rest",
                "horizon_years": np.nan,
                "n": len(test),
                "n_positive": int((test.dx == name).sum()),
                "score": "LogReg(d_mod3)",
                "auc_or_rho": auc,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "extra": "fit on finetune only",
            }
        )
    print(
        "3-class macro OvR AUC using only d_mod3 logistic fit on finetune: "
        f"{np.mean(macro):.3f} (CN {macro[0]:.3f}, MCI {macro[1]:.3f}, AD {macro[2]:.3f})"
    )

    print("\n=== STRICT 3-WAY TEST: conversion AUC using baseline d_mod3 directly ===")
    for task, bd, td in [("MCI_to_AD", "MCI", "AD"), ("CN_to_MCI", "NORMAL", "MCI")]:
        for h in [1, 2, 3, 4]:
            c = conversion_cohort(strict_test, bd, td, h)
            if len(c) == 0 or c["converted"].nunique() < 2:
                n_conv = int(c.converted.sum()) if len(c) else 0
                print(f"{task:9s} y{h}: n={len(c):3d}, conv={n_conv:3d}, AUC=NA")
                continue
            auc, lo, hi = boot_auc(c["converted"].values, c["d_mod3"].values)
            print(f"{task:9s} y{h}: n={len(c):3d}, conv={int(c.converted.sum()):3d}, AUC={auc:.3f} [{lo:.3f},{hi:.3f}]")
            rows.append(
                {
                    "split": "strict_3way_test",
                    "relation": "conversion",
                    "task": task,
                    "horizon_years": h,
                    "n": len(c),
                    "n_positive": int(c.converted.sum()),
                    "score": "baseline_d_mod3",
                    "auc_or_rho": auc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "extra": "direct score, no fitting",
                }
            )

    old_path = ROOT / "d_contrastive/results_old_split_summary/old_split_exp4_conversion_task_suite.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        oracle = old[old["method"].eq("oracle_true_d_mod3")].copy()
        print("\n=== OLD SPLIT: oracle_true_d_mod3 conversion rows already saved ===")
        print(oracle[["input_version", "task", "horizon_years", "n", "n_converters", "auc", "ci_lo", "ci_hi"]].to_string(index=False))
        for _, o in oracle.iterrows():
            rows.append(
                {
                    "split": "old_split",
                    "relation": "conversion",
                    "task": o["task"],
                    "horizon_years": o["horizon_years"],
                    "n": o["n"],
                    "n_positive": o["n_converters"],
                    "score": "oracle_true_d_mod3",
                    "auc_or_rho": o["auc"],
                    "ci_lo": o["ci_lo"],
                    "ci_hi": o["ci_hi"],
                    "extra": f"input_version={o['input_version']}",
                }
            )

    out = pd.DataFrame(rows)
    out_file = OUT_DIR / "d_mod3_direct_relationship_dx_conversion.csv"
    out.to_csv(out_file, index=False)
    md = OUT_DIR / "d_mod3_direct_relationship_dx_conversion.md"
    with open(md, "w") as f:
        f.write("# d_mod3 direct relationship with dx and conversion\n\n")
        f.write(
            "Main rows use the strict 3-way held-out TEST split. Conversion AUC uses "
            "baseline d_mod3 directly, with no model fitting.\n\n"
        )
        f.write(out.to_markdown(index=False))
        f.write("\n")
    print("\nSaved:", out_file)
    print("Saved:", md)


if __name__ == "__main__":
    main()
