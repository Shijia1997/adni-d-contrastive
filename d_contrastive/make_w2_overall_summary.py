"""Build a compact W2 mainline result summary.

This aggregates the current 2-split d-contrastive mainline:
  - Phase 0: true d / deployable d_hat relation to dx + conversion
  - Phase 1: frozen Swin embedding downstream
  - Phase 1 support: image + d_hat / true-d
  - Phase 2: LoRA-adapted Swin embeddings, followed by the same downstream

Age/APOE analyses are intentionally excluded.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_w2_overall_summary"


def read_csv(path):
    path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_source(df, stage, source, result_type):
    if df.empty:
        return df
    df = df.copy()
    df.insert(0, "stage", stage)
    df.insert(1, "source", source)
    df.insert(2, "result_type", result_type)
    return df


def compact_dx(df):
    keep = [
        "baseline_raw",
        "ridge_dhat_all",
        "contrastive_euclidean_probe",
        "contrastive_rank_kendall_basic_probe",
        "contrastive_hybrid_basic_probe",
        "contrastive_regress_d_probe",
    ]
    sub = df[df["method"].isin(keep)].copy()
    return sub[["stage", "source", "version", "task_kind", "task", "method", "n", "n_positive", "auc", "ci_lo", "ci_hi"]]


def compact_conv(df):
    keep_contains = [
        "ridge_dhat",
        "direct_logistic",
        "contrastive_euclidean_probe",
        "contrastive_euclidean_s",
        "contrastive_euclidean_finetune",
        "contrastive_rank_kendall_basic_probe",
        "contrastive_rank_kendall_basic_s",
        "contrastive_rank_kendall_basic_finetune",
        "contrastive_hybrid_basic_probe",
        "contrastive_hybrid_basic_s",
        "contrastive_hybrid_basic_finetune",
        "contrastive_regress_d_probe",
        "contrastive_regress_d_s",
        "contrastive_regress_d_finetune",
    ]
    sub = df[df["method"].isin(keep_contains)].copy()
    cols = ["stage", "source", "version", "task", "horizon_years", "method", "n", "n_converters", "auc", "ci_lo", "ci_hi"]
    return sub[cols]


def best_by_task(df, group_cols):
    if df.empty:
        return df
    idx = df.groupby(group_cols)["auc"].idxmax()
    return df.loc[idx].sort_values(group_cols).reset_index(drop=True)


def pivot_best_conversion(best):
    if best.empty:
        return best
    rows = []
    for (stage, source, version, task), g in best.groupby(["stage", "source", "version", "task"]):
        row = {"stage": stage, "source": source, "version": version, "task": task}
        for _, r in g.iterrows():
            h = int(r["horizon_years"])
            row[f"y{h}_auc"] = r["auc"]
            row[f"y{h}_method"] = r["method"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "stage", "source", "version"]).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Phase 0: true d and d_hat.
    phase0 = add_source(read_csv("results_w2_phase0/w2_phase0_dvalue.csv"), "phase0_d_relation", "frozen_swin", "mixed")

    # Phase 1 frozen.
    dx = add_source(read_csv("results_w2_phase1_dx/w2_phase1_diagnosis.csv"), "phase1_frozen", "frozen_swin", "diagnosis")
    conv = add_source(read_csv("results_w2_phase1_conv/w0_conversion_3way.csv"), "phase1_frozen", "frozen_swin", "conversion")

    # Phase 1 d support.
    dsup = add_source(read_csv("results_w2_phase1_dsupport/w2_phase1_dsupport.csv"), "phase1_d_support", "frozen_swin", "diagnosis_conversion")

    # Phase 2 LoRA adapted embeddings.
    lora_dx, lora_conv, lora_phase0 = [], [], []
    for source in ["lora_regress_d", "lora_euclidean", "lora_hybrid_basic"]:
        lora_dx.append(add_source(read_csv(f"results_w2_phase2/{source}/dx/w2_phase1_diagnosis.csv"), "phase2_lora_adapted_embedding", source, "diagnosis"))
        lora_conv.append(add_source(read_csv(f"results_w2_phase2/{source}/conv/w0_conversion_3way.csv"), "phase2_lora_adapted_embedding", source, "conversion"))
        lora_phase0.append(add_source(read_csv(f"results_w2_phase2/{source}/phase0/w2_phase0_dvalue.csv"), "phase2_lora_adapted_embedding", source, "d_relation"))

    dx_all = pd.concat([compact_dx(dx)] + [compact_dx(x) for x in lora_dx if not x.empty], ignore_index=True)
    conv_all = pd.concat([compact_conv(conv)] + [compact_conv(x) for x in lora_conv if not x.empty], ignore_index=True)

    # Phase 0 and d-support kept separately because their schemas differ.
    phase0.to_csv(OUT / "w2_d_relation_all.csv", index=False)
    dsup.to_csv(OUT / "w2_d_support_all.csv", index=False)
    dx_all.to_csv(OUT / "w2_diagnosis_all_methods.csv", index=False)
    conv_all.to_csv(OUT / "w2_conversion_all_methods.csv", index=False)

    best_dx = best_by_task(dx_all, ["stage", "source", "version", "task"])
    best_conv = best_by_task(conv_all, ["stage", "source", "version", "task", "horizon_years"])
    conv_pivot = pivot_best_conversion(best_conv)

    best_dx.to_csv(OUT / "w2_diagnosis_best_by_task.csv", index=False)
    best_conv.to_csv(OUT / "w2_conversion_best_by_horizon.csv", index=False)
    conv_pivot.to_csv(OUT / "w2_conversion_best_pivot.csv", index=False)

    # A readable markdown report.
    md = []
    md.append("# W2 d-contrastive overall summary\n")
    md.append("Age/APOE analyses are not included.\n")
    md.append("Two-split setup: train/fine-tune data vs held-out test. Test `d_mod3` is not used for deployable model fitting.\n")
    md.append("\n## Completion status\n")
    md.append("- D vs diagnosis/conversion: done (`results_w2_phase0`).\n")
    md.append("- Frozen image embedding downstream: done (`results_w2_phase1_dx`, `results_w2_phase1_conv`).\n")
    md.append("- d-pretraining arms on frozen embeddings: done (`euclidean`, `rank_kendall_basic`, `hybrid_basic`, `regress_d`).\n")
    md.append("- Image + d support analysis: done (`results_w2_phase1_dsupport`).\n")
    md.append("- LoRA d-adapted Swin embeddings: done for `lora_regress_d`, `lora_euclidean`, `lora_hybrid_basic`.\n")
    md.append("- Full fine-tune: attempted but failed with CUDA OOM before producing results.\n")
    md.append("- Direct LoRA fine-tune on downstream diagnosis/conversion labels: not run in current result set; current LoRA is d-adapted encoder followed by downstream probes/fine-tune heads.\n")

    md.append("\n## Best diagnosis rows\n")
    md.append(best_dx.round(4).to_markdown(index=False))
    md.append("\n\n## Best conversion by horizon\n")
    md.append(best_conv.round(4).to_markdown(index=False))
    md.append("\n\n## Best conversion pivot\n")
    md.append(conv_pivot.round(4).to_markdown(index=False))

    md.append("\n\n## d-support result table\n")
    if not dsup.empty:
        keep = dsup[["stage", "source", "version", "task_kind", "task", "method", "n", "n_positive", "auc", "ci_lo", "ci_hi"]]
        md.append(keep.round(4).to_markdown(index=False))

    (OUT / "W2_OVERALL_SUMMARY.md").write_text("\n".join(md))
    print(f"Wrote {OUT}")
    print("Key files:")
    for p in sorted(OUT.glob("*")):
        print(" ", p)


if __name__ == "__main__":
    main()
