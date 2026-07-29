#!/bin/bash
#SBATCH --job-name=w0_conv_3way
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w0_conv_3way_%j.log

# Workstream 0 (3-way split): contrastive conversion + two ridge d_hat baselines.
#   contrastive split -> pretrain encoder on d_mod3 (main method)
#   finetune split    -> train downstream conversion probe
#   test split        -> eval ONLY; its d_mod3 is never used for any fit.
# CPU-only (small MLP on frozen 768-d Swin features).
set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env

FEATS=../data/embeddings_128_05152016
SPLIT=../data/splits_3way_20260627_v2
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv

# ---------------------------------------------------------------------------
# Step 0: sanity self-check (no training). Verifies the loader finds all three
# splits, that they are RID-disjoint, and prints conversion-cohort counts.
# Aborts the job (set -e) if any split is missing or leaks.
# ---------------------------------------------------------------------------
echo "=== self-check: load 3-way split + leakage + cohort counts ==="
python - "$FEATS" "$SPLIT" "$DCSV" <<'PY'
import sys
from minimal_v0_contrastive import load_features_3way
from experiment_utils import build_censored_conversion_cohort
feats, split, dcsv = sys.argv[1], sys.argv[2], sys.argv[3]
data = load_features_3way(feats, split, dcsv, version="raw")
TASKS = [("MCI_to_AD", "MCI", "AD"), ("CN_to_MCI", "NORMAL", "MCI")]
for name in ("contrastive", "finetune", "test"):
    meta = data[name].get("long_meta", data[name]["meta"])
    for task, base, tgt in TASKS:
        for h in (2, 3, 4):
            c = build_censored_conversion_cohort(meta, name, f"{task}_{h}y", base, tgt, h)
            c = c[c["valid_censored"].astype(bool)]
            print(f"  {name:11s} {task:9s} {h}y: n={len(c):4d} converters={int(c['converted'].sum())}")
print("SELF-CHECK OK")
PY

# ---------------------------------------------------------------------------
# Step 1: full run -- raw + combat, horizons 2/3/4, three contrastive geometries.
# ---------------------------------------------------------------------------
echo "=== full run ==="
python run_w0_conversion_3way.py \
  --features_dir "$FEATS" \
  --split_dir "$SPLIT" \
  --d_csv "$DCSV" \
  --versions raw combat \
  --horizons 2 3 4 \
  --loss_modes euclidean rank_kendall_basic hybrid_basic \
  --epochs 150 \
  --device cpu \
  --out_dir results_w0_conv_3way

echo "W0 CONVERSION 3WAY DONE"
