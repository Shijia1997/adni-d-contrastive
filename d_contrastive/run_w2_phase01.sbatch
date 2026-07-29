#!/bin/bash
#SBATCH --job-name=w2_phase01
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_phase01_%j.log

# Workstream 2, Phases 0 + 1 (FROZEN encoder, image-only). These operate on the
# precomputed 768-d Swin embeddings, so they are CPU-only and run on `shared`.
#   NOTE: the GPU node `compute-126` is reserved for the Phase-2 Swin/LoRA jobs
#   (raw volumes through the backbone) -- NOT for these frozen-embedding phases.
set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export TMPDIR=/dcs07/zwang/data/adni_d/tmp_runtime/${SLURM_JOB_ID}
export MPLCONFIGDIR=${TMPDIR}/mpl
mkdir -p "${MPLCONFIGDIR}"

# MODE=2way (encoder pretrain + downstream head both on `train`, `test` held out).
# The previous partition, sourced from matched_TRAIN.csv / matched_TEST.csv.
FEATS=../data/embeddings_128_05152016
MASTER=../data/master_smri_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
COMMON="--features_dir ${FEATS} --mode 2way --master_dir ${MASTER} --d_csv ${DCSV} --versions raw combat"

# ---------------------------------------------------------------------------
# Step 0: self-check -- 2-way loader + train/test RID-disjointness + regress_d.
# ---------------------------------------------------------------------------
echo "=== self-check: 2-way loader + regress_d encoder smoke ==="
python - "$FEATS" "$MASTER" "$DCSV" <<'PY'
import sys, numpy as np
from minimal_v0_contrastive import load_features_2way, train_contrastive_encoder, encode_features
feats, master, dcsv = sys.argv[1], sys.argv[2], sys.argv[3]
data = load_features_2way(feats, master, dcsv, version="raw")
assert data["_mode"] == "2way"
Xtr = data["train"]["features"]; dtr = data["train"]["meta"]["d_mod3"].values.astype(float)
m = np.isfinite(dtr)
model, hist = train_contrastive_encoder(Xtr[m][:256], dtr[m][:256], epochs=2,
                                        loss_mode="regress_d", verbose=True)
z, s = encode_features(model, Xtr[:8])
assert z.shape[0] == 8 and s.shape[0] == 8
assert np.isfinite(hist[-1]["loss_reg"])
print("SELF-CHECK OK (2way loads, regress_d trains, loss_reg finite)")
PY

# ---------------------------------------------------------------------------
# Phase 0: d-value ceiling (true-d oracle vs frozen d_hat). No training.
# ---------------------------------------------------------------------------
echo "=== Phase 0: d-value ceiling ==="
python run_w0_phase0_dvalue.py ${COMMON} \
  --horizons 2 3 4 --ridge_alpha 10.0 --out_dir results_w2_phase0

# ---------------------------------------------------------------------------
# Phase 1a: diagnosis (3-class + binaries), arms a/b/c on frozen embeddings.
# ---------------------------------------------------------------------------
echo "=== Phase 1: diagnosis ==="
python run_w0_phase1_diagnosis.py ${COMMON} \
  --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d \
  --epochs 150 --device cpu --out_dir results_w2_phase1_dx

# ---------------------------------------------------------------------------
# Phase 1b: conversion, SAME arms -- reuse the W0 conversion driver, now with the
# new regress_d supervision arm added to the sweep.
# ---------------------------------------------------------------------------
echo "=== Phase 1: conversion (with regress_d arm) ==="
python run_w0_conversion_3way.py ${COMMON} \
  --horizons 2 3 4 \
  --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d \
  --epochs 150 --device cpu --out_dir results_w2_phase1_conv

# ---------------------------------------------------------------------------
# Phase 1c: d-as-support (R2 ceiling) -- image_only vs image+dhat vs image+dtrue.
# ---------------------------------------------------------------------------
echo "=== Phase 1: d-as-support ==="
python run_w2_phase1_dsupport.py ${COMMON} \
  --horizons 2 3 4 --ridge_alpha 10.0 --out_dir results_w2_phase1_dsupport

echo "W2 PHASE 0+1 DONE"
