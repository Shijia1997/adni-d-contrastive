#!/bin/bash
#SBATCH --job-name=w0_rank
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w0_rank_ablation_%j.log

# Workstream 0: pure-rank fix + loss-geometry ablation + alpha/nu sweep.
# All runs are CPU-only (small MLP on frozen 768-d Swin features).
set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export MPLCONFIGDIR=/tmp/mpl_w0_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

FEATS=../data/embeddings_128_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
COMMON="--features_dir ${FEATS} --d_csv ${DCSV} --versions raw combat --epochs 150 --device cpu"

# ---------------------------------------------------------------------------
# Part 1: loss-geometry comparison (same code, 5 modes, separate out dirs).
#   euclidean            = Y-Aware Euclidean contrastive (no rank)
#   rank_kendall         = ORIGINAL tanh+alpha surrogate (not pure rank)
#   rank_kendall_basic   = NEW principled smoothed Kendall (sigmoid + nu)  <-- pure-rank fix
#   hybrid               = euclidean + lambda * original rank
#   hybrid_basic         = euclidean + lambda * basic rank
# ---------------------------------------------------------------------------
for MODE in euclidean rank_kendall rank_kendall_basic hybrid hybrid_basic; do
  echo "=== loss_mode=${MODE} ==="
  python minimal_v0_contrastive.py ${COMMON} \
    --loss_mode ${MODE} \
    --output_dir results_w0/mode_${MODE}
done

# ---------------------------------------------------------------------------
# Part 2: sharpness/smoothing sweep -- shows how much "magnitude leak" the
# original alpha injects vs the principled nu. Read the setup3 rows.
# ---------------------------------------------------------------------------
for A in 1 2 5 10 20 50; do
  echo "=== original soft_kendall alpha=${A} ==="
  python minimal_v0_contrastive.py ${COMMON} \
    --loss_mode rank_kendall --rank_alpha ${A} \
    --output_dir results_w0/sweep_alpha_${A}
done

# basic Kendall: auto-nu (principled default) + a few explicit nu values
echo "=== basic Kendall nu=auto ==="
python minimal_v0_contrastive.py ${COMMON} \
  --loss_mode rank_kendall_basic \
  --output_dir results_w0/sweep_nu_auto
for NU in 0.05 0.1 0.25 0.5 1.0; do
  echo "=== basic Kendall nu=${NU} ==="
  python minimal_v0_contrastive.py ${COMMON} \
    --loss_mode rank_kendall_basic --rank_nu ${NU} \
    --output_dir results_w0/sweep_nu_${NU}
done

echo "W0 RANK ABLATION DONE"
