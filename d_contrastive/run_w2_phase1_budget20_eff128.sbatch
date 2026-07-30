#!/bin/bash
#SBATCH --job-name=w2_phase1_b20
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_phase1_budget20_eff128_%j.log

set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export TMPDIR=/dcs07/zwang/data/adni_d/tmp_runtime/${SLURM_JOB_ID}
export MPLCONFIGDIR=${TMPDIR}/mpl
mkdir -p "${MPLCONFIGDIR}"

FEATS=../data/embeddings_128_05152016
MASTER=../data/master_smri_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
ROOT=results_w2_phase1_budget20_eff128
COMMON="--features_dir ${FEATS} --mode 2way --master_dir ${MASTER} --d_csv ${DCSV} --versions raw"

echo "=== Frozen-feature Phase 1 fair rerun: 20 epochs, batch 128, raw only ==="

python run_w0_phase1_diagnosis.py ${COMMON} \
  --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d \
  --epochs 20 --batch_size 128 --device cpu --seed 42 \
  --out_dir "${ROOT}/dx"

python run_w0_conversion_3way.py ${COMMON} \
  --horizons 2 3 4 \
  --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d \
  --epochs 20 --batch_size 128 --finetune_epochs 20 \
  --device cpu --seed 42 \
  --out_dir "${ROOT}/conv"

echo "W2 PHASE1 BUDGET20 EFF128 DONE"
