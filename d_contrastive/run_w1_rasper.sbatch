#!/bin/bash
#SBATCH --job-name=w1_rasper
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w1_rasper_%j.log

# Workstream 1: RASPER for conversion (option-A external ranker = Ridge d_hat).
# CPU-only, pure numpy/scipy/sklearn.
set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env

# Self-check the estimator first (fast, no data).
python test_rasper_sim.py

python run_w1_rasper_conversion.py \
  --features_dir ../data/embeddings_128_05152016 \
  --d_csv ../data/master_smri_05152016/D_with_image_paths_full.csv \
  --versions raw combat \
  --horizons 2 3 4 \
  --n_pca 16 \
  --penalties kendall \
  --cv_splits 5 \
  --out_dir results_w1_rasper

echo "W1 RASPER DONE"
