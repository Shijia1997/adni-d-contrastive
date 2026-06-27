#!/bin/bash
#SBATCH --job-name=euclid0515cls
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/euclidean_0515_h384_l256_cls_%j.log

set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs results_plain_euclidean_05152016_s60_c150_h384_l256_cls_tasks

source /users/szhang1/fsl/bin/activate optuna_env

python minimal_v0_contrastive.py \
  --features_dir ../data/embeddings_128_05152016 \
  --d_csv ../data/master_smri_05152016/D_with_image_paths_full.csv \
  --output_dir results_plain_euclidean_05152016_s60_c150_h384_l256_cls_tasks \
  --device cpu \
  --versions raw combat \
  --supervised_epochs 60 \
  --epochs 150 \
  --mlp_hidden 384 \
  --mlp_latent 256 \
  --tau 1.0 \
  --beta 1.0
