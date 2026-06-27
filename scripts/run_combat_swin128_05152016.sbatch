#!/bin/bash
#SBATCH --job-name=combat128_0515
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/combat_swin128_0515_%j.log

set -euo pipefail

cd /dcs07/zwang/data/adni_d
mkdir -p logs data/embeddings_128_05152016

source /users/szhang1/fsl/bin/activate optuna_env

python scripts/combat_swin_site.py \
  --train_csv data/master_smri_05152016/matched_TRAIN.csv \
  --test_csv data/master_smri_05152016/matched_TEST.csv \
  --image_meta data/master_smri_05152016/smri_image_metadata_used.csv \
  --latent data/embeddings_128_05152016/swin_latent.npy \
  --image_ids data/embeddings_128_05152016/image_id_order.npy \
  --out_dir data/embeddings_128_05152016 \
  --pre_cn_ad_auc 0.799178
