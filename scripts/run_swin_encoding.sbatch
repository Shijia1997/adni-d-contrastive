#!/bin/bash
#SBATCH --job-name=adni_swin_encode
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/swin_encoding_%j.log
#SBATCH --exclude=compute-117

set -euo pipefail

cd /dcs07/zwang/data/adni_d
mkdir -p logs data/embeddings data/tmp_hf_cache

export HF_HOME=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export TRANSFORMERS_CACHE=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source /users/szhang1/fsl/bin/activate optuna_env

python scripts/encode_swin_smri.py \
  --matched_csv data/master_smri/D_with_image_paths_matched.csv \
  --out_dir data/embeddings \
  --model_id darragh/swinunetr-btcv-base \
  --img_size 96 96 96 \
  --print_every 25

python scripts/check_swin_disease_signal.py \
  --train_csv data/master_smri/matched_TRAIN.csv \
  --latent data/embeddings/swin_latent.npy \
  --image_ids data/embeddings/image_id_order.npy \
  --out_txt data/embeddings/swin_cn_ad_auc.txt
