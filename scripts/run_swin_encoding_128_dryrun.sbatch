#!/bin/bash
#SBATCH --job-name=adni_swin128_dry
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/swin_encoding_128_dryrun_%j.log
#SBATCH --exclude=compute-117

set -euo pipefail

cd /dcs07/zwang/data/adni_d
mkdir -p logs data/embeddings_128_dryrun data/tmp_hf_cache

export HF_HOME=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export TRANSFORMERS_CACHE=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source /users/szhang1/fsl/bin/activate optuna_env

python scripts/encode_swin_smri.py \
  --matched_csv data/master_smri/D_with_image_paths_matched.csv \
  --out_dir data/embeddings_128_dryrun \
  --model_id darragh/swinunetr-btcv-base \
  --img_size 128 128 128 \
  --limit 4 \
  --print_every 1
