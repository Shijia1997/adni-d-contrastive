#!/bin/bash
#SBATCH --job-name=w2_direct_full
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-126
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_direct_full_%j.log

set -euo pipefail
cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export TMPDIR=/dcs07/zwang/data/adni_d/tmp_runtime/${SLURM_JOB_ID}
export MPLCONFIGDIR=${TMPDIR}/mpl
export HF_HOME=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}"

python train_w2_direct_lora_downstream.py \
  --adapt full \
  --features_dir ../data/embeddings_128_05152016 \
  --master_dir ../data/master_smri_05152016 \
  --d_csv ../data/master_smri_05152016/D_with_image_paths_full.csv \
  --img_size 128 128 128 \
  --epochs 10 --batch_size 1 --accum_steps 32 \
  --lr 5e-5 --weight_decay 1e-4 \
  --num_workers 2 \
  --out_dir results_w2_direct_full_downstream

echo "W2 DIRECT FULL DOWNSTREAM DONE"
