#!/bin/bash
#SBATCH --job-name=exp1dxpre
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/exp1_dx_pretrain_%j.log

set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs results_exp1_dx_pretrain

source /users/szhang1/fsl/bin/activate optuna_env

python run_exp1_dx_pretrain.py \
  --seeds 0 1 2 3 4 \
  --epochs 150 \
  --finetune_epochs 60 \
  --device cpu \
  --out_dir results_exp1_dx_pretrain
