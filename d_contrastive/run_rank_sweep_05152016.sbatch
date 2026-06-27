#!/bin/bash
#SBATCH --job-name=rank0515
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/rank_sweep_0515_%j.log

set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env

run_one() {
  local mode="$1"
  local alpha="$2"
  local lambda="$3"
  local tag_lambda
  tag_lambda="${lambda/./p}"

  local out
  if [[ "${mode}" == "euclidean" ]]; then
    out="results_rank_euclidean_a${alpha}_05152016_s60_c150_h384_l256"
  elif [[ "${mode}" == "rank_kendall" ]]; then
    out="results_rank_rank_kendall_a${alpha}_05152016_s60_c150_h384_l256"
  else
    out="results_rank_hybrid_a${alpha}_l${tag_lambda}_05152016_s60_c150_h384_l256"
  fi

  echo
  echo "================================================================"
  echo "RUN ${mode} alpha=${alpha} lambda=${lambda} -> ${out}"
  echo "================================================================"

  python minimal_v0_contrastive.py \
    --features_dir ../data/embeddings_128_05152016 \
    --d_csv ../data/master_smri_05152016/D_with_image_paths_full.csv \
    --output_dir "${out}" \
    --device cpu \
    --versions raw combat \
    --supervised_epochs 60 \
    --epochs 150 \
    --mlp_hidden 384 \
    --mlp_latent 256 \
    --tau 1.0 \
    --beta 1.0 \
    --loss_mode "${mode}" \
    --rank_alpha "${alpha}" \
    --lambda_rank "${lambda}"
}

# Euclidean sanity. Alpha/lambda are logged but not used by the euclidean loss.
run_one euclidean 10 1.0

for alpha in 4 10 20; do
  run_one rank_kendall "${alpha}" 1.0
done

for alpha in 4 10 20; do
  for lambda in 0.3 1.0; do
    run_one hybrid "${alpha}" "${lambda}"
  done
done

python consolidate_rank_results.py
