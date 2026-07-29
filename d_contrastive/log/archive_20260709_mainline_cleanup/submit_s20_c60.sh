#!/bin/bash
#SBATCH -J dcon_s20_c60
#SBATCH -p shared
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/dcon_s20_c60_%j.out
#SBATCH -e logs/dcon_s20_c60_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_s20_c60

source /users/szhang1/fsl/bin/activate optuna_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== d_mod3 contrastive comparison ==="
echo "Started: $(date)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Threads: ${SLURM_CPUS_PER_TASK}"
echo "Supervised MLP epochs: 20"
echo "Contrastive epochs: 60"
echo "Contrastive loss print: epoch 1 and every 5 epochs"
echo ""

python minimal_v0_contrastive.py \
  --device cpu \
  --versions raw combat \
  --supervised_epochs 20 \
  --epochs 60 \
  --output_dir results_s20_c60

echo ""
echo "Done: $(date)"
