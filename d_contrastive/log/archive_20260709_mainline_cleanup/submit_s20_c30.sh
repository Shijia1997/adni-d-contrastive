#!/bin/bash
#SBATCH -J dcon_s20_c30
#SBATCH -p shared
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/dcon_s20_c30_%j.out
#SBATCH -e logs/dcon_s20_c30_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_s20_c30

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
echo "Contrastive epochs: 30"
echo ""

python minimal_v0_contrastive.py \
  --device cpu \
  --versions raw combat \
  --supervised_epochs 20 \
  --epochs 30 \
  --output_dir results_s20_c30

echo ""
echo "Done: $(date)"
