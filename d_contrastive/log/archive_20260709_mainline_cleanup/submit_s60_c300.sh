#!/bin/bash
#SBATCH -J dcon_s60_c300
#SBATCH -p shared
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/dcon_s60_c300_%j.out
#SBATCH -e logs/dcon_s60_c300_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_s60_c300

source /users/szhang1/fsl/bin/activate optuna_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== d_mod3 contrastive comparison, 4 setups, contrastive 300 epochs ==="
echo "Started: $(date)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Threads: ${SLURM_CPUS_PER_TASK}"
echo "Setup 1: direct 768-dim embedding -> LR/Ridge"
echo "Setup 2: supervised MLP, 60 epochs"
echo "Setup 3: contrastive MLP 300 epochs -> frozen embedding -> LR/Ridge"
echo "Setup 4: contrastive MLP 300 epochs -> supervised finetune MLP 60 epochs"
echo "Input versions: raw combat"
echo "Contrastive loss print: epoch 1 and every 5 epochs"
echo ""

python minimal_v0_contrastive.py \
  --device cpu \
  --versions raw combat \
  --supervised_epochs 60 \
  --epochs 300 \
  --output_dir results_s60_c300

echo ""
echo "Done: $(date)"
