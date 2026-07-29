#!/bin/bash
#SBATCH --job-name=w2_budget20_128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_budget20_eff128_%j.log

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

MASTER=../data/master_smri_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
IMG="128 128 128"
ROOT=results_w2_budget20_eff128
MODEARGS="--mode 2way --master_dir ${MASTER}"

echo "=== Unified budget: 20 epochs, effective batch 128, seed 42 ==="

# ---------------------------------------------------------------------------
# D-supervised LoRA encoder adaptation, shared encoder per d objective.
# This matches full_retry's epoch count and effective batch size.
# ---------------------------------------------------------------------------
for MODE in regress_d euclidean hybrid_basic; do
  OUT="${ROOT}/lora_${MODE}"
  echo "=== train LoRA d-adapt / ${MODE} ==="
  python train_w2_phase2_swin.py ${MODEARGS} --d_csv "$DCSV" \
    --img_size ${IMG} --adapt lora --loss_mode "${MODE}" \
    --epochs 20 --batch_size 2 --accum_steps 64 --seed 42 \
    --out_dir "${OUT}/emb"

  echo "=== downstream on adapted-768: lora/${MODE} ==="
  COMMON="--features_dir ${OUT}/emb ${MODEARGS} --d_csv ${DCSV} --versions raw"
  python run_w0_phase0_dvalue.py    ${COMMON} --horizons 2 3 4 --out_dir "${OUT}/phase0"
  python run_w0_phase1_diagnosis.py ${COMMON} --device cpu --seed 42 \
    --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/dx"
  python run_w0_conversion_3way.py  ${COMMON} --device cpu --seed 42 --horizons 2 3 4 \
    --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/conv"
done

# ---------------------------------------------------------------------------
# Direct downstream fine-tuning, task-specific encoder. Same epoch count and
# effective batch 128. LoRA uses physical batch 2; full uses physical batch 1.
# ---------------------------------------------------------------------------
echo "=== direct downstream LoRA, 20 epochs, effective batch 128 ==="
python train_w2_direct_lora_downstream.py \
  --adapt lora \
  --features_dir ../data/embeddings_128_05152016 \
  --master_dir "$MASTER" \
  --d_csv "$DCSV" \
  --img_size ${IMG} \
  --epochs 20 --batch_size 2 --accum_steps 64 \
  --lr 5e-4 --weight_decay 1e-4 --seed 42 \
  --out_dir "${ROOT}/direct_lora_downstream"

echo "=== direct downstream FULL, 20 epochs, effective batch 128 ==="
python train_w2_direct_lora_downstream.py \
  --adapt full \
  --features_dir ../data/embeddings_128_05152016 \
  --master_dir "$MASTER" \
  --d_csv "$DCSV" \
  --img_size ${IMG} \
  --epochs 20 --batch_size 1 --accum_steps 128 \
  --lr 5e-5 --weight_decay 1e-4 --seed 42 \
  --out_dir "${ROOT}/direct_full_downstream"

echo "W2 BUDGET20 EFF128 DONE"
