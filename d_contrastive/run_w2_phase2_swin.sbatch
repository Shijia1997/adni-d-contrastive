#!/bin/bash
#SBATCH --job-name=w2_phase2_swin
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-126
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_phase2_swin_%j.log
# NOTE: confirm the SLURM partition that owns compute-126 (add e.g.
#   #SBATCH --partition=<gpu-partition>  and  #SBATCH --account/--qos if required).

# Workstream 2, Phase 2: unfreeze the Swin on d (LoRA + full fine-tune), then run
# the SAME frozen downstream on the ADAPTED 768-d embeddings. GPU node only.
#   physical batch 8 + grad-accum 16 (effective 128) + gradient checkpointing on.
set -euo pipefail

cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export TMPDIR=/dcs07/zwang/data/adni_d/tmp_runtime/${SLURM_JOB_ID}
export MPLCONFIGDIR=${TMPDIR}/mpl
mkdir -p "${MPLCONFIGDIR}"
export HF_HOME=/dcs07/zwang/data/adni_d/data/tmp_hf_cache

# MODE=2way: pretrain on `train`, export train+test, downstream on train/test.
MASTER=../data/master_smri_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
IMG="128 128 128"
ROOT=results_w2_phase2
MODEARGS="--mode 2way --master_dir ${MASTER}"

# ---------------------------------------------------------------------------
# Step 0: introspect swinViT Linear names, then a tiny LoRA forward/backward +
# export smoke. If the LoRA target regex matches nothing, this aborts here with
# the real module names (set -e) BEFORE any long training.
# ---------------------------------------------------------------------------
echo "=== step 0a: swinViT Linear module tree ==="
python train_w2_phase2_swin.py ${MODEARGS} --d_csv "$DCSV" \
  --print_modules --out_dir "${ROOT}/_modules" | tee "${ROOT}_modules.txt" || true

echo "=== step 0b: LoRA train+export smoke (16 imgs, 1 epoch) ==="
python train_w2_phase2_swin.py ${MODEARGS} --d_csv "$DCSV" \
  --img_size ${IMG} --adapt lora --loss_mode regress_d \
  --limit 16 --epochs 1 --batch_size 8 --accum_steps 2 \
  --out_dir "${ROOT}/_smoke/emb"
echo "SMOKE OK"

# ---------------------------------------------------------------------------
# Step 1: full runs. adapt x loss-geometry; export adapted 768 -> run the frozen
# CPU downstream (Phase 0/1) on it, raw only (combat has no adapted counterpart).
# Extend the MODES/ADAPT lists as compute allows.
# ---------------------------------------------------------------------------
for ADAPT in lora full; do
  for MODE in regress_d euclidean hybrid_basic; do
    OUT="${ROOT}/${ADAPT}_${MODE}"
    echo "=== train ${ADAPT} / ${MODE} ==="
    python train_w2_phase2_swin.py ${MODEARGS} --d_csv "$DCSV" \
      --img_size ${IMG} --adapt "${ADAPT}" --loss_mode "${MODE}" \
      --epochs 30 --batch_size 8 --accum_steps 16 --out_dir "${OUT}/emb"

    echo "=== downstream on adapted-768: ${ADAPT}/${MODE} ==="
    COMMON="--features_dir ${OUT}/emb ${MODEARGS} --d_csv ${DCSV} --versions raw"
    python run_w0_phase0_dvalue.py    ${COMMON} --horizons 2 3 4 --out_dir "${OUT}/phase0"
    python run_w0_phase1_diagnosis.py ${COMMON} --device cpu \
      --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/dx"
    python run_w0_conversion_3way.py  ${COMMON} --device cpu --horizons 2 3 4 \
      --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/conv"
  done
done

echo "W2 PHASE 2 DONE"
