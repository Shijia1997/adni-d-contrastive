#!/bin/bash
#SBATCH --job-name=w2_full_retry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/dcs07/zwang/data/adni_d/logs/w2_full_retry_%j.log

set -euo pipefail
cd /dcs07/zwang/data/adni_d/d_contrastive
mkdir -p /dcs07/zwang/data/adni_d/logs

source /users/szhang1/fsl/bin/activate optuna_env
export TMPDIR=/dcs07/zwang/data/adni_d/tmp_runtime/${SLURM_JOB_ID}
export MPLCONFIGDIR=${TMPDIR}/mpl
export HF_HOME=/dcs07/zwang/data/adni_d/data/tmp_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}"

MASTER=../data/master_smri_05152016
DCSV=../data/master_smri_05152016/D_with_image_paths_full.csv
IMG="128 128 128"
ROOT=results_w2_phase2
MODEARGS="--mode 2way --master_dir ${MASTER}"

# Full fine-tune retry: smaller physical batch than the failed batch=8 run.
for MODE in regress_d euclidean hybrid_basic; do
  OUT="${ROOT}/full_${MODE}"
  echo "=== full retry ${MODE} ==="
  python train_w2_phase2_swin.py ${MODEARGS} --d_csv "$DCSV" \
    --img_size ${IMG} --adapt full --loss_mode "${MODE}" \
    --epochs 20 --batch_size 2 --accum_steps 64 --lr 5e-5 \
    --out_dir "${OUT}/emb"

  COMMON="--features_dir ${OUT}/emb ${MODEARGS} --d_csv ${DCSV} --versions raw"
  python run_w0_phase0_dvalue.py    ${COMMON} --horizons 2 3 4 --out_dir "${OUT}/phase0"
  python run_w0_phase1_diagnosis.py ${COMMON} --device cpu \
    --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/dx"
  python run_w0_conversion_3way.py  ${COMMON} --device cpu --horizons 2 3 4 \
    --loss_modes euclidean rank_kendall_basic hybrid_basic regress_d --out_dir "${OUT}/conv"
done

echo "W2 FULL RETRY DONE"
