# Mainline 2-way W2 manifest

This manifest lists the files intentionally kept at the top of
`d_contrastive/` after the July 30, 2026 fair20 cleanup.

## Experiment scope

- Split: RID-disjoint train/test only.
- Input: 128^3 frozen Swin image embeddings or raw SyN-registered volumes for
  encoder adaptation.
- Supervisor: Wang `d_mod3`.
- Excluded for now: age/APOE support analyses.

## Kept code

- `minimal_v0_contrastive.py`
- `experiment_utils.py`
- `run_w0_phase0_dvalue.py`
- `run_w0_phase1_diagnosis.py`
- `run_w0_conversion_3way.py`
- `run_w2_phase1_dsupport.py`
- `train_w2_phase2_swin.py`
- `train_w2_direct_lora_downstream.py`

## Kept submit scripts

- `run_w2_phase1_budget20_eff128.sbatch`
- `run_w2_phase1_conv_budget20_eff128.sbatch`
- `run_w2_budget20_eff128.sbatch`
- `run_w2_phase2_full_retry.sbatch`

## Kept result directories

- `results_w2_fair20_raw_final/`
  - final raw-only fair20 summary tables.

## Archived

Historical one-off scripts, detailed intermediate result folders, old summaries,
module dumps, and obsolete sbatch files were moved to:

- `log/archive_20260730_fair20_cleanup/`

They are not part of the mainline package and are ignored by git.
