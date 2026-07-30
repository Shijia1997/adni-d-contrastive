# ADNI d-contrastive W2 package

This directory contains the current two-way train/test W2 experiment code:
can a frozen or adapted 3D Swin image representation internalize Wang `d_mod3`
and improve image-only diagnosis/conversion prediction?

Age/APOE analyses are intentionally not included in this package.

## Core inputs

From `/dcs07/zwang/data/adni_d`:

- `data/embeddings_128_05152016/`
  - 128^3 frozen Swin embeddings and ComBat harmonized embeddings.
- `data/master_smri_05152016/`
  - matched metadata, train/test RID split, and authoritative `d_mod3`.

## Core code

- `minimal_v0_contrastive.py`
  - frozen 768-d feature baselines, MLP heads, Euclidean/rank/hybrid/regress-d losses.
- `experiment_utils.py`
  - conversion cohorts, censoring checks, bootstrap AUC helpers.
- `run_w0_phase0_dvalue.py`
  - true `d_mod3` and image-predicted `d_hat` versus diagnosis/conversion.
- `run_w0_phase1_diagnosis.py`
  - frozen-feature diagnosis benchmark.
- `run_w0_conversion_3way.py`
  - frozen/adapted-feature conversion benchmark for CN->MCI and MCI->AD.
- `run_w2_phase1_dsupport.py`
  - image-only vs image+`d_hat` vs image+true-d support analysis.
- `train_w2_phase2_swin.py`
  - Swin encoder adaptation with LoRA or full fine-tune using d-supervised losses.
- `train_w2_direct_lora_downstream.py`
  - direct task-label LoRA/full fine-tuning, no d pretraining.

## Current submit scripts

- `run_w2_phase1_budget20_eff128.sbatch`
  - frozen diagnosis plus conversion fair rerun, raw only, 20 epochs.
- `run_w2_phase1_conv_budget20_eff128.sbatch`
  - frozen conversion rerun only, raw only, 20 epochs.
- `run_w2_budget20_eff128.sbatch`
  - LoRA d-adaptation and direct downstream LoRA/full, all 20 epochs with effective batch 128.
- `run_w2_phase2_full_retry.sbatch`
  - full d-adaptation reference, 20 epochs with effective batch 128.

## Current results

The final raw-only fair20 tables are in:

`results_w2_fair20_raw_final/`

- `raw_phase0_d_and_dhat.csv`
- `raw_dsupport.csv`
- `raw_diagnosis_all_methods_fair20.csv`
- `raw_diagnosis_all_methods_fair20_pivot.csv`
- `raw_conversion_all_methods_fair20.csv`
- `raw_conversion_all_methods_fair20_pivot.csv`

Older scripts, old result folders, old summaries, and module dumps were moved to:

`log/archive_20260730_fair20_cleanup/`

The archive also contains detailed intermediate run directories such as
`results_w2_phase1_budget20_eff128/` and `results_w2_budget20_eff128/`. It is
for local provenance only and should not be used as the main result source.

## Environment

On JHPCE:

```bash
source /users/szhang1/fsl/bin/activate optuna_env
cd /dcs07/zwang/data/adni_d/d_contrastive
```

Required packages include `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`,
`nibabel`, and the local BTCV SwinUNETR code at
`/dcs07/zwang/data/pmrc/SwinUNETR/BTCV`.

## Main read

- `d_mod3` is a strong AD axis.
- Frozen Swin already recovers deployable `d_hat`.
- Direct d-regression LoRA is strongest for diagnosis.
- Euclidean/hybrid d-supervised adaptation is more useful for conversion.
- Full fine-tune and direct downstream fine-tune are included as comparators,
  but the main deployable story is image-only inference from frozen/adapted Swin
  features without recomputing Wang `d_mod3`.
