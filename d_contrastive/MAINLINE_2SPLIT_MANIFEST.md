# d-contrastive mainline 2-split package

This directory is trimmed to the current mainline experiment:

1. Test whether Wang `d_mod3` tracks diagnosis and conversion.
2. Test whether frozen/adapted Swin image embeddings learn usable `d_mod3` information.
3. Evaluate downstream diagnosis and conversion with two splits: train and held-out test.
4. Compare frozen embeddings, d-supervised heads, explicit d-support, and LoRA-adapted Swin embeddings.

Age/APOE support analyses are intentionally not included here yet.

## Core code

- `minimal_v0_contrastive.py`
- `experiment_utils.py`
- `compute_d_mod3_direct_relationship.py`
- `run_w0_phase0_dvalue.py`
- `run_w0_phase1_diagnosis.py`
- `run_w0_conversion_3way.py`
- `run_w2_phase1_dsupport.py`
- `train_w2_phase2_swin.py`
- `run_w2_phase01.sbatch`
- `run_w2_phase2_swin.sbatch`

## Main results

- `results_w2_phase0/`
  - true-d oracle and deployable `d_hat` relation to diagnosis/conversion
- `results_w2_phase1_dx/`
  - frozen image embedding diagnosis downstream results
- `results_w2_phase1_conv/`
  - frozen image embedding conversion downstream results
- `results_w2_phase1_dsupport/`
  - image-only vs image + `d_hat` vs image + true-d support analysis
- `results_w2_phase2/`
  - LoRA-adapted Swin embedding runs and downstream evaluations

## Archive

Older smoke tests, old split/3-way runs, RASP/RASPER experiments, rank sweeps,
download helpers, and historical summaries were moved to:

- `log/archive_20260709_mainline_cleanup/`

No files were deleted during this cleanup.
