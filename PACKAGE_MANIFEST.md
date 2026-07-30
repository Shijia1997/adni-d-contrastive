# ADNI d-contrastive package manifest

This repository is trimmed for the current W2 fair20 two-way experiment. It keeps
the core code, compact frozen/adapted-result summaries, 128^3 Swin embeddings,
and matched metadata. Historical scripts/results/logs are archived locally under
`d_contrastive/log/` and ignored by git.

## Core Code

```bash
git add .gitignore PACKAGE_MANIFEST.md
git add d_contrastive/README.md d_contrastive/MAINLINE_2SPLIT_MANIFEST.md
git add d_contrastive/minimal_v0_contrastive.py
git add d_contrastive/experiment_utils.py
git add d_contrastive/run_w0_phase0_dvalue.py
git add d_contrastive/run_w0_phase1_diagnosis.py
git add d_contrastive/run_w0_conversion_3way.py
git add d_contrastive/run_w2_phase1_dsupport.py
git add d_contrastive/train_w2_phase2_swin.py
git add d_contrastive/train_w2_direct_lora_downstream.py
git add d_contrastive/run_w2_phase1_budget20_eff128.sbatch
git add d_contrastive/run_w2_phase1_conv_budget20_eff128.sbatch
git add d_contrastive/run_w2_budget20_eff128.sbatch
git add d_contrastive/run_w2_phase2_full_retry.sbatch
```

## Core Results

Final raw-only fair20 tables:

```bash
git add d_contrastive/results_w2_fair20_raw_final
```

This directory contains:

- `raw_phase0_d_and_dhat.csv`
- `raw_dsupport.csv`
- `raw_diagnosis_all_methods_fair20.csv`
- `raw_diagnosis_all_methods_fair20_pivot.csv`
- `raw_conversion_all_methods_fair20.csv`
- `raw_conversion_all_methods_fair20_pivot.csv`

Detailed run directories such as `results_w2_budget20_eff128/` and
`results_w2_phase1_budget20_eff128/` are archived under `d_contrastive/log/`
because they contain intermediate outputs and adapted embeddings.

## Core Data

Compact frozen Swin features and matched metadata:

```bash
git add data/embeddings_128_05152016
git add data/master_smri_05152016
```

## Do Not Add

These are ignored and should not be committed:

- `d_contrastive/log/`
- `d_contrastive/results*/` except `results_w2_fair20_raw_final/`
- `d_contrastive/__pycache__/`
- checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- scheduler logs: `*.out`, `*.err`, `slurm-*.out`, `logs/`
- raw DICOM/NIfTI preprocessing outputs
- broad `data/**` outside the explicitly unignored data directories

## Sanity Check

```bash
python -m py_compile \
  d_contrastive/minimal_v0_contrastive.py \
  d_contrastive/experiment_utils.py \
  d_contrastive/run_w0_phase0_dvalue.py \
  d_contrastive/run_w0_phase1_diagnosis.py \
  d_contrastive/run_w0_conversion_3way.py \
  d_contrastive/run_w2_phase1_dsupport.py \
  d_contrastive/train_w2_phase2_swin.py \
  d_contrastive/train_w2_direct_lora_downstream.py
rm -rf d_contrastive/__pycache__
```

## Suggested Commit

```bash
git status --short
git add .gitignore PACKAGE_MANIFEST.md d_contrastive data/embeddings_128_05152016 data/master_smri_05152016
git status --short
git commit -m "Package ADNI W2 d-contrastive fair20 experiments"
```

Inspect `git status --short` before committing. It should not include log
archives, old result directories, checkpoints, or raw imaging files.
