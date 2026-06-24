# ADNI d_contrastive Package Manifest

This directory has been trimmed conceptually for a git package. The core code,
compact frozen embeddings, matched metadata, and final summary tables are kept.
Historical logs, intermediate run directories, checkpoints, caches, and old
download-target files are ignored by `.gitignore`.

## Core Code

Add these files:

```bash
git add .gitignore PACKAGE_MANIFEST.md
git add d_contrastive/README.md
git add d_contrastive/minimal_v0_contrastive.py
git add d_contrastive/experiment_utils.py
git add d_contrastive/run_exp2_exp0.py
git add d_contrastive/run_exp1_dx_pretrain.py
git add d_contrastive/run_exp1_dx_pretrain.sbatch
git add d_contrastive/run_exp4_conversion_suite.py
git add d_contrastive/consolidate_rank_results.py
git add d_contrastive/make_new_results_summary.py
git add d_contrastive/smoke_test_package.py
```

Optional helper scripts for image download planning:

```bash
git add d_contrastive/build_final_image_id_download_list.py
git add d_contrastive/build_v3_final_and_coverage.py
git add d_contrastive/check_projected_longitudinal_gain.py
git add d_contrastive/diagnose_cn_mci_conversion.py
```

## Core Result Tables

```bash
git add d_contrastive/AD_contrastive_new_results_summary_20260605.xlsx
git add d_contrastive/AD_contrastive_key_results_20260605.csv
git add d_contrastive/AD_contrastive_new_results_summary_20260605.md
git add d_contrastive/rank_sweep_with_ml_baselines.csv
git add d_contrastive/rank_sweep_best_comparison.csv
git add d_contrastive/rank_sweep_conversion_ci.csv
git add d_contrastive/rank_sweep_all_results.csv
git add d_contrastive/exp0_dhat_vs_contrastive_conversion.csv
git add d_contrastive/exp1_method2_dx_pretrain.csv
git add d_contrastive/exp2_censoring_report.csv
git add d_contrastive/exp4_conversion_task_suite.csv
git add d_contrastive/exp4_conversion_cohort_counts.csv
git add d_contrastive/conversion_cohorts_censored.csv
```

## Core Data

Compact frozen Swin features and matched metadata:

```bash
git add data/embeddings_128_05152016
git add data/master_smri_05152016
```

Sizes at cleanup time:

- `data/embeddings_128_05152016`: about 34 MB
- `data/master_smri_05152016`: about 5 MB

## Do Not Add

These are ignored and should not be committed:

- `d_contrastive/logs/`
- `d_contrastive/results*/`
- `d_contrastive/__pycache__/`
- `d_contrastive/download_targets_20260515/`
- `d_contrastive/submit_*.sh`
- `*.pt`, `*.pth`, `*.ckpt`
- raw DICOM/NIfTI preprocessing outputs
- broad `data/**` outside the two explicitly kept data directories

## Smoke Test

Run before committing:

```bash
source /users/szhang1/fsl/bin/activate optuna_env
python d_contrastive/smoke_test_package.py
```

Expected final line:

```text
SMOKE TEST PASSED
```

## Suggested Commit

After running `git init` in a normal shell if needed:

```bash
git status --short
git add .gitignore PACKAGE_MANIFEST.md d_contrastive data/embeddings_128_05152016 data/master_smri_05152016
git status --short
git commit -m "Package ADNI frozen Swin contrastive experiments"
```

Inspect `git status --short` carefully before commit. It should not include
logs, historical `results*/` directories, checkpoints, or raw imaging files.
