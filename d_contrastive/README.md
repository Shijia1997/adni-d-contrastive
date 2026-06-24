# ADNI d_mod3 Contrastive Experiments

This package evaluates whether frozen 3D Swin MRI embeddings contain AD
progression signal and whether d_mod3-supervised representation learning helps
downstream diagnosis/conversion tasks.

## What Is Kept

Core code:

- `minimal_v0_contrastive.py`: main frozen-embedding experiment runner.
- `experiment_utils.py`: shared loading, metrics, censoring, bootstrap helpers.
- `run_exp2_exp0.py`: follow-up censoring and d-hat conversion baseline.
- `run_exp1_dx_pretrain.py`: diagnosis-pretraining baseline.
- `run_exp4_conversion_suite.py`: conversion-window evaluation.
- `consolidate_rank_results.py`: rank/euclidean/hybrid result consolidation.
- `make_new_results_summary.py`: writes the final Excel/CSV/Markdown summary.

Core data:

- `../data/embeddings_128_05152016/`: frozen Swin 128^3 raw/combat embeddings.
- `../data/master_smri_05152016/`: matched metadata and train/test split.

Core outputs:

- `AD_contrastive_new_results_summary_20260605.xlsx`
- `AD_contrastive_key_results_20260605.csv`
- `AD_contrastive_new_results_summary_20260605.md`
- `rank_sweep_with_ml_baselines.csv`
- `rank_sweep_best_comparison.csv`
- `exp0_dhat_vs_contrastive_conversion.csv`
- `exp1_method2_dx_pretrain.csv`
- `exp4_conversion_task_suite.csv`
- `exp4_conversion_cohort_counts.csv`

Historical `results*/`, `logs/`, `__pycache__/`, checkpoints, and temporary
download-target files are intentionally ignored by git.

## Environment

On JHPCE, the scripts were run in:

```bash
source /users/szhang1/fsl/bin/activate optuna_env
```

Required Python packages include `numpy`, `pandas`, `scipy`, `scikit-learn`,
and `torch`.

## Smoke Test

From `/dcs07/zwang/data/adni_d`:

```bash
source /users/szhang1/fsl/bin/activate optuna_env
python d_contrastive/smoke_test_package.py
```

The smoke test checks that the frozen embedding files and metadata can be
loaded and that the core modules import.

## Recreate Summary Workbook

```bash
source /users/szhang1/fsl/bin/activate optuna_env
python d_contrastive/make_new_results_summary.py
```

## Main Result Read

- Direct LR/Ridge on raw 768-d Swin embeddings is strongest for current
  diagnosis and d_mod3 regression.
- Euclidean/hybrid contrastive learning helps some conversion tasks, but the
  cohorts are small, especially CN->MCI.
- Ridge d-hat explains much of the conversion gain, so the safest claim is
  dense progression supervision rather than contrastive-specific geometry.
- Diagnosis pretraining is close to contrastive pretraining, so contrastive is
  not uniquely superior in the current frozen-feature setting.
