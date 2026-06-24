#!/usr/bin/env bash
set -euo pipefail

# Stage only the core ADNI frozen-Swin contrastive package files.
# This intentionally excludes logs, historical results directories, checkpoints,
# caches, raw imaging outputs, and old download-target files.

git add .gitignore PACKAGE_MANIFEST.md git_add_core_files.sh

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

git add data/embeddings_128_05152016
git add data/master_smri_05152016

git status --short
