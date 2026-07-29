# Old split ridge d_hat results

Pulled from existing old train/test split. Conversion rows come from `exp4_conversion_task_suite.csv`; current-dx rows were recomputed as Ridge(image -> d_mod3) on old train baseline, evaluated on old test baseline.

## Current dx / d_mod3 metrics

| version   | metric                  |   n |   auc_or_value |   n_pos |
|:----------|:------------------------|----:|---------------:|--------:|
| raw       | d_mod3_R2               | 239 |          0.077 |     nan |
| raw       | d_mod3_pearson          | 239 |          0.346 |     nan |
| raw       | d_mod3_spearman         | 239 |          0.325 |     nan |
| raw       | NORMAL_vs_AD_AUC        | 125 |          0.745 |      31 |
| raw       | NORMAL_vs_MCI_AUC       | 208 |          0.495 |     114 |
| raw       | MCI_vs_AD_AUC           | 145 |          0.724 |      31 |
| raw       | 3class_macro_direct_MCI | 239 |          0.574 |     nan |
| raw       | 3class_macro_middle_MCI | 239 |          0.576 |     nan |
| raw       | OvR_NORMAL              | 239 |          0.549 |     nan |
| raw       | OvR_MCI_direct          | 239 |          0.441 |     nan |
| raw       | OvR_MCI_middle          | 239 |          0.447 |     nan |
| raw       | OvR_AD                  | 239 |          0.733 |     nan |
| combat    | d_mod3_R2               | 239 |          0.082 |     nan |
| combat    | d_mod3_pearson          | 239 |          0.369 |     nan |
| combat    | d_mod3_spearman         | 239 |          0.359 |     nan |
| combat    | NORMAL_vs_AD_AUC        | 125 |          0.771 |      31 |
| combat    | NORMAL_vs_MCI_AUC       | 208 |          0.508 |     114 |
| combat    | MCI_vs_AD_AUC           | 145 |          0.723 |      31 |
| combat    | 3class_macro_direct_MCI | 239 |          0.587 |     nan |
| combat    | 3class_macro_middle_MCI | 239 |          0.578 |     nan |
| combat    | OvR_NORMAL              | 239 |          0.565 |     nan |
| combat    | OvR_MCI_direct          | 239 |          0.451 |     nan |
| combat    | OvR_MCI_middle          | 239 |          0.424 |     nan |
| combat    | OvR_AD                  | 239 |          0.745 |     nan |

## Conversion ridge_d_hat

| input_version   | task                         |   horizon_years |   n |   n_converters |     auc |   ci_lo |   ci_hi |
|:----------------|:-----------------------------|----------------:|----:|---------------:|--------:|--------:|--------:|
| raw             | MCI_to_AD_1y                 |               1 |  85 |              4 |   0.843 |   0.729 |   0.941 |
| raw             | MCI_to_AD_2y                 |               2 |  65 |             14 |   0.704 |   0.54  |   0.846 |
| raw             | MCI_to_AD_3y                 |               3 |  51 |             19 |   0.778 |   0.629 |   0.901 |
| raw             | MCI_to_AD_4y                 |               4 |  45 |             22 |   0.731 |   0.565 |   0.875 |
| raw             | CN_to_MCI_2y                 |               2 |  59 |              5 |   0.767 |   0.474 |   0.956 |
| raw             | CN_to_MCI_3y                 |               3 |  49 |              9 |   0.703 |   0.5   |   0.885 |
| raw             | CN_to_MCI_4y                 |               4 |  42 |              9 |   0.687 |   0.475 |   0.871 |
| raw             | MCI_time_to_AD_4y_converters |               4 |  22 |             22 | nan     | nan     | nan     |
| combat          | MCI_to_AD_1y                 |               1 |  85 |              4 |   0.827 |   0.731 |   0.91  |
| combat          | MCI_to_AD_2y                 |               2 |  65 |             14 |   0.772 |   0.619 |   0.898 |
| combat          | MCI_to_AD_3y                 |               3 |  51 |             19 |   0.811 |   0.672 |   0.929 |
| combat          | MCI_to_AD_4y                 |               4 |  45 |             22 |   0.763 |   0.609 |   0.896 |
| combat          | CN_to_MCI_2y                 |               2 |  59 |              5 |   0.844 |   0.696 |   0.955 |
| combat          | CN_to_MCI_3y                 |               3 |  49 |              9 |   0.719 |   0.548 |   0.874 |
| combat          | CN_to_MCI_4y                 |               4 |  42 |              9 |   0.694 |   0.503 |   0.855 |
| combat          | MCI_time_to_AD_4y_converters |               4 |  22 |             22 | nan     | nan     | nan     |

## Files

- `d_contrastive/results_old_split_summary/old_split_ridge_dhat_current_dx.csv`
- `d_contrastive/results_old_split_summary/old_split_ridge_dhat_conversion.csv`
- `d_contrastive/results_old_split_summary/old_split_ridge_dhat_test_scores.csv`
