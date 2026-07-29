# Old train/test split results pulled from existing files

These are the previous results before the strict 3-way contrastive/finetune/test split. No experiment was rerun.

## Main baseline / rank table

| method                          | input_version   |   cn_ad_auc |   cn_mci_cls_auc |   mci_ad_cls_auc |   3class_macro_auc |   mci_conv_auc |   cn_mci_conv_auc |   r2_d_mod3 |   pearson_d_mod3 |   spearman_d_mod3 |   z_std_test |   z_active_dims_005 |
|:--------------------------------|:----------------|------------:|-----------------:|-----------------:|-------------------:|---------------:|------------------:|------------:|-----------------:|------------------:|-------------:|--------------------:|
| Pure ML direct LR/Ridge on 768  | combat          |       0.847 |            0.579 |            0.761 |              0.661 |          0.684 |             0.29  |       0.148 |            0.445 |             0.438 |      nan     |                 nan |
| Pure ML direct LR/Ridge on 768  | raw             |       0.86  |            0.577 |            0.762 |              0.657 |          0.644 |             0.317 |       0.237 |            0.512 |             0.467 |      nan     |                 nan |
| Pure supervised MLP 768-384-256 | combat          |       0.719 |            0.566 |            0.709 |              0.641 |          0.599 |             0.202 |       0.038 |            0.336 |             0.331 |      nan     |                 nan |
| Pure supervised MLP 768-384-256 | raw             |       0.752 |            0.562 |            0.693 |              0.61  |          0.626 |             0.24  |       0.057 |            0.377 |             0.338 |      nan     |                 nan |
| Euclidean contrastive + linear  | combat          |       0.819 |            0.602 |            0.739 |              0.656 |          0.718 |             0.602 |       0.121 |            0.437 |             0.417 |        0.024 |                  16 |
| Euclidean contrastive + linear  | raw             |       0.793 |            0.594 |            0.711 |              0.635 |          0.769 |             0.674 |       0.149 |            0.456 |             0.429 |        0.03  |                  25 |
| Hybrid rank+euclidean + linear  | combat          |       0.832 |            0.609 |            0.751 |              0.66  |          0.754 |             0.697 |       0.106 |            0.441 |             0.423 |        0.026 |                  16 |
| Hybrid rank+euclidean + linear  | raw             |       0.793 |            0.572 |            0.739 |              0.633 |          0.754 |             0.676 |       0.106 |            0.433 |             0.412 |        0.027 |                  12 |
| Rank-only Kendall + linear      | combat          |       0.827 |            0.594 |            0.774 |              0.659 |          0.726 |             0.613 |       0.077 |            0.417 |             0.396 |        0.995 |                 256 |
| Rank-only Kendall + linear      | raw             |       0.808 |            0.597 |            0.757 |              0.66  |          0.706 |             0.501 |       0.094 |            0.423 |             0.393 |        0.94  |                 256 |
| Hybrid direct 1D rank head s    | combat          |     nan     |          nan     |          nan     |            nan     |          0.736 |             0.658 |     nan     |            0.401 |             0.385 |      nan     |                 nan |
| Hybrid direct 1D rank head s    | raw             |     nan     |          nan     |          nan     |            nan     |          0.752 |             0.645 |     nan     |            0.431 |             0.409 |      nan     |                 nan |
| Rank-only direct 1D rank head s | combat          |     nan     |          nan     |          nan     |            nan     |          0.746 |             0.67  |     nan     |            0.339 |             0.375 |      nan     |                 nan |
| Rank-only direct 1D rank head s | raw             |     nan     |          nan     |          nan     |            nan     |          0.679 |             0.566 |     nan     |            0.35  |             0.341 |      nan     |                 nan |

## W0 setup3-only old split

| input_version   | loss_mode          |   cn_ad_auc |   cn_mci_cls_auc |   mci_ad_cls_auc |   3class_macro_auc |   mci_conv_auc |   cn_mci_conv_auc |   r2_d_mod3 |   pearson_d_mod3 |   spearman_d_mod3 |   z_std_test |   z_active_dims_005 |
|:----------------|:-------------------|------------:|-----------------:|-----------------:|-------------------:|---------------:|------------------:|------------:|-----------------:|------------------:|-------------:|--------------------:|
| raw             | euclidean          |       0.805 |            0.592 |            0.735 |              0.646 |          0.741 |             0.67  |       0.134 |            0.452 |             0.423 |        0.026 |                  17 |
| raw             | rank_kendall       |       0.802 |            0.599 |            0.748 |              0.655 |          0.696 |             0.443 |       0.103 |            0.429 |             0.396 |        1.035 |                 256 |
| raw             | rank_kendall_basic |       0.8   |            0.605 |            0.757 |              0.659 |          0.684 |             0.47  |       0.062 |            0.409 |             0.378 |        1.071 |                 256 |
| raw             | hybrid             |       0.804 |            0.585 |            0.733 |              0.641 |          0.774 |             0.656 |       0.106 |            0.434 |             0.407 |        0.027 |                  14 |
| raw             | hybrid_basic       |       0.8   |            0.577 |            0.735 |              0.634 |          0.725 |             0.661 |       0.095 |            0.427 |             0.4   |        0.025 |                  16 |
| combat          | euclidean          |       0.83  |            0.621 |            0.739 |              0.669 |          0.739 |             0.62  |       0.14  |            0.452 |             0.44  |        0.024 |                  18 |
| combat          | rank_kendall       |       0.825 |            0.593 |            0.768 |              0.663 |          0.699 |             0.479 |       0.074 |            0.413 |             0.392 |        0.956 |                 256 |
| combat          | rank_kendall_basic |       0.816 |            0.595 |            0.762 |              0.659 |          0.688 |             0.508 |       0.062 |            0.424 |             0.389 |        1.098 |                 256 |
| combat          | hybrid             |       0.799 |            0.599 |            0.724 |              0.644 |          0.744 |             0.724 |       0.072 |            0.411 |             0.401 |        0.027 |                  16 |
| combat          | hybrid_basic       |       0.829 |            0.612 |            0.743 |              0.656 |          0.777 |             0.685 |       0.12  |            0.455 |             0.438 |        0.026 |                  23 |

## W0 all-year conversion mean over available horizons

| input_version   | task      | score_type      | loss_mode          | horizons   |   n_test_sum |   n_converters_sum |   auc_mean |   auc_weighted_by_n |   auc_weighted_by_converters |
|:----------------|:----------|:----------------|:-------------------|:-----------|-------------:|-------------------:|-----------:|--------------------:|-----------------------------:|
| combat          | CN_to_MCI | setup3_s_direct | hybrid             | 2/3/4      |          282 |                 23 |      0.679 |               0.679 |                        0.67  |
| combat          | CN_to_MCI | setup3_probe    | hybrid             | 2/3/4      |          282 |                 23 |      0.666 |               0.666 |                        0.657 |
| combat          | CN_to_MCI | setup3_s_direct | hybrid_basic       | 2/3/4      |          282 |                 23 |      0.648 |               0.648 |                        0.639 |
| combat          | CN_to_MCI | setup3_probe    | hybrid_basic       | 2/3/4      |          282 |                 23 |      0.647 |               0.647 |                        0.64  |
| combat          | CN_to_MCI | setup3_s_direct | rank_kendall       | 2/3/4      |          282 |                 23 |      0.619 |               0.619 |                        0.614 |
| combat          | CN_to_MCI | setup3_probe    | euclidean          | 2/3/4      |          282 |                 23 |      0.6   |               0.6   |                        0.596 |
| combat          | CN_to_MCI | setup3_s_direct | rank_kendall_basic | 2/3/4      |          282 |                 23 |      0.58  |               0.58  |                        0.579 |
| combat          | CN_to_MCI | setup3_probe    | rank_kendall_basic | 2/3/4      |          282 |                 23 |      0.49  |               0.49  |                        0.487 |
| combat          | CN_to_MCI | setup3_probe    | rank_kendall       | 2/3/4      |          282 |                 23 |      0.485 |               0.485 |                        0.486 |
| combat          | CN_to_MCI | setup3_s_direct | euclidean          | 2/3/4      |          282 |                 23 |      0.333 |               0.333 |                        0.346 |
| raw             | CN_to_MCI | setup3_probe    | euclidean          | 2/3/4      |          282 |                 23 |      0.599 |               0.599 |                        0.587 |
| raw             | CN_to_MCI | setup3_probe    | hybrid_basic       | 2/3/4      |          282 |                 23 |      0.595 |               0.595 |                        0.583 |
| raw             | CN_to_MCI | setup3_s_direct | hybrid             | 2/3/4      |          282 |                 23 |      0.593 |               0.593 |                        0.584 |
| raw             | CN_to_MCI | setup3_s_direct | rank_kendall       | 2/3/4      |          282 |                 23 |      0.59  |               0.59  |                        0.586 |
| raw             | CN_to_MCI | setup3_s_direct | hybrid_basic       | 2/3/4      |          282 |                 23 |      0.589 |               0.589 |                        0.582 |
| raw             | CN_to_MCI | setup3_probe    | hybrid             | 2/3/4      |          282 |                 23 |      0.589 |               0.589 |                        0.577 |
| raw             | CN_to_MCI | setup3_s_direct | rank_kendall_basic | 2/3/4      |          282 |                 23 |      0.557 |               0.557 |                        0.554 |
| raw             | CN_to_MCI | setup3_probe    | rank_kendall_basic | 2/3/4      |          282 |                 23 |      0.486 |               0.486 |                        0.489 |
| raw             | CN_to_MCI | setup3_s_direct | euclidean          | 2/3/4      |          282 |                 23 |      0.475 |               0.475 |                        0.471 |
| raw             | CN_to_MCI | setup3_probe    | rank_kendall       | 2/3/4      |          282 |                 23 |      0.46  |               0.46  |                        0.462 |
| combat          | MCI_to_AD | setup3_s_direct | hybrid_basic       | 1/2/3/4    |          456 |                 59 |      0.811 |               0.811 |                        0.799 |
| combat          | MCI_to_AD | setup3_probe    | hybrid_basic       | 1/2/3/4    |          456 |                 59 |      0.806 |               0.806 |                        0.798 |
| combat          | MCI_to_AD | setup3_probe    | hybrid             | 1/2/3/4    |          456 |                 59 |      0.786 |               0.786 |                        0.783 |
| combat          | MCI_to_AD | setup3_s_direct | hybrid             | 1/2/3/4    |          456 |                 59 |      0.786 |               0.786 |                        0.775 |
| combat          | MCI_to_AD | setup3_probe    | euclidean          | 1/2/3/4    |          456 |                 59 |      0.778 |               0.778 |                        0.782 |
| combat          | MCI_to_AD | setup3_probe    | rank_kendall_basic | 1/2/3/4    |          456 |                 59 |      0.76  |               0.76  |                        0.735 |
| combat          | MCI_to_AD | setup3_probe    | rank_kendall       | 1/2/3/4    |          456 |                 59 |      0.759 |               0.759 |                        0.736 |
| combat          | MCI_to_AD | setup3_s_direct | rank_kendall_basic | 1/2/3/4    |          456 |                 59 |      0.75  |               0.75  |                        0.748 |
| combat          | MCI_to_AD | setup3_s_direct | rank_kendall       | 1/2/3/4    |          456 |                 59 |      0.737 |               0.737 |                        0.735 |
| combat          | MCI_to_AD | setup3_s_direct | euclidean          | 1/2/3/4    |          456 |                 59 |      0.208 |               0.208 |                        0.209 |
| raw             | MCI_to_AD | setup3_s_direct | hybrid             | 1/2/3/4    |          456 |                 59 |      0.803 |               0.803 |                        0.795 |
| raw             | MCI_to_AD | setup3_probe    | hybrid             | 1/2/3/4    |          456 |                 59 |      0.802 |               0.802 |                        0.801 |
| raw             | MCI_to_AD | setup3_probe    | hybrid_basic       | 1/2/3/4    |          456 |                 59 |      0.783 |               0.783 |                        0.778 |
| raw             | MCI_to_AD | setup3_probe    | euclidean          | 1/2/3/4    |          456 |                 59 |      0.781 |               0.781 |                        0.794 |
| raw             | MCI_to_AD | setup3_probe    | rank_kendall       | 1/2/3/4    |          456 |                 59 |      0.774 |               0.774 |                        0.747 |
| raw             | MCI_to_AD | setup3_s_direct | hybrid_basic       | 1/2/3/4    |          456 |                 59 |      0.767 |               0.767 |                        0.751 |
| raw             | MCI_to_AD | setup3_probe    | rank_kendall_basic | 1/2/3/4    |          456 |                 59 |      0.76  |               0.76  |                        0.737 |
| raw             | MCI_to_AD | setup3_s_direct | rank_kendall       | 1/2/3/4    |          456 |                 59 |      0.72  |               0.72  |                        0.721 |
| raw             | MCI_to_AD | setup3_s_direct | rank_kendall_basic | 1/2/3/4    |          456 |                 59 |      0.716 |               0.716 |                        0.717 |
| raw             | MCI_to_AD | setup3_s_direct | euclidean          | 1/2/3/4    |          456 |                 59 |      0.703 |               0.703 |                        0.728 |

## Files

- `d_contrastive/results_old_split_summary/old_split_exp4_conversion_task_suite.csv`
- `d_contrastive/results_old_split_summary/old_split_main_baselines.csv`
- `d_contrastive/results_old_split_summary/old_split_w0_all_years_summary.csv`
- `d_contrastive/results_old_split_summary/old_split_w0_conversion_by_year.csv`
- `d_contrastive/results_old_split_summary/old_split_w0_setup3_summary.csv`
