# ADNI d_contrastive plan results, 2026-06-24

## What ran

- W0 rank ablation: Setup3-only contrastive/probe on frozen Swin 768-d embeddings, 150 epochs, raw and ComBat.
- W1 RASPER: quick fixed-parameter conversion test, PCA=16, lambda=3.1623, alpha=10, no bootstrap CI.
- Full Swin encoder was not retrained. Existing LR/MLP baselines from the package remain the comparison baseline.

## W0 main Setup3 probe table

| input_version   | loss_mode          |   cn_ad_auc |   cn_mci_cls_auc |   mci_ad_cls_auc |   3class_macro_auc |   mci_conv_auc |   cn_mci_conv_auc |   r2_d_mod3 |   pearson_d_mod3 |   spearman_d_mod3 |   alignment_spearman |   z_std_test |   z_active_dims_005 |
|:----------------|:-------------------|------------:|-----------------:|-----------------:|-------------------:|---------------:|------------------:|------------:|-----------------:|------------------:|---------------------:|-------------:|--------------------:|
| raw             | euclidean          |       0.805 |            0.592 |            0.735 |              0.646 |          0.741 |             0.67  |       0.134 |            0.452 |             0.423 |                0.19  |        0.026 |                  17 |
| raw             | rank_kendall       |       0.802 |            0.599 |            0.748 |              0.655 |          0.696 |             0.443 |       0.103 |            0.429 |             0.396 |                0.065 |        1.035 |                 256 |
| raw             | rank_kendall_basic |       0.8   |            0.605 |            0.757 |              0.659 |          0.684 |             0.47  |       0.062 |            0.409 |             0.378 |                0.084 |        1.071 |                 256 |
| raw             | hybrid             |       0.804 |            0.585 |            0.733 |              0.641 |          0.774 |             0.656 |       0.106 |            0.434 |             0.407 |                0.188 |        0.027 |                  14 |
| raw             | hybrid_basic       |       0.8   |            0.577 |            0.735 |              0.634 |          0.725 |             0.661 |       0.095 |            0.427 |             0.4   |                0.17  |        0.025 |                  16 |
| combat          | euclidean          |       0.83  |            0.621 |            0.739 |              0.669 |          0.739 |             0.62  |       0.14  |            0.452 |             0.44  |                0.197 |        0.024 |                  18 |
| combat          | rank_kendall       |       0.825 |            0.593 |            0.768 |              0.663 |          0.699 |             0.479 |       0.074 |            0.413 |             0.392 |                0.073 |        0.956 |                 256 |
| combat          | rank_kendall_basic |       0.816 |            0.595 |            0.762 |              0.659 |          0.688 |             0.508 |       0.062 |            0.424 |             0.389 |                0.101 |        1.098 |                 256 |
| combat          | hybrid             |       0.799 |            0.599 |            0.724 |              0.644 |          0.744 |             0.724 |       0.072 |            0.411 |             0.401 |                0.15  |        0.027 |                  16 |
| combat          | hybrid_basic       |       0.829 |            0.612 |            0.743 |              0.656 |          0.777 |             0.685 |       0.12  |            0.455 |             0.438 |                0.193 |        0.026 |                  23 |

## W0 Setup3-s direct rank/progression score

| input_version   | loss_mode          |   mci_conv_auc |   cn_mci_conv_auc |   d_pearson_s |   d_spearman_s |
|:----------------|:-------------------|---------------:|------------------:|--------------:|---------------:|
| raw             | euclidean          |          0.661 |             0.503 |         0.389 |          0.352 |
| raw             | rank_kendall       |          0.676 |             0.616 |         0.273 |          0.335 |
| raw             | rank_kendall_basic |          0.681 |             0.575 |         0.335 |          0.326 |
| raw             | hybrid             |          0.77  |             0.645 |         0.426 |          0.399 |
| raw             | hybrid_basic       |          0.705 |             0.634 |         0.391 |          0.362 |
| combat          | euclidean          |          0.235 |             0.263 |        -0.461 |         -0.437 |
| combat          | rank_kendall       |          0.704 |             0.647 |         0.287 |          0.332 |
| combat          | rank_kendall_basic |          0.717 |             0.587 |         0.315 |          0.346 |
| combat          | hybrid             |          0.744 |             0.733 |         0.391 |          0.381 |
| combat          | hybrid_basic       |          0.778 |             0.699 |         0.447 |          0.428 |

## W0 winners

- mci_conv_auc: combat/hybrid_basic = 0.777
- cn_mci_conv_auc: combat/hybrid = 0.724
- r2_d_mod3: combat/euclidean = 0.140
- cn_ad_auc: combat/euclidean = 0.830
- 3class_macro_auc: combat/euclidean = 0.669

## W1 RASPER quick conversion table

| version   | task      |   horizon_years |   n |   n_converters |   direct_logistic |   rasper_kendall |   ridge_d_hat |
|:----------|:----------|----------------:|----:|---------------:|------------------:|-----------------:|--------------:|
| combat    | CN_to_MCI |               2 |  59 |              5 |             0.322 |            0.385 |         0.811 |
| combat    | CN_to_MCI |               3 |  49 |              9 |             0.406 |            0.411 |         0.656 |
| combat    | CN_to_MCI |               4 |  42 |              9 |             0.451 |            0.411 |         0.646 |
| combat    | MCI_to_AD |               2 |  65 |             14 |             0.559 |            0.574 |         0.78  |
| combat    | MCI_to_AD |               3 |  51 |             19 |             0.701 |            0.684 |         0.826 |
| combat    | MCI_to_AD |               4 |  45 |             22 |             0.64  |            0.65  |         0.761 |
| raw       | CN_to_MCI |               2 |  59 |              5 |             0.307 |            0.419 |         0.789 |
| raw       | CN_to_MCI |               3 |  49 |              9 |             0.344 |            0.347 |         0.65  |
| raw       | CN_to_MCI |               4 |  42 |              9 |             0.387 |            0.465 |         0.65  |
| raw       | MCI_to_AD |               2 |  65 |             14 |             0.529 |            0.543 |         0.756 |
| raw       | MCI_to_AD |               3 |  51 |             19 |             0.676 |            0.678 |         0.817 |
| raw       | MCI_to_AD |               4 |  45 |             22 |             0.642 |            0.642 |         0.765 |

## Readout

- Plain Euclidean remains best for continuous d_mod3 regression among W0 contrastive variants: raw R2=0.134, combat R2=0.140.
- Rank-only variants prevent latent collapse (z.std about 1.0, 256 active dims) but are weaker for conversion and d_mod3 regression than Euclidean/hybrid.
- Hybrid variants are strongest for conversion. Best MCI->AD is combat/hybrid_basic AUC=0.777, essentially tied with raw/hybrid AUC=0.774. Best CN->MCI is combat/hybrid AUC=0.724 by probe, and 0.733 using Setup3-s direct score.
- RASPER quick test does not beat ridge_d_hat. For every reported task/horizon, ridge_d_hat AUC is higher than rasper_kendall. This argues against the external ranking objective as implemented here, at least with fixed lambda/alpha.
- The best direct image baseline from prior package remains strong: raw direct Ridge/LR had d_mod3 R2=0.237 and CN/AD AUC=0.860. Contrastive is most useful for conversion signal, not for replacing direct d_mod3 Ridge.

## Output files

- W0 all rows: `d_contrastive/results_w0_setup3_only/w0_setup3_only_all_results.csv`
- W0 summary rows: `d_contrastive/results_w0_setup3_only/w0_setup3_only_summary.csv`
- W0 plots: `d_contrastive/results_w0_setup3_only/mode_*/setup3_*_{raw,combat}.png`
- W1 RASPER CSV: `d_contrastive/results_w1_rasper_quick_l3p16_a10/w1_rasper_conversion.csv`
- W1 RASPER markdown: `d_contrastive/results_w1_rasper_quick_l3p16_a10/w1_rasper_conversion.md`
