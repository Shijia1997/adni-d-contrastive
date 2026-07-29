# Workstream 1: RASPER conversion results

Headline contrast: `rasper_*` (borrow ranking) vs `ridge_d_hat` (borrow score). `delta_*` columns are paired bootstrap AUC differences vs `ridge_d_hat`.

| version   | task      |   horizon_years | method             |   n |   n_converters |    auc |   ci_lo |   ci_hi |   delta_vs_ridge_dhat |   delta_ci_lo |   delta_ci_hi |   delta_p |
|:----------|:----------|----------------:|:-------------------|----:|---------------:|-------:|--------:|--------:|----------------------:|--------------:|--------------:|----------:|
| raw       | MCI_to_AD |               2 | ridge_d_hat        |  65 |             14 | 0.7563 |  0.6064 |  0.8853 |              nan      |      nan      |      nan      |  nan      |
| raw       | MCI_to_AD |               2 | direct_logistic    |  65 |             14 | 0.5294 |  0.3619 |  0.6919 |               -0.2269 |       -0.3691 |       -0.0808 |    0.0032 |
| raw       | MCI_to_AD |               2 | oracle_true_d_mod3 |  65 |             14 | 0.6625 |  0.4988 |  0.8109 |               -0.094  |       -0.2186 |        0.0254 |    0.1144 |
| raw       | MCI_to_AD |               2 | rasper_kendall     |  65 |             14 | 0.5434 |  0.3698 |  0.7101 |               -0.2131 |       -0.3603 |       -0.064  |    0.0072 |
| raw       | MCI_to_AD |               3 | ridge_d_hat        |  51 |             19 | 0.8174 |  0.6855 |  0.9277 |              nan      |      nan      |      nan      |  nan      |
| raw       | MCI_to_AD |               3 | direct_logistic    |  51 |             19 | 0.676  |  0.5148 |  0.8204 |               -0.1427 |       -0.2841 |       -0.0048 |    0.0428 |
| raw       | MCI_to_AD |               3 | oracle_true_d_mod3 |  51 |             19 | 0.6859 |  0.5236 |  0.8398 |               -0.1323 |       -0.2813 |       -0.0023 |    0.0432 |
| raw       | MCI_to_AD |               3 | rasper_kendall     |  51 |             19 | 0.6562 |  0.4892 |  0.8048 |               -0.162  |       -0.3032 |       -0.0309 |    0.0152 |
| raw       | MCI_to_AD |               4 | ridge_d_hat        |  45 |             22 | 0.7648 |  0.6073 |  0.8968 |              nan      |      nan      |      nan      |  nan      |
| raw       | MCI_to_AD |               4 | direct_logistic    |  45 |             22 | 0.6423 |  0.4685 |  0.806  |               -0.1218 |       -0.282  |        0.0278 |    0.1168 |
| raw       | MCI_to_AD |               4 | oracle_true_d_mod3 |  45 |             22 | 0.664  |  0.486  |  0.816  |               -0.1008 |       -0.2648 |        0.058  |    0.2172 |
| raw       | MCI_to_AD |               4 | rasper_kendall     |  45 |             22 | 0.6423 |  0.4732 |  0.8115 |               -0.1217 |       -0.2767 |        0.0243 |    0.1164 |
| raw       | CN_to_MCI |               2 | ridge_d_hat        |  59 |              5 | 0.7889 |  0.5789 |  0.947  |              nan      |      nan      |      nan      |  nan      |
| raw       | CN_to_MCI |               2 | direct_logistic    |  59 |              5 | 0.3074 |  0.1488 |  0.4779 |               -0.4817 |       -0.7183 |       -0.1905 |    0.0072 |
| raw       | CN_to_MCI |               2 | oracle_true_d_mod3 |  59 |              5 | 0.6519 |  0.4407 |  0.8481 |               -0.1374 |       -0.4325 |        0.1956 |    0.3701 |
| raw       | CN_to_MCI |               2 | rasper_kendall     |  59 |              5 | 0.3296 |  0.1682 |  0.5    |               -0.4599 |       -0.7107 |       -0.1461 |    0.012  |
| raw       | CN_to_MCI |               3 | ridge_d_hat        |  49 |              9 | 0.65   |  0.4679 |  0.8278 |              nan      |      nan      |      nan      |  nan      |
| raw       | CN_to_MCI |               3 | direct_logistic    |  49 |              9 | 0.3444 |  0.1738 |  0.5374 |               -0.3063 |       -0.603  |       -0.006  |    0.0484 |
| raw       | CN_to_MCI |               3 | oracle_true_d_mod3 |  49 |              9 | 0.5778 |  0.4136 |  0.7393 |               -0.0712 |       -0.3086 |        0.1702 |    0.5725 |
| raw       | CN_to_MCI |               3 | rasper_kendall     |  49 |              9 | 0.3472 |  0.1745 |  0.5335 |               -0.304  |       -0.6    |       -0.0055 |    0.0484 |
| raw       | CN_to_MCI |               4 | ridge_d_hat        |  42 |              9 | 0.6498 |  0.4521 |  0.8216 |              nan      |      nan      |      nan      |  nan      |
| raw       | CN_to_MCI |               4 | direct_logistic    |  42 |              9 | 0.3872 |  0.2204 |  0.5622 |               -0.2638 |       -0.5331 |        0.0344 |    0.0864 |
| raw       | CN_to_MCI |               4 | oracle_true_d_mod3 |  42 |              9 | 0.5859 |  0.412  |  0.7551 |               -0.0618 |       -0.2961 |        0.1837 |    0.6177 |
| raw       | CN_to_MCI |               4 | rasper_kendall     |  42 |              9 | 0.3771 |  0.2066 |  0.5551 |               -0.2743 |       -0.5526 |        0.0304 |    0.0796 |
| combat    | MCI_to_AD |               2 | ridge_d_hat        |  65 |             14 | 0.7801 |  0.6289 |  0.9083 |              nan      |      nan      |      nan      |  nan      |
| combat    | MCI_to_AD |               2 | direct_logistic    |  65 |             14 | 0.5588 |  0.3831 |  0.7236 |               -0.2215 |       -0.3654 |       -0.0882 |    0.0008 |
| combat    | MCI_to_AD |               2 | oracle_true_d_mod3 |  65 |             14 | 0.6625 |  0.4988 |  0.8109 |               -0.1181 |       -0.2453 |        0.0051 |    0.0616 |
| combat    | MCI_to_AD |               2 | rasper_kendall     |  65 |             14 | 0.5784 |  0.3935 |  0.7507 |               -0.2021 |       -0.3455 |       -0.0714 |    0.002  |
| combat    | MCI_to_AD |               3 | ridge_d_hat        |  51 |             19 | 0.8257 |  0.6953 |  0.9338 |              nan      |      nan      |      nan      |  nan      |
| combat    | MCI_to_AD |               3 | direct_logistic    |  51 |             19 | 0.7007 |  0.5446 |  0.8357 |               -0.1259 |       -0.2572 |       -0.0064 |    0.0432 |
| combat    | MCI_to_AD |               3 | oracle_true_d_mod3 |  51 |             19 | 0.6859 |  0.5236 |  0.8398 |               -0.1403 |       -0.2857 |       -0.0081 |    0.0384 |
| combat    | MCI_to_AD |               3 | rasper_kendall     |  51 |             19 | 0.6842 |  0.5242 |  0.8268 |               -0.1422 |       -0.2778 |       -0.0156 |    0.0244 |
| combat    | MCI_to_AD |               4 | ridge_d_hat        |  45 |             22 | 0.7609 |  0.6032 |  0.8933 |              nan      |      nan      |      nan      |  nan      |
| combat    | MCI_to_AD |               4 | direct_logistic    |  45 |             22 | 0.6403 |  0.4664 |  0.8037 |               -0.1199 |       -0.2823 |        0.0317 |    0.1328 |
| combat    | MCI_to_AD |               4 | oracle_true_d_mod3 |  45 |             22 | 0.664  |  0.486  |  0.816  |               -0.0969 |       -0.2589 |        0.0652 |    0.23   |
| combat    | MCI_to_AD |               4 | rasper_kendall     |  45 |             22 | 0.6561 |  0.4876 |  0.8234 |               -0.1042 |       -0.2696 |        0.0506 |    0.1996 |
| combat    | CN_to_MCI |               2 | ridge_d_hat        |  59 |              5 | 0.8111 |  0.6842 |  0.924  |              nan      |      nan      |      nan      |  nan      |
| combat    | CN_to_MCI |               2 | direct_logistic    |  59 |              5 | 0.3222 |  0.1531 |  0.5091 |               -0.4915 |       -0.7224 |       -0.2289 |    0      |
| combat    | CN_to_MCI |               2 | oracle_true_d_mod3 |  59 |              5 | 0.6519 |  0.4407 |  0.8481 |               -0.1604 |       -0.3684 |        0.0614 |    0.1497 |
| combat    | CN_to_MCI |               2 | rasper_kendall     |  59 |              5 | 0.3852 |  0.2381 |  0.5364 |               -0.4271 |       -0.6053 |       -0.2407 |    0      |
| combat    | CN_to_MCI |               3 | ridge_d_hat        |  49 |              9 | 0.6556 |  0.4864 |  0.8116 |              nan      |      nan      |      nan      |  nan      |
| combat    | CN_to_MCI |               3 | direct_logistic    |  49 |              9 | 0.4056 |  0.2278 |  0.5942 |               -0.2528 |       -0.5375 |        0.0528 |    0.0976 |
| combat    | CN_to_MCI |               3 | oracle_true_d_mod3 |  49 |              9 | 0.5778 |  0.4136 |  0.7393 |               -0.0777 |       -0.2755 |        0.1341 |    0.4833 |
| combat    | CN_to_MCI |               3 | rasper_kendall     |  49 |              9 | 0.3972 |  0.2278 |  0.5751 |               -0.2608 |       -0.5244 |        0.0102 |    0.0604 |
| combat    | CN_to_MCI |               4 | ridge_d_hat        |  42 |              9 | 0.6465 |  0.4649 |  0.8088 |              nan      |      nan      |      nan      |  nan      |
| combat    | CN_to_MCI |               4 | direct_logistic    |  42 |              9 | 0.4512 |  0.2625 |  0.6487 |               -0.1971 |       -0.478  |        0.1085 |    0.1868 |
| combat    | CN_to_MCI |               4 | oracle_true_d_mod3 |  42 |              9 | 0.5859 |  0.412  |  0.7551 |               -0.0587 |       -0.2735 |        0.1633 |    0.5973 |
| combat    | CN_to_MCI |               4 | rasper_kendall     |  42 |              9 | 0.4377 |  0.2404 |  0.6434 |               -0.2117 |       -0.4908 |        0.0957 |    0.1536 |
