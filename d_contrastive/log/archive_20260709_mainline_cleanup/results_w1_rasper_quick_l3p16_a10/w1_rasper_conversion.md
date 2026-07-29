# Workstream 1: RASPER conversion results

Headline contrast: `rasper_*` (borrow ranking) vs `ridge_d_hat` (borrow score). `delta_*` columns are paired bootstrap AUC differences vs `ridge_d_hat`.

| version   | task      |   horizon_years | method             |   n |   n_converters |    auc |   ci_lo |   ci_hi |   delta_vs_ridge_dhat |   delta_ci_lo |   delta_ci_hi |   delta_p |
|:----------|:----------|----------------:|:-------------------|----:|---------------:|-------:|--------:|--------:|----------------------:|--------------:|--------------:|----------:|
| raw       | MCI_to_AD |               2 | ridge_d_hat        |  65 |             14 | 0.7563 |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | MCI_to_AD |               2 | direct_logistic    |  65 |             14 | 0.5294 |     nan |     nan |               -0.2269 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               2 | oracle_true_d_mod3 |  65 |             14 | 0.6625 |     nan |     nan |               -0.0938 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               2 | rasper_kendall     |  65 |             14 | 0.5434 |     nan |     nan |               -0.2129 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               3 | ridge_d_hat        |  51 |             19 | 0.8174 |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | MCI_to_AD |               3 | direct_logistic    |  51 |             19 | 0.676  |     nan |     nan |               -0.1414 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               3 | oracle_true_d_mod3 |  51 |             19 | 0.6859 |     nan |     nan |               -0.1316 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               3 | rasper_kendall     |  51 |             19 | 0.6776 |     nan |     nan |               -0.1398 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               4 | ridge_d_hat        |  45 |             22 | 0.7648 |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | MCI_to_AD |               4 | direct_logistic    |  45 |             22 | 0.6423 |     nan |     nan |               -0.1225 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               4 | oracle_true_d_mod3 |  45 |             22 | 0.664  |     nan |     nan |               -0.1008 |           nan |           nan |       nan |
| raw       | MCI_to_AD |               4 | rasper_kendall     |  45 |             22 | 0.6423 |     nan |     nan |               -0.1225 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               2 | ridge_d_hat        |  59 |              5 | 0.7889 |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | CN_to_MCI |               2 | direct_logistic    |  59 |              5 | 0.3074 |     nan |     nan |               -0.4815 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               2 | oracle_true_d_mod3 |  59 |              5 | 0.6519 |     nan |     nan |               -0.137  |           nan |           nan |       nan |
| raw       | CN_to_MCI |               2 | rasper_kendall     |  59 |              5 | 0.4185 |     nan |     nan |               -0.3704 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               3 | ridge_d_hat        |  49 |              9 | 0.65   |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | CN_to_MCI |               3 | direct_logistic    |  49 |              9 | 0.3444 |     nan |     nan |               -0.3056 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               3 | oracle_true_d_mod3 |  49 |              9 | 0.5778 |     nan |     nan |               -0.0722 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               3 | rasper_kendall     |  49 |              9 | 0.3472 |     nan |     nan |               -0.3028 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               4 | ridge_d_hat        |  42 |              9 | 0.6498 |     nan |     nan |              nan      |           nan |           nan |       nan |
| raw       | CN_to_MCI |               4 | direct_logistic    |  42 |              9 | 0.3872 |     nan |     nan |               -0.2626 |           nan |           nan |       nan |
| raw       | CN_to_MCI |               4 | oracle_true_d_mod3 |  42 |              9 | 0.5859 |     nan |     nan |               -0.064  |           nan |           nan |       nan |
| raw       | CN_to_MCI |               4 | rasper_kendall     |  42 |              9 | 0.4646 |     nan |     nan |               -0.1852 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               2 | ridge_d_hat        |  65 |             14 | 0.7801 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | MCI_to_AD |               2 | direct_logistic    |  65 |             14 | 0.5588 |     nan |     nan |               -0.2213 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               2 | oracle_true_d_mod3 |  65 |             14 | 0.6625 |     nan |     nan |               -0.1176 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               2 | rasper_kendall     |  65 |             14 | 0.5742 |     nan |     nan |               -0.2059 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               3 | ridge_d_hat        |  51 |             19 | 0.8257 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | MCI_to_AD |               3 | direct_logistic    |  51 |             19 | 0.7007 |     nan |     nan |               -0.125  |           nan |           nan |       nan |
| combat    | MCI_to_AD |               3 | oracle_true_d_mod3 |  51 |             19 | 0.6859 |     nan |     nan |               -0.1398 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               3 | rasper_kendall     |  51 |             19 | 0.6842 |     nan |     nan |               -0.1414 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               4 | ridge_d_hat        |  45 |             22 | 0.7609 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | MCI_to_AD |               4 | direct_logistic    |  45 |             22 | 0.6403 |     nan |     nan |               -0.1206 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               4 | oracle_true_d_mod3 |  45 |             22 | 0.664  |     nan |     nan |               -0.0968 |           nan |           nan |       nan |
| combat    | MCI_to_AD |               4 | rasper_kendall     |  45 |             22 | 0.6502 |     nan |     nan |               -0.1107 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               2 | ridge_d_hat        |  59 |              5 | 0.8111 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | CN_to_MCI |               2 | direct_logistic    |  59 |              5 | 0.3222 |     nan |     nan |               -0.4889 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               2 | oracle_true_d_mod3 |  59 |              5 | 0.6519 |     nan |     nan |               -0.1593 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               2 | rasper_kendall     |  59 |              5 | 0.3852 |     nan |     nan |               -0.4259 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               3 | ridge_d_hat        |  49 |              9 | 0.6556 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | CN_to_MCI |               3 | direct_logistic    |  49 |              9 | 0.4056 |     nan |     nan |               -0.25   |           nan |           nan |       nan |
| combat    | CN_to_MCI |               3 | oracle_true_d_mod3 |  49 |              9 | 0.5778 |     nan |     nan |               -0.0778 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               3 | rasper_kendall     |  49 |              9 | 0.4111 |     nan |     nan |               -0.2444 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               4 | ridge_d_hat        |  42 |              9 | 0.6465 |     nan |     nan |              nan      |           nan |           nan |       nan |
| combat    | CN_to_MCI |               4 | direct_logistic    |  42 |              9 | 0.4512 |     nan |     nan |               -0.1953 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               4 | oracle_true_d_mod3 |  42 |              9 | 0.5859 |     nan |     nan |               -0.0606 |           nan |           nan |       nan |
| combat    | CN_to_MCI |               4 | rasper_kendall     |  42 |              9 | 0.4108 |     nan |     nan |               -0.2357 |           nan |           nan |       nan |
