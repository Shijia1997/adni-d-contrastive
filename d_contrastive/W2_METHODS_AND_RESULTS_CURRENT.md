# W2 Methods and Results Current Snapshot

Generated from current files under `d_contrastive/results_w2_*`. Age/APOE analyses are intentionally excluded. Two-split setup: train/fine-tune split vs held-out test; no patient overlap.

## Current job status

- `w2_full_retry`: submitted as SLURM job `34087065`; pending at snapshot time. This retries full Swin fine-tune with smaller physical batch and gradient accumulation.

- `w2_direct_lora`: submitted as SLURM job `34087066`; pending at snapshot time. This is direct downstream LoRA fine-tuning on diagnosis/conversion labels.

- Existing completed results below do not yet include those two pending jobs.

## Methods

- `oracle_true_d`: Uses held-out test `d_mod3` directly as an upper-bound reference. Not deployable; this asks how much Wang D itself separates diagnosis/conversion.

- `ridge_dhat_all / ridge_dhat_finetune`: Train Ridge on frozen Swin 768-d image embedding to predict `d_mod3`; use predicted d-hat for diagnosis/conversion. Deployable because it uses image only at test time.

- `baseline_raw / direct_logistic`: Classical linear model on the 768-d embedding: logistic regression for classification/conversion, Ridge for D prediction. No MLP pretraining.

- `contrastive_euclidean_probe`: MLP projection of frozen 768-d Swin features, trained with Y-aware Euclidean contrastive loss using `d_mod3`; then freeze MLP and train a linear downstream probe.

- `contrastive_euclidean_s`: Use the contrastive MLP 1D progression head/score directly as the task score, no downstream linear probe.

- `contrastive_euclidean_finetune`: Initialize from Euclidean d-pretrained MLP, then fine-tune MLP/head on the downstream task label.

- `contrastive_rank_kendall_basic_*`: Train a 1D progression score with differentiable soft-Kendall/rank ordering against `d_mod3`; evaluated as probe, direct score `_s`, or downstream fine-tune.

- `contrastive_hybrid_basic_*`: Hybrid d-pretraining: Y-aware Euclidean contrastive loss plus rank/Kendall loss; evaluated as probe, direct score `_s`, or downstream fine-tune.

- `contrastive_regress_d_*`: Direct supervised d-pretraining: MLP score trained by MSE to `d_mod3`, then evaluated as probe, direct score, or downstream fine-tune.

- `image_only / image+dhat / dhat_only / image+dtrue / dtrue_only`: Support analysis comparing image features, image plus deployable d-hat, d-hat alone, image plus true d, and true d alone.

- `lora_regress_d / lora_euclidean / lora_hybrid_basic`: SwinUNETR encoder adapted with LoRA using d-supervised losses, then exported back to 768-d embeddings and evaluated with the same CPU downstream harness.

- `full_*`: Full Swin fine-tune variants are pending retry; previous attempt OOMed.

- `direct_lora_downstream`: Direct LoRA fine-tune on downstream diagnosis/conversion labels is pending; it is not d-pretraining.


## Result files

- `d_contrastive/results_w2_overall_summary/w2_conversion_all_methods.csv`: 390 rows x 11 cols

- `d_contrastive/results_w2_overall_summary/w2_conversion_best_by_horizon.csv`: 30 rows x 11 cols

- `d_contrastive/results_w2_overall_summary/w2_conversion_best_pivot.csv`: 10 rows x 10 cols

- `d_contrastive/results_w2_overall_summary/w2_d_relation_all.csv`: 40 rows x 12 cols

- `d_contrastive/results_w2_overall_summary/w2_d_support_all.csv`: 90 rows x 12 cols

- `d_contrastive/results_w2_overall_summary/w2_diagnosis_all_methods.csv`: 115 rows x 11 cols

- `d_contrastive/results_w2_overall_summary/w2_diagnosis_best_by_task.csv`: 20 rows x 11 cols


## D vs diagnosis/conversion

| stage             | source      | result_type   | version   | task_kind   | task               |   n |   n_positive | method         |      auc |      ci_lo |      ci_hi |
|:------------------|:------------|:--------------|:----------|:------------|:-------------------|----:|-------------:|:---------------|---------:|-----------:|-----------:|
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | CN_vs_AD           | 352 |          102 | oracle_true_d  | 0.934667 |   0.90532  |   0.959883 |
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | CN_vs_AD           | 352 |          102 | ridge_dhat_all | 0.825765 |   0.776746 |   0.870851 |
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | CN_vs_MCI          | 509 |          259 | oracle_true_d  | 0.718672 |   0.673262 |   0.761721 |
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | CN_vs_MCI          | 509 |          259 | ridge_dhat_all | 0.599228 |   0.550835 |   0.647405 |
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | MCI_vs_AD          | 361 |          102 | oracle_true_d  | 0.773526 |   0.719754 |   0.822235 |
| phase0_d_relation | frozen_swin | mixed         | raw       | diagnosis   | MCI_vs_AD          | 361 |          102 | ridge_dhat_all | 0.744757 |   0.688308 |   0.798676 |
| phase0_d_relation | frozen_swin | mixed         | raw       | dx_ordinal  | CN<MCI<AD_spearman | 611 |          nan | oracle_true_d  | 0.554954 | nan        | nan        |
| phase0_d_relation | frozen_swin | mixed         | raw       | dx_ordinal  | CN<MCI<AD_spearman | 611 |          nan | ridge_dhat_all | 0.365955 | nan        | nan        |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | CN_vs_AD           | 352 |          102 | oracle_true_d  | 0.934667 |   0.90532  |   0.959883 |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | CN_vs_AD           | 352 |          102 | ridge_dhat_all | 0.802235 |   0.752559 |   0.848983 |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | CN_vs_MCI          | 509 |          259 | oracle_true_d  | 0.718672 |   0.673262 |   0.761721 |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | CN_vs_MCI          | 509 |          259 | ridge_dhat_all | 0.590147 |   0.541164 |   0.639361 |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | MCI_vs_AD          | 361 |          102 | oracle_true_d  | 0.773526 |   0.719754 |   0.822235 |
| phase0_d_relation | frozen_swin | mixed         | combat    | diagnosis   | MCI_vs_AD          | 361 |          102 | ridge_dhat_all | 0.742373 |   0.685682 |   0.79578  |
| phase0_d_relation | frozen_swin | mixed         | combat    | dx_ordinal  | CN<MCI<AD_spearman | 611 |          nan | oracle_true_d  | 0.554954 | nan        | nan        |
| phase0_d_relation | frozen_swin | mixed         | combat    | dx_ordinal  | CN<MCI<AD_spearman | 611 |          nan | ridge_dhat_all | 0.341883 | nan        | nan        |


## D vs conversion by year

| stage             | source      | result_type   | version   | task_kind   | task         |   n |   n_positive | method         |      auc |    ci_lo |    ci_hi |
|:------------------|:------------|:--------------|:----------|:------------|:-------------|----:|-------------:|:---------------|---------:|---------:|---------:|
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | oracle_true_d  | 0.662465 | 0.498817 | 0.810876 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | ridge_dhat_all | 0.754902 | 0.603634 | 0.883846 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | oracle_true_d  | 0.685855 | 0.523555 | 0.839774 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | ridge_dhat_all | 0.815789 | 0.683638 | 0.927421 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | oracle_true_d  | 0.664032 | 0.485955 | 0.816    |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | ridge_dhat_all | 0.762846 | 0.605263 | 0.896552 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | oracle_true_d  | 0.643636 | 0.429528 | 0.856647 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | ridge_dhat_all | 0.778182 | 0.573425 | 0.941964 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | oracle_true_d  | 0.577778 | 0.413607 | 0.739314 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | ridge_dhat_all | 0.65     | 0.467949 | 0.827778 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | oracle_true_d  | 0.585859 | 0.412023 | 0.755102 |
| phase0_d_relation | frozen_swin | mixed         | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | ridge_dhat_all | 0.649832 | 0.452096 | 0.821552 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | oracle_true_d  | 0.662465 | 0.498817 | 0.810876 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | ridge_dhat_all | 0.780112 | 0.62891  | 0.908295 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | oracle_true_d  | 0.685855 | 0.523555 | 0.839774 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | ridge_dhat_all | 0.825658 | 0.695286 | 0.933846 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | oracle_true_d  | 0.664032 | 0.485955 | 0.816    |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | ridge_dhat_all | 0.76087  | 0.603175 | 0.893299 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | oracle_true_d  | 0.643636 | 0.429528 | 0.856647 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | ridge_dhat_all | 0.796364 | 0.667027 | 0.910714 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | oracle_true_d  | 0.577778 | 0.413607 | 0.739314 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | ridge_dhat_all | 0.655556 | 0.486364 | 0.811594 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | oracle_true_d  | 0.585859 | 0.412023 | 0.755102 |
| phase0_d_relation | frozen_swin | mixed         | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | ridge_dhat_all | 0.646465 | 0.464865 | 0.808824 |


## Best diagnosis rows by stage/source/version/task

| stage                         | source            | version   | task_kind   | task      | method                               |   n |   n_positive |      auc |      ci_lo |      ci_hi |
|:------------------------------|:------------------|:----------|:------------|:----------|:-------------------------------------|----:|-------------:|---------:|-----------:|-----------:|
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.676919 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.846863 |   0.803901 |   0.885736 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.610131 |   0.56178  |   0.658168 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.763078 |   0.708182 |   0.814512 |
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.663726 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.859686 |   0.817303 |   0.897029 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.602718 |   0.554198 |   0.651267 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.762132 |   0.709178 |   0.812586 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.660599 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.863765 |   0.823444 |   0.901691 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.58939  |   0.54012  |   0.638109 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.752782 |   0.700138 |   0.803491 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.664532 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.871451 |   0.830249 |   0.908484 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.610672 |   0.561057 |   0.659167 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.765766 |   0.71372  |   0.815153 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.697713 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.909412 |   0.878302 |   0.938426 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.635876 |   0.587866 |   0.683143 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.79003  |   0.742078 |   0.836896 |


## Best conversion rows by stage/source/version/task/year

| stage                         | source            | version   | task      |   horizon_years | method                                  |   n |   n_converters |      auc |    ci_lo |    ci_hi |
|:------------------------------|:------------------|:----------|:----------|----------------:|:----------------------------------------|----:|---------------:|---------:|---------:|---------:|
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.716364 | 0.532164 | 0.877193 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.658333 | 0.502436 | 0.810976 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.632997 | 0.449074 | 0.790635 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.7493   | 0.601468 | 0.875798 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.809211 | 0.682258 | 0.920809 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.756917 | 0.603237 | 0.888889 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.705455 | 0.465469 | 0.915254 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.633333 | 0.466667 | 0.795741 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.606061 | 0.413427 | 0.777778 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.733894 | 0.585314 | 0.863636 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.8125   | 0.682258 | 0.921059 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.784585 | 0.633323 | 0.908911 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.832727 | 0.687273 | 0.949153 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.655556 | 0.45237  | 0.847222 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.646465 | 0.403122 | 0.857215 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.768908 | 0.636477 | 0.878676 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.838816 | 0.720494 | 0.935855 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.798419 | 0.650794 | 0.923077 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.829091 | 0.689739 | 0.949153 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.677778 | 0.502767 | 0.844961 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.676768 | 0.488803 | 0.837844 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.767507 | 0.646226 | 0.872551 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.838816 | 0.721429 | 0.932567 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.774704 | 0.619048 | 0.906749 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.698182 | 0.465455 | 0.867925 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.572222 | 0.368868 | 0.766667 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.619529 | 0.402778 | 0.810216 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.722689 | 0.574443 | 0.850773 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.726974 | 0.569355 | 0.863658 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.697628 | 0.529644 | 0.849802 |


## Best conversion pivot

| stage                         | source            | version   | task      |   y2_auc | y2_method                            |   y3_auc | y3_method                               |   y4_auc | y4_method                               |
|:------------------------------|:------------------|:----------|:----------|---------:|:-------------------------------------|---------:|:----------------------------------------|---------:|:----------------------------------------|
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI | 0.716364 | contrastive_regress_d_s              | 0.658333 | contrastive_regress_d_s                 | 0.632997 | contrastive_regress_d_s                 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI | 0.705455 | contrastive_hybrid_basic_s           | 0.633333 | contrastive_hybrid_basic_probe          | 0.606061 | contrastive_hybrid_basic_probe          |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI | 0.832727 | contrastive_euclidean_s              | 0.655556 | contrastive_regress_d_s                 | 0.646465 | contrastive_rank_kendall_basic_finetune |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI | 0.829091 | contrastive_euclidean_probe          | 0.677778 | contrastive_rank_kendall_basic_s        | 0.676768 | contrastive_rank_kendall_basic_s        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI | 0.698182 | contrastive_rank_kendall_basic_probe | 0.572222 | contrastive_euclidean_s                 | 0.619529 | contrastive_regress_d_finetune          |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD | 0.7493   | contrastive_euclidean_probe          | 0.809211 | contrastive_rank_kendall_basic_s        | 0.756917 | contrastive_euclidean_probe             |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD | 0.733894 | contrastive_hybrid_basic_s           | 0.8125   | contrastive_regress_d_probe             | 0.784585 | contrastive_euclidean_probe             |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD | 0.768908 | contrastive_regress_d_probe          | 0.838816 | contrastive_rank_kendall_basic_finetune | 0.798419 | contrastive_regress_d_s                 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD | 0.767507 | contrastive_rank_kendall_basic_s     | 0.838816 | contrastive_rank_kendall_basic_probe    | 0.774704 | contrastive_hybrid_basic_probe          |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD | 0.722689 | direct_logistic                      | 0.726974 | direct_logistic                         | 0.697628 | direct_logistic                         |


## d-support full table

| stage            | source      | result_type          | version   | task_kind   | task         |   n |   n_positive | method      |      auc |    ci_lo |    ci_hi |
|:-----------------|:------------|:---------------------|:----------|:------------|:-------------|----:|-------------:|:------------|---------:|---------:|---------:|
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_AD     | 352 |          102 | image_only  | 0.859686 | 0.817285 | 0.896949 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_AD     | 352 |          102 | image+dhat  | 0.856353 | 0.814313 | 0.895213 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_AD     | 352 |          102 | dhat_only   | 0.825725 | 0.776644 | 0.870809 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_AD     | 352 |          102 | image+dtrue | 0.960275 | 0.940372 | 0.977092 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_AD     | 352 |          102 | dtrue_only  | 0.934667 | 0.90532  | 0.959883 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_MCI    | 509 |          259 | image_only  | 0.576479 | 0.52842  | 0.624869 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_MCI    | 509 |          259 | image+dhat  | 0.587042 | 0.539078 | 0.635727 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_MCI    | 509 |          259 | dhat_only   | 0.599259 | 0.550849 | 0.647452 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_MCI    | 509 |          259 | image+dtrue | 0.67912  | 0.632222 | 0.724107 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | CN_vs_MCI    | 509 |          259 | dtrue_only  | 0.718672 | 0.673262 | 0.761721 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | MCI_vs_AD    | 361 |          102 | image_only  | 0.761791 | 0.709031 | 0.812147 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | MCI_vs_AD    | 361 |          102 | image+dhat  | 0.760921 | 0.70727  | 0.811323 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | MCI_vs_AD    | 361 |          102 | dhat_only   | 0.744795 | 0.688363 | 0.798677 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | MCI_vs_AD    | 361 |          102 | image+dtrue | 0.800855 | 0.75372  | 0.844077 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | dx_binary   | MCI_vs_AD    | 361 |          102 | dtrue_only  | 0.773526 | 0.719754 | 0.822235 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | image_only  | 0.665266 | 0.516801 | 0.809172 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | image+dhat  | 0.705882 | 0.563993 | 0.835559 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | dhat_only   | 0.754902 | 0.603634 | 0.883846 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | image+dtrue | 0.710084 | 0.571421 | 0.836478 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_2y |  65 |           14 | dtrue_only  | 0.662465 | 0.498817 | 0.810876 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | image_only  | 0.764803 | 0.624989 | 0.882452 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | image+dhat  | 0.804276 | 0.673003 | 0.912698 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | dhat_only   | 0.815789 | 0.683638 | 0.927421 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | image+dtrue | 0.799342 | 0.667711 | 0.907407 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_3y |  51 |           19 | dtrue_only  | 0.685855 | 0.523555 | 0.839774 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | image_only  | 0.656126 | 0.488141 | 0.812506 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | image+dhat  | 0.715415 | 0.55357  | 0.861133 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | dhat_only   | 0.762846 | 0.605263 | 0.896552 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | image+dtrue | 0.703557 | 0.54251  | 0.850202 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | MCI_to_AD_4y |  45 |           22 | dtrue_only  | 0.664032 | 0.485955 | 0.816    |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | image_only  | 0.363636 | 0.156364 | 0.576271 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | image+dhat  | 0.476364 | 0.210909 | 0.749795 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | dhat_only   | 0.778182 | 0.573425 | 0.941964 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | image+dtrue | 0.512727 | 0.261656 | 0.728814 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_2y |  60 |            5 | dtrue_only  | 0.643636 | 0.429528 | 0.856647 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | image_only  | 0.486111 | 0.304878 | 0.674362 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | image+dhat  | 0.555556 | 0.370751 | 0.736434 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | dhat_only   | 0.65     | 0.467949 | 0.827778 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | image+dtrue | 0.597222 | 0.404246 | 0.78604  |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_3y |  49 |            9 | dtrue_only  | 0.577778 | 0.413607 | 0.739314 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | image_only  | 0.424242 | 0.22449  | 0.626301 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | image+dhat  | 0.498316 | 0.305147 | 0.6875   |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | dhat_only   | 0.649832 | 0.452096 | 0.821552 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | image+dtrue | 0.56229  | 0.368734 | 0.746939 |
| phase1_d_support | frozen_swin | diagnosis_conversion | raw       | conversion  | CN_to_MCI_4y |  42 |            9 | dtrue_only  | 0.585859 | 0.412023 | 0.755102 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_AD     | 352 |          102 | image_only  | 0.84698  | 0.804157 | 0.886043 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_AD     | 352 |          102 | image+dhat  | 0.842118 | 0.799921 | 0.8819   |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_AD     | 352 |          102 | dhat_only   | 0.802157 | 0.752488 | 0.84895  |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_AD     | 352 |          102 | image+dtrue | 0.958118 | 0.938016 | 0.975574 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_AD     | 352 |          102 | dtrue_only  | 0.934667 | 0.90532  | 0.959883 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_MCI    | 509 |          259 | image_only  | 0.578826 | 0.530616 | 0.627672 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_MCI    | 509 |          259 | image+dhat  | 0.589066 | 0.541492 | 0.636855 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_MCI    | 509 |          259 | dhat_only   | 0.590147 | 0.541164 | 0.639361 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_MCI    | 509 |          259 | image+dtrue | 0.685884 | 0.640177 | 0.729464 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | CN_vs_MCI    | 509 |          259 | dtrue_only  | 0.718672 | 0.673262 | 0.761721 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | MCI_vs_AD    | 361 |          102 | image_only  | 0.760656 | 0.707649 | 0.811883 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | MCI_vs_AD    | 361 |          102 | image+dhat  | 0.761186 | 0.708315 | 0.812641 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | MCI_vs_AD    | 361 |          102 | dhat_only   | 0.742335 | 0.685611 | 0.79578  |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | MCI_vs_AD    | 361 |          102 | image+dtrue | 0.80112  | 0.753912 | 0.844027 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | dx_binary   | MCI_vs_AD    | 361 |          102 | dtrue_only  | 0.773526 | 0.719754 | 0.822235 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | image_only  | 0.70028  | 0.548796 | 0.841611 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | image+dhat  | 0.717087 | 0.574074 | 0.846939 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | dhat_only   | 0.780112 | 0.62891  | 0.908295 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | image+dtrue | 0.710084 | 0.568649 | 0.838398 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_2y |  65 |           14 | dtrue_only  | 0.662465 | 0.498817 | 0.810876 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | image_only  | 0.777961 | 0.637648 | 0.892734 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | image+dhat  | 0.800987 | 0.665077 | 0.910783 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | dhat_only   | 0.825658 | 0.695286 | 0.933846 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | image+dtrue | 0.8125   | 0.683491 | 0.91524  |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_3y |  51 |           19 | dtrue_only  | 0.685855 | 0.523555 | 0.839774 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | image_only  | 0.644269 | 0.47     | 0.8      |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | image+dhat  | 0.70751  | 0.541667 | 0.85119  |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | dhat_only   | 0.76087  | 0.603175 | 0.893299 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | image+dtrue | 0.711462 | 0.550607 | 0.851779 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | MCI_to_AD_4y |  45 |           22 | dtrue_only  | 0.664032 | 0.485955 | 0.816    |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | image_only  | 0.316364 | 0.129326 | 0.532164 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | image+dhat  | 0.44     | 0.197823 | 0.690058 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | dhat_only   | 0.796364 | 0.667027 | 0.910714 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | image+dtrue | 0.454545 | 0.215517 | 0.676364 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_2y |  60 |            5 | dtrue_only  | 0.643636 | 0.429528 | 0.856647 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | image_only  | 0.447222 | 0.252724 | 0.652778 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | image+dhat  | 0.511111 | 0.329017 | 0.691687 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | dhat_only   | 0.655556 | 0.486364 | 0.811594 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | image+dtrue | 0.552778 | 0.350579 | 0.752818 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_3y |  49 |            9 | dtrue_only  | 0.577778 | 0.413607 | 0.739314 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | image_only  | 0.420875 | 0.228739 | 0.619571 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | image+dhat  | 0.484848 | 0.282828 | 0.681081 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | dhat_only   | 0.646465 | 0.464865 | 0.808824 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | image+dtrue | 0.545455 | 0.352731 | 0.734025 |
| phase1_d_support | frozen_swin | diagnosis_conversion | combat    | conversion  | CN_to_MCI_4y |  42 |            9 | dtrue_only  | 0.585859 | 0.412023 | 0.755102 |


## Full diagnosis all-method table

| stage                         | source            | version   | task_kind   | task      | method                               |   n |   n_positive |      auc |      ci_lo |      ci_hi |
|:------------------------------|:------------------|:----------|:------------|:----------|:-------------------------------------|----:|-------------:|---------:|-----------:|-----------:|
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.660425 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.646347 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.663726 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | contrastive_hybrid_basic_probe       | 611 |          nan | 0.625592 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx3         | CN_MCI_AD | contrastive_regress_d_probe          | 611 |          nan | 0.652709 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.859686 |   0.817303 |   0.897029 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | ridge_dhat_all                       | 352 |          102 | 0.825765 |   0.776746 |   0.870851 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | contrastive_euclidean_probe          | 352 |          102 | 0.806431 |   0.758623 |   0.85138  |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.806078 |   0.755252 |   0.854223 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | contrastive_hybrid_basic_probe       | 352 |          102 | 0.789647 |   0.741406 |   0.837019 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_AD  | contrastive_regress_d_probe          | 352 |          102 | 0.804275 |   0.753244 |   0.85164  |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.57668  |   0.528461 |   0.62505  |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.599228 |   0.550835 |   0.647405 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.594795 |   0.54587  |   0.643694 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.602718 |   0.554198 |   0.651267 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | contrastive_hybrid_basic_probe       | 509 |          259 | 0.581205 |   0.53165  |   0.630459 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | CN_vs_MCI | contrastive_regress_d_probe          | 509 |          259 | 0.586178 |   0.538415 |   0.634828 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.762132 |   0.709178 |   0.812586 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.744757 |   0.688308 |   0.798676 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | contrastive_euclidean_probe          | 361 |          102 | 0.726399 |   0.668733 |   0.779459 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | contrastive_rank_kendall_basic_probe | 361 |          102 | 0.749224 |   0.691991 |   0.80341  |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | contrastive_hybrid_basic_probe       | 361 |          102 | 0.708948 |   0.652349 |   0.763431 |
| phase1_frozen                 | frozen_swin       | raw       | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.751117 |   0.69639  |   0.802999 |
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.670822 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.64673  | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.676919 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | contrastive_hybrid_basic_probe       | 611 |          nan | 0.638685 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx3         | CN_MCI_AD | contrastive_regress_d_probe          | 611 |          nan | 0.654554 | nan        | nan        |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.846863 |   0.803901 |   0.885736 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | ridge_dhat_all                       | 352 |          102 | 0.802235 |   0.752559 |   0.848983 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | contrastive_euclidean_probe          | 352 |          102 | 0.792    |   0.743006 |   0.839569 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.829686 |   0.7814   |   0.874318 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | contrastive_hybrid_basic_probe       | 352 |          102 | 0.785686 |   0.736104 |   0.83365  |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_AD  | contrastive_regress_d_probe          | 352 |          102 | 0.819176 |   0.772382 |   0.86544  |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.578703 |   0.530772 |   0.627631 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.590147 |   0.541164 |   0.639361 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.608062 |   0.559106 |   0.656697 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.610131 |   0.56178  |   0.658168 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | contrastive_hybrid_basic_probe       | 509 |          259 | 0.602054 |   0.552942 |   0.651791 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | CN_vs_MCI | contrastive_regress_d_probe          | 509 |          259 | 0.575614 |   0.527623 |   0.625465 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.760807 |   0.708029 |   0.81182  |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.742373 |   0.685682 |   0.79578  |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | contrastive_euclidean_probe          | 361 |          102 | 0.703535 |   0.644968 |   0.759151 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | contrastive_rank_kendall_basic_probe | 361 |          102 | 0.762094 |   0.706342 |   0.812707 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | contrastive_hybrid_basic_probe       | 361 |          102 | 0.703535 |   0.642598 |   0.759671 |
| phase1_frozen                 | frozen_swin       | combat    | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.763078 |   0.708182 |   0.814512 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.697713 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.689481 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.677961 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | contrastive_hybrid_basic_probe       | 611 |          nan | 0.686731 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx3         | CN_MCI_AD | contrastive_regress_d_probe          | 611 |          nan | 0.679439 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.903843 |   0.870969 |   0.935703 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | ridge_dhat_all                       | 352 |          102 | 0.881255 |   0.839547 |   0.918417 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | contrastive_euclidean_probe          | 352 |          102 | 0.890588 |   0.850664 |   0.926511 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.909412 |   0.878302 |   0.938426 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | contrastive_hybrid_basic_probe       | 352 |          102 | 0.89102  |   0.85202  |   0.926232 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_AD  | contrastive_regress_d_probe          | 352 |          102 | 0.905765 |   0.873929 |   0.93523  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.635876 |   0.587866 |   0.683143 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.628247 |   0.580049 |   0.676116 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.62939  |   0.580814 |   0.677395 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.603367 |   0.554215 |   0.651983 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | contrastive_hybrid_basic_probe       | 509 |          259 | 0.624772 |   0.576313 |   0.672763 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | CN_vs_MCI | contrastive_regress_d_probe          | 509 |          259 | 0.61512  |   0.566585 |   0.664404 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.773677 |   0.72192  |   0.823391 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.774434 |   0.722199 |   0.825162 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | contrastive_euclidean_probe          | 361 |          102 | 0.779241 |   0.728211 |   0.828981 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | contrastive_rank_kendall_basic_probe | 361 |          102 | 0.785071 |   0.735719 |   0.832314 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | contrastive_hybrid_basic_probe       | 361 |          102 | 0.779809 |   0.728583 |   0.828944 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.79003  |   0.742078 |   0.836896 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.660599 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.622163 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.64648  | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | contrastive_hybrid_basic_probe       | 611 |          nan | 0.613387 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx3         | CN_MCI_AD | contrastive_regress_d_probe          | 611 |          nan | 0.639124 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.863765 |   0.823444 |   0.901691 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | ridge_dhat_all                       | 352 |          102 | 0.825843 |   0.779158 |   0.868395 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | contrastive_euclidean_probe          | 352 |          102 | 0.791098 |   0.741646 |   0.837229 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.797961 |   0.74708  |   0.846265 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | contrastive_hybrid_basic_probe       | 352 |          102 | 0.76702  |   0.714449 |   0.815268 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_AD  | contrastive_regress_d_probe          | 352 |          102 | 0.813137 |   0.764427 |   0.859611 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.572942 |   0.524252 |   0.622919 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.58939  |   0.54012  |   0.638109 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.57895  |   0.529506 |   0.627492 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.584587 |   0.536141 |   0.633922 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | contrastive_hybrid_basic_probe       | 509 |          259 | 0.572293 |   0.521702 |   0.621009 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | CN_vs_MCI | contrastive_regress_d_probe          | 509 |          259 | 0.583676 |   0.535656 |   0.63333  |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.752782 |   0.700138 |   0.803491 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.744455 |   0.690528 |   0.796851 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | contrastive_euclidean_probe          | 361 |          102 | 0.703006 |   0.647411 |   0.757184 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | contrastive_rank_kendall_basic_probe | 361 |          102 | 0.726323 |   0.668397 |   0.780047 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | contrastive_hybrid_basic_probe       | 361 |          102 | 0.690741 |   0.634188 |   0.746992 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.721175 |   0.664105 |   0.775786 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | baseline_raw                         | 611 |          nan | 0.659423 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | contrastive_euclidean_probe          | 611 |          nan | 0.664532 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | contrastive_rank_kendall_basic_probe | 611 |          nan | 0.648701 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | contrastive_hybrid_basic_probe       | 611 |          nan | 0.638799 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx3         | CN_MCI_AD | contrastive_regress_d_probe          | 611 |          nan | 0.640892 | nan        | nan        |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | baseline_raw                         | 352 |          102 | 0.871451 |   0.830249 |   0.908484 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | ridge_dhat_all                       | 352 |          102 | 0.851333 |   0.808159 |   0.891399 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | contrastive_euclidean_probe          | 352 |          102 | 0.842706 |   0.800554 |   0.881554 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | contrastive_rank_kendall_basic_probe | 352 |          102 | 0.833608 |   0.785328 |   0.878743 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | contrastive_hybrid_basic_probe       | 352 |          102 | 0.807333 |   0.760217 |   0.851777 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_AD  | contrastive_regress_d_probe          | 352 |          102 | 0.819373 |   0.770568 |   0.864867 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | baseline_raw                         | 509 |          259 | 0.570965 |   0.521613 |   0.620221 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | ridge_dhat_all                       | 509 |          259 | 0.610347 |   0.562142 |   0.658269 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | contrastive_euclidean_probe          | 509 |          259 | 0.610672 |   0.561057 |   0.659167 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | contrastive_rank_kendall_basic_probe | 509 |          259 | 0.579305 |   0.531702 |   0.6279   |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | contrastive_hybrid_basic_probe       | 509 |          259 | 0.584247 |   0.534805 |   0.632751 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | CN_vs_MCI | contrastive_regress_d_probe          | 509 |          259 | 0.567166 |   0.517821 |   0.615414 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | baseline_raw                         | 361 |          102 | 0.757211 |   0.702366 |   0.808011 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | ridge_dhat_all                       | 361 |          102 | 0.765766 |   0.71372  |   0.815153 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | contrastive_euclidean_probe          | 361 |          102 | 0.748088 |   0.695636 |   0.798578 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | contrastive_rank_kendall_basic_probe | 361 |          102 | 0.744341 |   0.687089 |   0.796797 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | contrastive_hybrid_basic_probe       | 361 |          102 | 0.732872 |   0.67779  |   0.785214 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | dx_binary   | MCI_vs_AD | contrastive_regress_d_probe          | 361 |          102 | 0.744    |   0.687926 |   0.796627 |


## Full conversion all-method table

| stage                         | source            | version   | task      |   horizon_years | method                                  |   n |   n_converters |      auc |     ci_lo |    ci_hi |
|:------------------------------|:------------------|:----------|:----------|----------------:|:----------------------------------------|----:|---------------:|---------:|----------:|---------:|
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.666667 | 0.518667  | 0.809748 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.729692 | 0.572483  | 0.863646 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_euclidean_s                 |  65 |             14 | 0.271709 | 0.141212  | 0.424361 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_euclidean_finetune          |  65 |             14 | 0.719888 | 0.566038  | 0.855219 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_probe    |  65 |             14 | 0.726891 | 0.572391  | 0.858683 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.72409  | 0.567323  | 0.85692  |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_finetune |  65 |             14 | 0.677871 | 0.511198  | 0.827727 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_probe          |  65 |             14 | 0.715686 | 0.570909  | 0.84314  |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.733894 | 0.585314  | 0.863636 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_finetune       |  65 |             14 | 0.72549  | 0.578947  | 0.853559 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.715686 | 0.567595  | 0.84849  |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_regress_d_s                 |  65 |             14 | 0.69888  | 0.547336  | 0.831368 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               2 | contrastive_regress_d_finetune          |  65 |             14 | 0.693277 | 0.547132  | 0.82398  |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.764803 | 0.624989  | 0.882452 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_euclidean_probe             |  51 |             19 | 0.792763 | 0.65161   | 0.907407 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_euclidean_s                 |  51 |             19 | 0.220395 | 0.10101   | 0.361116 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_euclidean_finetune          |  51 |             19 | 0.769737 | 0.634773  | 0.88871  |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.810855 | 0.679923  | 0.921554 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.794408 | 0.658924  | 0.911113 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.731908 | 0.589965  | 0.860269 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_probe          |  51 |             19 | 0.786184 | 0.65459   | 0.901254 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_s              |  51 |             19 | 0.796053 | 0.663008  | 0.908907 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_finetune       |  51 |             19 | 0.754934 | 0.614815  | 0.882143 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.8125   | 0.682258  | 0.921059 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_regress_d_s                 |  51 |             19 | 0.773026 | 0.636504  | 0.893126 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               3 | contrastive_regress_d_finetune          |  51 |             19 | 0.740132 | 0.598684  | 0.867742 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.656126 | 0.488141  | 0.812506 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.784585 | 0.633323  | 0.908911 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_euclidean_s                 |  45 |             22 | 0.233202 | 0.103239  | 0.385408 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_euclidean_finetune          |  45 |             22 | 0.715415 | 0.546     | 0.862    |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_probe    |  45 |             22 | 0.768775 | 0.61      | 0.912    |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_s        |  45 |             22 | 0.76087  | 0.603986  | 0.900811 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_finetune |  45 |             22 | 0.73913  | 0.571121  | 0.888    |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.745059 | 0.587045  | 0.886833 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_s              |  45 |             22 | 0.752964 | 0.598721  | 0.892716 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_finetune       |  45 |             22 | 0.717391 | 0.551721  | 0.864198 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_regress_d_probe             |  45 |             22 | 0.749012 | 0.586413  | 0.891317 |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.73913  | 0.578924  | 0.884    |
| phase1_frozen                 | frozen_swin       | raw       | MCI_to_AD |               4 | contrastive_regress_d_finetune          |  45 |             22 | 0.6917   | 0.52      | 0.846154 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | direct_logistic                         |  60 |              5 | 0.363636 | 0.156364  | 0.576271 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.690909 | 0.450893  | 0.896143 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.294545 | 0.0994152 | 0.520363 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_euclidean_finetune          |  60 |              5 | 0.48     | 0.187674  | 0.801599 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.534545 | 0.163793  | 0.862069 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_s        |  60 |              5 | 0.625455 | 0.327508  | 0.883636 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_finetune |  60 |              5 | 0.472727 | 0.052843  | 0.846866 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_probe          |  60 |              5 | 0.701818 | 0.491262  | 0.894545 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.705455 | 0.465469  | 0.915254 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_finetune       |  60 |              5 | 0.461818 | 0.155172  | 0.847458 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_regress_d_probe             |  60 |              5 | 0.578182 | 0.25      | 0.867839 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.632727 | 0.269025  | 0.897321 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               2 | contrastive_regress_d_finetune          |  60 |              5 | 0.476364 | 0.155172  | 0.830508 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | direct_logistic                         |  49 |              9 | 0.486111 | 0.304878  | 0.674362 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_euclidean_probe             |  49 |              9 | 0.625    | 0.455526  | 0.789634 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.375    | 0.211111  | 0.541711 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_euclidean_finetune          |  49 |              9 | 0.458333 | 0.286351  | 0.63772  |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_probe    |  49 |              9 | 0.508333 | 0.322222  | 0.695455 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.6      | 0.416603  | 0.780491 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_finetune |  49 |              9 | 0.466667 | 0.297222  | 0.638483 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.633333 | 0.466667  | 0.795741 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_s              |  49 |              9 | 0.627778 | 0.452381  | 0.804358 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_finetune       |  49 |              9 | 0.405556 | 0.258274  | 0.562109 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_regress_d_probe             |  49 |              9 | 0.533333 | 0.355556  | 0.710366 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.580556 | 0.375     | 0.782345 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               3 | contrastive_regress_d_finetune          |  49 |              9 | 0.536111 | 0.359091  | 0.711197 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | direct_logistic                         |  42 |              9 | 0.417508 | 0.220588  | 0.616393 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_euclidean_probe             |  42 |              9 | 0.592593 | 0.402727  | 0.768389 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_euclidean_s                 |  42 |              9 | 0.400673 | 0.228937  | 0.587963 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_euclidean_finetune          |  42 |              9 | 0.350168 | 0.150699  | 0.567957 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_probe    |  42 |              9 | 0.427609 | 0.237838  | 0.624652 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.575758 | 0.381169  | 0.759187 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.343434 | 0.138776  | 0.559375 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.606061 | 0.413427  | 0.777778 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_s              |  42 |              9 | 0.606061 | 0.407881  | 0.783784 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_finetune       |  42 |              9 | 0.37037  | 0.161802  | 0.594398 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_regress_d_probe             |  42 |              9 | 0.484848 | 0.289784  | 0.677551 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.548822 | 0.334372  | 0.744921 |
| phase1_frozen                 | frozen_swin       | raw       | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.360269 | 0.191667  | 0.541667 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.69888  | 0.547612  | 0.840008 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.7493   | 0.601468  | 0.875798 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_euclidean_s                 |  65 |             14 | 0.239496 | 0.114544  | 0.390572 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_euclidean_finetune          |  65 |             14 | 0.719888 | 0.557597  | 0.867282 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_rank_kendall_basic_probe    |  65 |             14 | 0.72549  | 0.575466  | 0.85516  |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.736695 | 0.591268  | 0.860978 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_rank_kendall_basic_finetune |  65 |             14 | 0.691877 | 0.543644  | 0.825333 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_hybrid_basic_probe          |  65 |             14 | 0.740896 | 0.589614  | 0.867935 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.740896 | 0.591033  | 0.871149 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_hybrid_basic_finetune       |  65 |             14 | 0.70028  | 0.547774  | 0.838054 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.710084 | 0.550342  | 0.84927  |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_regress_d_s                 |  65 |             14 | 0.710084 | 0.565455  | 0.838187 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               2 | contrastive_regress_d_finetune          |  65 |             14 | 0.673669 | 0.522009  | 0.813495 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.777961 | 0.637648  | 0.892734 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_euclidean_probe             |  51 |             19 | 0.794408 | 0.655345  | 0.910769 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_euclidean_s                 |  51 |             19 | 0.197368 | 0.0851791 | 0.336712 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_euclidean_finetune          |  51 |             19 | 0.774671 | 0.645315  | 0.891852 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.786184 | 0.652949  | 0.899694 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.809211 | 0.682258  | 0.920809 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.756579 | 0.620961  | 0.876928 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_hybrid_basic_probe          |  51 |             19 | 0.800987 | 0.664815  | 0.914145 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_hybrid_basic_s              |  51 |             19 | 0.799342 | 0.661522  | 0.913498 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_hybrid_basic_finetune       |  51 |             19 | 0.741776 | 0.602074  | 0.865059 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.769737 | 0.633331  | 0.889807 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_regress_d_s                 |  51 |             19 | 0.779605 | 0.646288  | 0.895459 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               3 | contrastive_regress_d_finetune          |  51 |             19 | 0.761513 | 0.624981  | 0.880958 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.644269 | 0.47      | 0.8      |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.756917 | 0.603237  | 0.888889 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_euclidean_s                 |  45 |             22 | 0.245059 | 0.114614  | 0.401456 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_euclidean_finetune          |  45 |             22 | 0.719368 | 0.548     | 0.870445 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_rank_kendall_basic_probe    |  45 |             22 | 0.749012 | 0.591089  | 0.887933 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_rank_kendall_basic_s        |  45 |             22 | 0.735178 | 0.573095  | 0.880569 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_rank_kendall_basic_finetune |  45 |             22 | 0.729249 | 0.562469  | 0.872012 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.747036 | 0.589277  | 0.884011 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_hybrid_basic_s              |  45 |             22 | 0.743083 | 0.585983  | 0.881443 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_hybrid_basic_finetune       |  45 |             22 | 0.721344 | 0.55      | 0.869765 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_regress_d_probe             |  45 |             22 | 0.735178 | 0.571429  | 0.88     |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.711462 | 0.546559  | 0.860344 |
| phase1_frozen                 | frozen_swin       | combat    | MCI_to_AD |               4 | contrastive_regress_d_finetune          |  45 |             22 | 0.721344 | 0.555556  | 0.869048 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | direct_logistic                         |  60 |              5 | 0.316364 | 0.129326  | 0.532164 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.658182 | 0.37847   | 0.877047 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.283636 | 0.103448  | 0.487822 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_euclidean_finetune          |  60 |              5 | 0.407273 | 0.0657942 | 0.847458 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.443636 | 0.0892857 | 0.818713 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_rank_kendall_basic_s        |  60 |              5 | 0.698182 | 0.509168  | 0.859649 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_rank_kendall_basic_finetune |  60 |              5 | 0.4      | 0.112069  | 0.676508 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_hybrid_basic_probe          |  60 |              5 | 0.687273 | 0.402173  | 0.916114 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.690909 | 0.38855   | 0.92697  |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_hybrid_basic_finetune       |  60 |              5 | 0.472727 | 0.169591  | 0.813559 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_regress_d_probe             |  60 |              5 | 0.632727 | 0.356445  | 0.835409 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.716364 | 0.532164  | 0.877193 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               2 | contrastive_regress_d_finetune          |  60 |              5 | 0.530909 | 0.233918  | 0.777778 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | direct_logistic                         |  49 |              9 | 0.447222 | 0.252724  | 0.652778 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_euclidean_probe             |  49 |              9 | 0.619444 | 0.439018  | 0.792525 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.372222 | 0.207317  | 0.541667 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_euclidean_finetune          |  49 |              9 | 0.461111 | 0.277778  | 0.64194  |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_rank_kendall_basic_probe    |  49 |              9 | 0.516667 | 0.339672  | 0.690476 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.636111 | 0.469509  | 0.790909 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_rank_kendall_basic_finetune |  49 |              9 | 0.477778 | 0.313374  | 0.642857 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.627778 | 0.442823  | 0.807927 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_hybrid_basic_s              |  49 |              9 | 0.619444 | 0.428571  | 0.802721 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_hybrid_basic_finetune       |  49 |              9 | 0.458333 | 0.276796  | 0.637205 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_regress_d_probe             |  49 |              9 | 0.597222 | 0.421733  | 0.765247 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.658333 | 0.502436  | 0.810976 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               3 | contrastive_regress_d_finetune          |  49 |              9 | 0.511111 | 0.339672  | 0.67736  |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | direct_logistic                         |  42 |              9 | 0.420875 | 0.228739  | 0.619571 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_euclidean_probe             |  42 |              9 | 0.582492 | 0.389665  | 0.767606 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_euclidean_s                 |  42 |              9 | 0.397306 | 0.222222  | 0.59596  |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_euclidean_finetune          |  42 |              9 | 0.417508 | 0.232427  | 0.610335 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_rank_kendall_basic_probe    |  42 |              9 | 0.417508 | 0.222782  | 0.622256 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.622896 | 0.444416  | 0.781145 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.387205 | 0.212198  | 0.566378 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.602694 | 0.40318   | 0.787037 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_hybrid_basic_s              |  42 |              9 | 0.592593 | 0.381215  | 0.786347 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_hybrid_basic_finetune       |  42 |              9 | 0.457912 | 0.231446  | 0.680556 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_regress_d_probe             |  42 |              9 | 0.535354 | 0.335507  | 0.724373 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.632997 | 0.449074  | 0.790635 |
| phase1_frozen                 | frozen_swin       | combat    | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.444444 | 0.252189  | 0.639706 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.722689 | 0.574443  | 0.850773 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.682073 | 0.508267  | 0.83284  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_euclidean_s                 |  65 |             14 | 0.303922 | 0.155325  | 0.47479  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_euclidean_finetune          |  65 |             14 | 0.680672 | 0.48      | 0.855886 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_probe    |  65 |             14 | 0.640056 | 0.457975  | 0.805422 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.670868 | 0.5       | 0.821333 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_finetune |  65 |             14 | 0.621849 | 0.429643  | 0.797351 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_probe          |  65 |             14 | 0.682073 | 0.516541  | 0.828286 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.689076 | 0.522445  | 0.835531 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_finetune       |  65 |             14 | 0.672269 | 0.466658  | 0.858191 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.680672 | 0.511777  | 0.831811 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_regress_d_s                 |  65 |             14 | 0.680672 | 0.510293  | 0.828629 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               2 | contrastive_regress_d_finetune          |  65 |             14 | 0.662465 | 0.454664  | 0.840336 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.726974 | 0.569355  | 0.863658 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_euclidean_probe             |  51 |             19 | 0.682566 | 0.51265   | 0.835485 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_euclidean_s                 |  51 |             19 | 0.300987 | 0.153225  | 0.470223 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_euclidean_finetune          |  51 |             19 | 0.707237 | 0.544444  | 0.85179  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.705592 | 0.544983  | 0.847753 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.677632 | 0.507129  | 0.830164 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.697368 | 0.537037  | 0.840069 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_probe          |  51 |             19 | 0.690789 | 0.526307  | 0.837508 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_s              |  51 |             19 | 0.695724 | 0.530847  | 0.841751 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_finetune       |  51 |             19 | 0.710526 | 0.541512  | 0.858696 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.705592 | 0.541109  | 0.855827 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_regress_d_s                 |  51 |             19 | 0.689145 | 0.524061  | 0.835485 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               3 | contrastive_regress_d_finetune          |  51 |             19 | 0.707237 | 0.548772  | 0.845119 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.697628 | 0.529644  | 0.849802 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.642292 | 0.468254  | 0.796471 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_euclidean_s                 |  45 |             22 | 0.349802 | 0.196429  | 0.522646 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_euclidean_finetune          |  45 |             22 | 0.626482 | 0.451417  | 0.785721 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_probe    |  45 |             22 | 0.624506 | 0.449373  | 0.788    |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_s        |  45 |             22 | 0.626482 | 0.446     | 0.782628 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_finetune |  45 |             22 | 0.600791 | 0.423073  | 0.764018 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.644269 | 0.471975  | 0.796443 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_s              |  45 |             22 | 0.648221 | 0.475309  | 0.800396 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_finetune       |  45 |             22 | 0.616601 | 0.437988  | 0.775799 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_regress_d_probe             |  45 |             22 | 0.65415  | 0.479424  | 0.812266 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.642292 | 0.467581  | 0.795644 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | MCI_to_AD |               4 | contrastive_regress_d_finetune          |  45 |             22 | 0.646245 | 0.473673  | 0.809724 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | direct_logistic                         |  60 |              5 | 0.625455 | 0.479589  | 0.776786 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.378182 | 0.100381  | 0.697283 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.64     | 0.336651  | 0.901786 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_euclidean_finetune          |  60 |              5 | 0.581818 | 0.241696  | 0.881356 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.698182 | 0.465455  | 0.867925 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_s        |  60 |              5 | 0.363636 | 0.0862069 | 0.677966 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_finetune |  60 |              5 | 0.621818 | 0.436364  | 0.810909 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_probe          |  60 |              5 | 0.396364 | 0.105686  | 0.719298 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.396364 | 0.105686  | 0.719298 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_finetune       |  60 |              5 | 0.570909 | 0.255167  | 0.85228  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_regress_d_probe             |  60 |              5 | 0.541818 | 0.28655   | 0.770371 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.389091 | 0.105263  | 0.706897 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               2 | contrastive_regress_d_finetune          |  60 |              5 | 0.556364 | 0.241141  | 0.847458 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | direct_logistic                         |  49 |              9 | 0.530556 | 0.340558  | 0.724506 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_euclidean_probe             |  49 |              9 | 0.438889 | 0.240987  | 0.648752 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.572222 | 0.368868  | 0.766667 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_euclidean_finetune          |  49 |              9 | 0.527778 | 0.330524  | 0.72866  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_probe    |  49 |              9 | 0.536111 | 0.337186  | 0.740337 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.447222 | 0.244186  | 0.65247  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_finetune |  49 |              9 | 0.508333 | 0.312925  | 0.705742 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.436111 | 0.238095  | 0.640277 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_s              |  49 |              9 | 0.438889 | 0.24359   | 0.644444 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_finetune       |  49 |              9 | 0.530556 | 0.335871  | 0.733333 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_regress_d_probe             |  49 |              9 | 0.488889 | 0.325     | 0.658141 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.436111 | 0.238462  | 0.63957  |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               3 | contrastive_regress_d_finetune          |  49 |              9 | 0.558333 | 0.358968  | 0.748299 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | direct_logistic                         |  42 |              9 | 0.572391 | 0.360693  | 0.773045 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_euclidean_probe             |  42 |              9 | 0.424242 | 0.224479  | 0.625    |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_euclidean_s                 |  42 |              9 | 0.585859 | 0.387047  | 0.781308 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_euclidean_finetune          |  42 |              9 | 0.599327 | 0.371276  | 0.805413 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_probe    |  42 |              9 | 0.558923 | 0.343734  | 0.764706 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.427609 | 0.228739  | 0.627795 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.589226 | 0.356895  | 0.805405 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.424242 | 0.231085  | 0.624515 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_s              |  42 |              9 | 0.430976 | 0.231669  | 0.628724 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_finetune       |  42 |              9 | 0.56229  | 0.340522  | 0.772982 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_regress_d_probe             |  42 |              9 | 0.468013 | 0.290616  | 0.646494 |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.424242 | 0.228571  | 0.6245   |
| phase2_lora_adapted_embedding | lora_regress_d    | raw       | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.619529 | 0.402778  | 0.810216 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.668067 | 0.520703  | 0.808686 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.739496 | 0.605031  | 0.85819  |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_euclidean_s                 |  65 |             14 | 0.732493 | 0.598483  | 0.848709 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_euclidean_finetune          |  65 |             14 | 0.669468 | 0.501676  | 0.826599 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_probe    |  65 |             14 | 0.742297 | 0.610666  | 0.857295 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.735294 | 0.598039  | 0.853553 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_finetune |  65 |             14 | 0.717087 | 0.573892  | 0.844675 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_probe          |  65 |             14 | 0.742297 | 0.610969  | 0.860269 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.753501 | 0.623722  | 0.869904 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_finetune       |  65 |             14 | 0.714286 | 0.569522  | 0.846154 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.768908 | 0.636477  | 0.878676 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_regress_d_s                 |  65 |             14 | 0.746499 | 0.608484  | 0.860531 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               2 | contrastive_regress_d_finetune          |  65 |             14 | 0.683473 | 0.535354  | 0.814422 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.774671 | 0.637097  | 0.888158 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_euclidean_probe             |  51 |             19 | 0.810855 | 0.677778  | 0.917774 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_euclidean_s                 |  51 |             19 | 0.804276 | 0.669872  | 0.912459 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_euclidean_finetune          |  51 |             19 | 0.820724 | 0.695918  | 0.924194 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.833882 | 0.709676  | 0.9321   |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.830592 | 0.708483  | 0.928571 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.838816 | 0.720494  | 0.935855 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_probe          |  51 |             19 | 0.828947 | 0.701587  | 0.934928 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_s              |  51 |             19 | 0.833882 | 0.709519  | 0.936733 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_finetune       |  51 |             19 | 0.822368 | 0.700652  | 0.924245 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.832237 | 0.709223  | 0.933935 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_regress_d_s                 |  51 |             19 | 0.8125   | 0.685175  | 0.916129 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               3 | contrastive_regress_d_finetune          |  51 |             19 | 0.824013 | 0.702422  | 0.925932 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.687747 | 0.522625  | 0.843254 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.784585 | 0.63786   | 0.91502  |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_euclidean_s                 |  45 |             22 | 0.772727 | 0.618563  | 0.912017 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_euclidean_finetune          |  45 |             22 | 0.788538 | 0.637633  | 0.92068  |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_probe    |  45 |             22 | 0.770751 | 0.611336  | 0.909672 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_s        |  45 |             22 | 0.796443 | 0.651257  | 0.922925 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_finetune |  45 |             22 | 0.772727 | 0.617995  | 0.906    |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.782609 | 0.623987  | 0.920949 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_s              |  45 |             22 | 0.788538 | 0.631579  | 0.923881 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_finetune       |  45 |             22 | 0.790514 | 0.638889  | 0.921053 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_regress_d_probe             |  45 |             22 | 0.786561 | 0.63      | 0.920168 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.798419 | 0.650794  | 0.923077 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | MCI_to_AD |               4 | contrastive_regress_d_finetune          |  45 |             22 | 0.770751 | 0.614593  | 0.909101 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | direct_logistic                         |  60 |              5 | 0.425455 | 0.0657942 | 0.775862 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.810909 | 0.669215  | 0.927273 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.832727 | 0.687273  | 0.949153 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_euclidean_finetune          |  60 |              5 | 0.425455 | 0         | 0.888889 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.56     | 0.210526  | 0.964286 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_s        |  60 |              5 | 0.76     | 0.549708  | 0.919753 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_finetune |  60 |              5 | 0.421818 | 0.0400402 | 0.844828 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_probe          |  60 |              5 | 0.792727 | 0.642857  | 0.915254 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.781818 | 0.627119  | 0.918129 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_finetune       |  60 |              5 | 0.425455 | 0         | 0.888889 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_regress_d_probe             |  60 |              5 | 0.752727 | 0.596917  | 0.895062 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.778182 | 0.51521   | 0.955357 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               2 | contrastive_regress_d_finetune          |  60 |              5 | 0.505455 | 0.0935673 | 0.939655 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | direct_logistic                         |  49 |              9 | 0.477778 | 0.306169  | 0.648752 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_euclidean_probe             |  49 |              9 | 0.627778 | 0.424922  | 0.822222 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.636111 | 0.428228  | 0.833333 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_euclidean_finetune          |  49 |              9 | 0.55     | 0.367347  | 0.734694 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_probe    |  49 |              9 | 0.633333 | 0.460366  | 0.799154 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.638889 | 0.452961  | 0.820572 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_finetune |  49 |              9 | 0.638889 | 0.451282  | 0.820513 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.630556 | 0.445736  | 0.812826 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_s              |  49 |              9 | 0.622222 | 0.434749  | 0.804121 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_finetune       |  49 |              9 | 0.586111 | 0.394102  | 0.77551  |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_regress_d_probe             |  49 |              9 | 0.652778 | 0.484756  | 0.816327 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.655556 | 0.45237   | 0.847222 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               3 | contrastive_regress_d_finetune          |  49 |              9 | 0.619444 | 0.427778  | 0.8      |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | direct_logistic                         |  42 |              9 | 0.393939 | 0.2       | 0.598291 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_euclidean_probe             |  42 |              9 | 0.602694 | 0.385947  | 0.800926 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_euclidean_s                 |  42 |              9 | 0.619529 | 0.393371  | 0.821552 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_euclidean_finetune          |  42 |              9 | 0.474747 | 0.25641   | 0.6875   |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_probe    |  42 |              9 | 0.582492 | 0.399941  | 0.755982 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.636364 | 0.428072  | 0.819853 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.646465 | 0.403122  | 0.857215 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.616162 | 0.413479  | 0.796065 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_s              |  42 |              9 | 0.606061 | 0.402778  | 0.787037 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_finetune       |  42 |              9 | 0.505051 | 0.290435  | 0.703947 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_regress_d_probe             |  42 |              9 | 0.616162 | 0.452155  | 0.771883 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.639731 | 0.422267  | 0.838235 |
| phase2_lora_adapted_embedding | lora_euclidean    | raw       | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.59596  | 0.385959  | 0.790625 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | direct_logistic                         |  65 |             14 | 0.635854 | 0.482654  | 0.785508 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_euclidean_probe             |  65 |             14 | 0.718487 | 0.566032  | 0.856072 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_euclidean_s                 |  65 |             14 | 0.289916 | 0.146453  | 0.448179 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_euclidean_finetune          |  65 |             14 | 0.676471 | 0.516348  | 0.816038 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_probe    |  65 |             14 | 0.736695 | 0.605262  | 0.85503  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_s        |  65 |             14 | 0.767507 | 0.646226  | 0.872551 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_rank_kendall_basic_finetune |  65 |             14 | 0.670868 | 0.524492  | 0.802525 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_probe          |  65 |             14 | 0.753501 | 0.623722  | 0.867935 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_s              |  65 |             14 | 0.738095 | 0.602682  | 0.857143 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_hybrid_basic_finetune       |  65 |             14 | 0.655462 | 0.494545  | 0.803336 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_regress_d_probe             |  65 |             14 | 0.731092 | 0.597484  | 0.849503 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_regress_d_s                 |  65 |             14 | 0.736695 | 0.605025  | 0.850909 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               2 | contrastive_regress_d_finetune          |  65 |             14 | 0.676471 | 0.531894  | 0.808082 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | direct_logistic                         |  51 |             19 | 0.792763 | 0.656452  | 0.905556 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_euclidean_probe             |  51 |             19 | 0.787829 | 0.652027  | 0.906452 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_euclidean_s                 |  51 |             19 | 0.228618 | 0.106481  | 0.372325 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_euclidean_finetune          |  51 |             19 | 0.787829 | 0.649659  | 0.901165 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_probe    |  51 |             19 | 0.838816 | 0.721429  | 0.932567 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_s        |  51 |             19 | 0.822368 | 0.697368  | 0.927276 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_rank_kendall_basic_finetune |  51 |             19 | 0.763158 | 0.625     | 0.884111 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_probe          |  51 |             19 | 0.817434 | 0.692778  | 0.921541 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_s              |  51 |             19 | 0.796053 | 0.664358  | 0.905358 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_hybrid_basic_finetune       |  51 |             19 | 0.789474 | 0.657393  | 0.90002  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_regress_d_probe             |  51 |             19 | 0.822368 | 0.697527  | 0.925    |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_regress_d_s                 |  51 |             19 | 0.774671 | 0.638044  | 0.891308 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               3 | contrastive_regress_d_finetune          |  51 |             19 | 0.786184 | 0.655701  | 0.9      |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | direct_logistic                         |  45 |             22 | 0.671937 | 0.502154  | 0.827381 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_euclidean_probe             |  45 |             22 | 0.743083 | 0.584016  | 0.8826   |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_euclidean_s                 |  45 |             22 | 0.266798 | 0.127151  | 0.426589 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_euclidean_finetune          |  45 |             22 | 0.731225 | 0.56746   | 0.870372 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_probe    |  45 |             22 | 0.756917 | 0.593117  | 0.896    |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_s        |  45 |             22 | 0.768775 | 0.610584  | 0.909672 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_rank_kendall_basic_finetune |  45 |             22 | 0.725296 | 0.559988  | 0.870445 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_probe          |  45 |             22 | 0.774704 | 0.619048  | 0.906749 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_s              |  45 |             22 | 0.752964 | 0.59669   | 0.892018 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_hybrid_basic_finetune       |  45 |             22 | 0.727273 | 0.567227  | 0.869565 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_regress_d_probe             |  45 |             22 | 0.747036 | 0.582996  | 0.892857 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_regress_d_s                 |  45 |             22 | 0.733202 | 0.568805  | 0.880567 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | MCI_to_AD |               4 | contrastive_regress_d_finetune          |  45 |             22 | 0.735178 | 0.572853  | 0.875494 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | direct_logistic                         |  60 |              5 | 0.341818 | 0.08      | 0.683501 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_euclidean_probe             |  60 |              5 | 0.829091 | 0.689739  | 0.949153 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_euclidean_s                 |  60 |              5 | 0.214545 | 0.0611656 | 0.37926  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_euclidean_finetune          |  60 |              5 | 0.389091 | 0.0401786 | 0.788569 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_probe    |  60 |              5 | 0.596364 | 0.266935  | 0.922283 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_s        |  60 |              5 | 0.727273 | 0.450893  | 0.927262 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_rank_kendall_basic_finetune |  60 |              5 | 0.4      | 0.0948276 | 0.759185 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_probe          |  60 |              5 | 0.709091 | 0.436619  | 0.923361 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_s              |  60 |              5 | 0.72     | 0.431034  | 0.939324 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_hybrid_basic_finetune       |  60 |              5 | 0.381818 | 0.0464059 | 0.789474 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_regress_d_probe             |  60 |              5 | 0.745455 | 0.422795  | 0.963636 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_regress_d_s                 |  60 |              5 | 0.730909 | 0.3805    | 0.965392 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               2 | contrastive_regress_d_finetune          |  60 |              5 | 0.501818 | 0.193758  | 0.832727 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | direct_logistic                         |  49 |              9 | 0.488889 | 0.307619  | 0.668219 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_euclidean_probe             |  49 |              9 | 0.647222 | 0.446539  | 0.837375 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_euclidean_s                 |  49 |              9 | 0.363889 | 0.181159  | 0.553867 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_euclidean_finetune          |  49 |              9 | 0.458333 | 0.275184  | 0.65     |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_probe    |  49 |              9 | 0.655556 | 0.480849  | 0.819459 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_s        |  49 |              9 | 0.677778 | 0.502767  | 0.844961 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_rank_kendall_basic_finetune |  49 |              9 | 0.55     | 0.353741  | 0.751701 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_probe          |  49 |              9 | 0.641667 | 0.466505  | 0.816344 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_s              |  49 |              9 | 0.652778 | 0.466659  | 0.82622  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_hybrid_basic_finetune       |  49 |              9 | 0.527778 | 0.327273  | 0.733333 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_regress_d_probe             |  49 |              9 | 0.661111 | 0.490833  | 0.817847 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_regress_d_s                 |  49 |              9 | 0.658333 | 0.452381  | 0.845455 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               3 | contrastive_regress_d_finetune          |  49 |              9 | 0.527778 | 0.346812  | 0.709101 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | direct_logistic                         |  42 |              9 | 0.400673 | 0.191837  | 0.62505  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_euclidean_probe             |  42 |              9 | 0.62963  | 0.422205  | 0.821875 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_euclidean_s                 |  42 |              9 | 0.37037  | 0.189815  | 0.578395 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_euclidean_finetune          |  42 |              9 | 0.390572 | 0.163814  | 0.618802 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_probe    |  42 |              9 | 0.62963  | 0.428571  | 0.811448 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_s        |  42 |              9 | 0.676768 | 0.488803  | 0.837844 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_rank_kendall_basic_finetune |  42 |              9 | 0.508418 | 0.285714  | 0.714353 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_probe          |  42 |              9 | 0.636364 | 0.441176  | 0.808196 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_s              |  42 |              9 | 0.646465 | 0.448105  | 0.824048 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_hybrid_basic_finetune       |  42 |              9 | 0.43771  | 0.215488  | 0.64984  |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_regress_d_probe             |  42 |              9 | 0.612795 | 0.434375  | 0.779615 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_regress_d_s                 |  42 |              9 | 0.643098 | 0.432583  | 0.829102 |
| phase2_lora_adapted_embedding | lora_hybrid_basic | raw       | CN_to_MCI |               4 | contrastive_regress_d_finetune          |  42 |              9 | 0.481481 | 0.268382  | 0.684219 |