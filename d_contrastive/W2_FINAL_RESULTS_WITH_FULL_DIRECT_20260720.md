# W2 Final Results With Full Fine-Tune and Direct LoRA

Generated 2026-07-20 from completed jobs `34087065` and `34087066`. Age/APOE not included.

## Completion status

| item                                | status   | path                                         |
|:------------------------------------|:---------|:---------------------------------------------|
| Phase 0 d-relation                  | done     | results_w2_phase0                            |
| Phase 1 frozen diagnosis/conversion | done     | results_w2_phase1_dx, results_w2_phase1_conv |
| Phase 1c d-as-support               | done     | results_w2_phase1_dsupport                   |
| Phase 2 LoRA d-adapted encoder      | done     | results_w2_phase2/lora_*                     |
| Phase 2 direct downstream LoRA      | done     | results_w2_direct_lora_downstream            |
| Phase 2 full fine-tune retry        | done     | results_w2_phase2/full_*                     |
| Age/APOE                            | deferred | not included                                 |

## Direct downstream LoRA results

| task_kind   | task      |   horizon_years |   n |   n_positive | method                 |      auc |    ci_lo |    ci_hi |
|:------------|:----------|----------------:|----:|-------------:|:-----------------------|---------:|---------:|---------:|
| dx_binary   | CN_vs_AD  |             nan | 352 |          102 | direct_lora_downstream | 0.896863 | 0.863    | 0.928079 |
| dx_binary   | CN_vs_MCI |             nan | 509 |          259 | direct_lora_downstream | 0.564958 | 0.515554 | 0.613795 |
| dx_binary   | MCI_vs_AD |             nan | 361 |          102 | direct_lora_downstream | 0.811644 | 0.76305  | 0.857201 |
| conversion  | MCI_to_AD |               2 |  65 |           14 | direct_lora_downstream | 0.623249 | 0.459948 | 0.766133 |
| conversion  | MCI_to_AD |               3 |  51 |           19 | direct_lora_downstream | 0.684211 | 0.528213 | 0.821549 |
| conversion  | MCI_to_AD |               4 |  45 |           22 | direct_lora_downstream | 0.596838 | 0.422604 | 0.766798 |
| conversion  | CN_to_MCI |               2 |  60 |            5 | direct_lora_downstream | 0.654545 | 0.392059 | 0.894166 |
| conversion  | CN_to_MCI |               3 |  49 |            9 | direct_lora_downstream | 0.422222 | 0.232558 | 0.619644 |
| conversion  | CN_to_MCI |               4 |  42 |            9 | direct_lora_downstream | 0.414141 | 0.213675 | 0.628724 |

## Best diagnosis by encoder/training stage

| stage                  | kind   | task      | horizon   | method                               |   n |   pos |      auc |
|:-----------------------|:-------|:----------|:----------|:-------------------------------------|----:|------:|---------:|
| lora_regress_d         | dx     | CN_MCI_AD |           | baseline_raw                         | 611 |   nan | 0.697713 |
| full_regress_d         | dx     | CN_MCI_AD |           | baseline_raw                         | 611 |   nan | 0.688615 |
| lora_hybrid_basic      | dx     | CN_MCI_AD |           | contrastive_euclidean_probe          | 611 |   nan | 0.664532 |
| lora_euclidean         | dx     | CN_MCI_AD |           | baseline_raw                         | 611 |   nan | 0.660599 |
| full_euclidean         | dx     | CN_MCI_AD |           | baseline_raw                         | 611 |   nan | 0.660434 |
| full_hybrid_basic      | dx     | CN_MCI_AD |           | baseline_raw                         | 611 |   nan | 0.620197 |
| lora_regress_d         | dx     | CN_vs_AD  |           | contrastive_rank_kendall_basic_probe | 352 |   102 | 0.909412 |
| direct_lora_downstream | dx     | CN_vs_AD  |           | direct_lora_downstream               | 352 |   102 | 0.896863 |
| full_regress_d         | dx     | CN_vs_AD  |           | contrastive_hybrid_basic_probe       | 352 |   102 | 0.874902 |
| lora_hybrid_basic      | dx     | CN_vs_AD  |           | baseline_raw                         | 352 |   102 | 0.871451 |
| lora_euclidean         | dx     | CN_vs_AD  |           | baseline_raw                         | 352 |   102 | 0.863765 |
| full_euclidean         | dx     | CN_vs_AD  |           | baseline_raw                         | 352 |   102 | 0.85949  |
| full_hybrid_basic      | dx     | CN_vs_AD  |           | baseline_raw                         | 352 |   102 | 0.776353 |
| lora_regress_d         | dx     | CN_vs_MCI |           | baseline_raw                         | 509 |   259 | 0.635876 |
| full_regress_d         | dx     | CN_vs_MCI |           | contrastive_rank_kendall_basic_probe | 509 |   259 | 0.625375 |
| lora_hybrid_basic      | dx     | CN_vs_MCI |           | contrastive_euclidean_probe          | 509 |   259 | 0.610672 |
| full_euclidean         | dx     | CN_vs_MCI |           | ridge_dhat_all                       | 509 |   259 | 0.599212 |
| lora_euclidean         | dx     | CN_vs_MCI |           | ridge_dhat_all                       | 509 |   259 | 0.58939  |
| direct_lora_downstream | dx     | CN_vs_MCI |           | direct_lora_downstream               | 509 |   259 | 0.564958 |
| full_hybrid_basic      | dx     | CN_vs_MCI |           | ridge_dhat_all                       | 509 |   259 | 0.558826 |
| direct_lora_downstream | dx     | MCI_vs_AD |           | direct_lora_downstream               | 361 |   102 | 0.811644 |
| lora_regress_d         | dx     | MCI_vs_AD |           | contrastive_regress_d_probe          | 361 |   102 | 0.79003  |
| full_regress_d         | dx     | MCI_vs_AD |           | contrastive_hybrid_basic_probe       | 361 |   102 | 0.770611 |
| lora_hybrid_basic      | dx     | MCI_vs_AD |           | ridge_dhat_finetune                  | 361 |   102 | 0.765766 |
| full_euclidean         | dx     | MCI_vs_AD |           | baseline_raw                         | 361 |   102 | 0.761753 |
| lora_euclidean         | dx     | MCI_vs_AD |           | baseline_raw                         | 361 |   102 | 0.752782 |
| full_hybrid_basic      | dx     | MCI_vs_AD |           | baseline_raw                         | 361 |   102 | 0.723257 |

## Best conversion by encoder/training stage

| stage                  | kind   | task      |   horizon | method                                  |   n |   pos |      auc |
|:-----------------------|:-------|:----------|----------:|:----------------------------------------|----:|------:|---------:|
| full_regress_d         | conv   | CN_to_MCI |         2 | contrastive_hybrid_basic_probe          |  60 |     5 | 0.887273 |
| full_hybrid_basic      | conv   | CN_to_MCI |         2 | ridge_dhat_all                          |  60 |     5 | 0.883636 |
| lora_euclidean         | conv   | CN_to_MCI |         2 | contrastive_euclidean_s                 |  60 |     5 | 0.832727 |
| lora_hybrid_basic      | conv   | CN_to_MCI |         2 | contrastive_euclidean_probe             |  60 |     5 | 0.829091 |
| full_euclidean         | conv   | CN_to_MCI |         2 | ridge_dhat_all                          |  60 |     5 | 0.778182 |
| lora_regress_d         | conv   | CN_to_MCI |         2 | contrastive_rank_kendall_basic_probe    |  60 |     5 | 0.698182 |
| direct_lora_downstream | conv   | CN_to_MCI |         2 | direct_lora_downstream                  |  60 |     5 | 0.654545 |
| full_regress_d         | conv   | CN_to_MCI |         3 | contrastive_euclidean_s                 |  49 |     9 | 0.794444 |
| lora_euclidean         | conv   | CN_to_MCI |         3 | ridge_dhat_finetune                     |  49 |     9 | 0.716667 |
| lora_hybrid_basic      | conv   | CN_to_MCI |         3 | ridge_dhat_all                          |  49 |     9 | 0.683333 |
| full_euclidean         | conv   | CN_to_MCI |         3 | contrastive_euclidean_probe             |  49 |     9 | 0.677778 |
| full_hybrid_basic      | conv   | CN_to_MCI |         3 | contrastive_rank_kendall_basic_s        |  49 |     9 | 0.652778 |
| lora_regress_d         | conv   | CN_to_MCI |         3 | contrastive_euclidean_s                 |  49 |     9 | 0.572222 |
| direct_lora_downstream | conv   | CN_to_MCI |         3 | direct_lora_downstream                  |  49 |     9 | 0.422222 |
| full_regress_d         | conv   | CN_to_MCI |         4 | contrastive_euclidean_s                 |  42 |     9 | 0.791246 |
| lora_euclidean         | conv   | CN_to_MCI |         4 | ridge_dhat_all                          |  42 |     9 | 0.69697  |
| lora_hybrid_basic      | conv   | CN_to_MCI |         4 | ridge_dhat_finetune                     |  42 |     9 | 0.680135 |
| full_hybrid_basic      | conv   | CN_to_MCI |         4 | contrastive_rank_kendall_basic_s        |  42 |     9 | 0.6633   |
| full_euclidean         | conv   | CN_to_MCI |         4 | contrastive_euclidean_probe             |  42 |     9 | 0.653199 |
| lora_regress_d         | conv   | CN_to_MCI |         4 | contrastive_regress_d_finetune          |  42 |     9 | 0.619529 |
| direct_lora_downstream | conv   | CN_to_MCI |         4 | direct_lora_downstream                  |  42 |     9 | 0.414141 |
| full_hybrid_basic      | conv   | MCI_to_AD |         2 | contrastive_hybrid_basic_probe          |  65 |    14 | 0.784314 |
| lora_euclidean         | conv   | MCI_to_AD |         2 | contrastive_regress_d_probe             |  65 |    14 | 0.768908 |
| lora_hybrid_basic      | conv   | MCI_to_AD |         2 | contrastive_rank_kendall_basic_s        |  65 |    14 | 0.767507 |
| full_euclidean         | conv   | MCI_to_AD |         2 | ridge_dhat_finetune                     |  65 |    14 | 0.756303 |
| full_regress_d         | conv   | MCI_to_AD |         2 | contrastive_rank_kendall_basic_s        |  65 |    14 | 0.740896 |
| lora_regress_d         | conv   | MCI_to_AD |         2 | direct_logistic                         |  65 |    14 | 0.722689 |
| direct_lora_downstream | conv   | MCI_to_AD |         2 | direct_lora_downstream                  |  65 |    14 | 0.623249 |
| lora_euclidean         | conv   | MCI_to_AD |         3 | contrastive_rank_kendall_basic_finetune |  51 |    19 | 0.838816 |
| lora_hybrid_basic      | conv   | MCI_to_AD |         3 | contrastive_rank_kendall_basic_probe    |  51 |    19 | 0.838816 |
| full_regress_d         | conv   | MCI_to_AD |         3 | contrastive_rank_kendall_basic_s        |  51 |    19 | 0.835526 |
| full_euclidean         | conv   | MCI_to_AD |         3 | ridge_dhat_all                          |  51 |    19 | 0.817434 |
| full_hybrid_basic      | conv   | MCI_to_AD |         3 | contrastive_hybrid_basic_probe          |  51 |    19 | 0.800987 |
| lora_regress_d         | conv   | MCI_to_AD |         3 | direct_logistic                         |  51 |    19 | 0.726974 |
| direct_lora_downstream | conv   | MCI_to_AD |         3 | direct_lora_downstream                  |  51 |    19 | 0.684211 |
| lora_euclidean         | conv   | MCI_to_AD |         4 | ridge_dhat_all                          |  45 |    22 | 0.802372 |
| lora_hybrid_basic      | conv   | MCI_to_AD |         4 | contrastive_hybrid_basic_probe          |  45 |    22 | 0.774704 |
| full_euclidean         | conv   | MCI_to_AD |         4 | contrastive_rank_kendall_basic_probe    |  45 |    22 | 0.764822 |
| full_regress_d         | conv   | MCI_to_AD |         4 | contrastive_rank_kendall_basic_s        |  45 |    22 | 0.754941 |
| full_hybrid_basic      | conv   | MCI_to_AD |         4 | ridge_dhat_finetune                     |  45 |    22 | 0.745059 |
| lora_regress_d         | conv   | MCI_to_AD |         4 | direct_logistic                         |  45 |    22 | 0.697628 |
| direct_lora_downstream | conv   | MCI_to_AD |         4 | direct_lora_downstream                  |  45 |    22 | 0.596838 |

## Read

- Direct downstream LoRA is now done for all requested diagnosis and conversion tasks. It is strong for CN/AD and MCI/AD, but weak for conversion relative to d-supervised LoRA/full embeddings.

- Full fine-tune is now done for regress_d, euclidean, and hybrid_basic. Full-regress_d improves diagnosis relative to frozen, but does not beat LoRA-regress_d. Full-hybrid/full-regress show high CN->MCI numbers, but CN->MCI has only 5-9 converters, so treat that as exploratory.

- The most reliable diagnosis result remains LoRA-regress_d. The most reliable conversion pattern remains geometric/rank objectives, especially LoRA-euclidean/hybrid for MCI->AD and full/LoRA variants for CN->MCI with caveat on small n.
