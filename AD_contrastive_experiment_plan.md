# AD Contrastive — Master Experiment Plan (for Claude Code)

All experiments run on **frozen Swin 768-d embeddings**. The Swin encoder is
never trained or touched in any experiment below. This document is the single
source of truth: global rules + one shared metric definition + 5 experiments
each with full instructions, then the execution order.

---

## 0. Shared context (paste into every experiment prompt)

```
Working dir: /dcs07/zwang/data/adni_d/d_contrastive/
Embeddings:  data/embeddings_128_05152016/swin_latent.npy      (3019 x 768)
             data/embeddings_128_05152016/image_id_order.npy
             data/embeddings_128_05152016/swin_combat_train.npy  (2408)
             data/embeddings_128_05152016/swin_combat_test.npy   (611)
             data/embeddings_128_05152016/swin_combat_train_ids.npy
             data/embeddings_128_05152016/swin_combat_test_ids.npy
Master meta: D_with_image_paths.csv
             columns include: RID, image_id, dx, d_mod3, visit date,
             conversion labels (MCI->AD 2y, CN->MCI 2y)
Split:       by RID, no overlap. Train RIDs 954 (2408 img), Test RIDs 239 (611 img).
Main script: scripts/minimal_v0_contrastive.py
Conversion eval + bootstrap: scripts/diagnose_cn_mci_conversion.py
Reference results to stay comparable to:
  results_plain_euclidean_05152016_s60_c150_h384_l256_cls_tasks/
  rank_sweep_with_ml_baselines.csv
```

## 1. Global hard rules (apply to ALL experiments)

```
- Frozen Swin 768-d only. NEVER retrain / finetune / LoRA the Swin encoder
  in any experiment in this document. (LoRA is a separate later decision,
  gated on EXP-0 + EXP-1 outcomes — see section 9.)
- Reuse the EXISTING Stage-2 eval harness and the bootstrap from
  diagnose_cn_mci_conversion.py UNCHANGED, so every number is directly
  comparable across experiments and to the reference dirs above.
- Run every trained-model experiment on BOTH raw and combat embeddings.
- >= 5 seeds for anything that involves training an MLP/head. Report
  mean +/- std across seeds. Single-seed numbers are not trusted because
  test converters are tiny (MCI->AD = 14, CN->MCI = 5).
- Apply the EXP-2 follow-up censoring (section 4) to the conversion labels
  BEFORE computing any conversion metric in EXP-0/1/3/4. Conversion numbers
  computed on uncensored labels are not valid.
- Optimizer (unless stated otherwise): AdamW lr=1e-3 wd=1e-4, grad clip 1.0,
  batch=128, contrastive/pretrain epochs=150, supervised-MLP epochs=60,
  tau=1.0, beta=1.0. NO uniformity loss anywhere.
- SELECTION METRIC for any method-characterization run = d_mod3 Spearman on
  full test (611), tie-broken by d_mod3 R^2. Conversion AUC is DIRECTIONAL
  ONLY (too few converters) — report it with bootstrap CI, never select on it.
```

## 2. Unified metric block (the "what metrics" answer — emit for EVERY run)

Every experiment that trains/probes a model must output this exact metric
block per config (and per raw/combat), as `mean +/- std` over the seeds:

```
PRIMARY (reliable, full test = 611):
  d_mod3 R^2
  d_mod3 Pearson
  d_mod3 Spearman          <- selection metric

GEOMETRY / HEALTH DIAGNOSTICS:
  z.std (test)
  # active dims (std > 0.05) out of latent dim
  alignment Spearman( ||z - centroid||, d_mod3 )   on test
  final soft-tau value (train + test)   [only for runs with a rank head]

CURRENT-STATE CLASSIFICATION (AUC):
  CN/AD, CN/MCI, MCI/AD, 3-class macro AUC

FUTURE CONVERSION (DIRECTIONAL ONLY — always with bootstrap CI):
  MCI->AD 2y AUC  + 95% bootstrap CI + n converters
  CN->MCI 2y AUC  + 95% bootstrap CI + n converters
  (MCI->AD CI is currently MISSING in old runs — add it everywhere.)

SETUPS (where applicable, same definitions as before):
  Setup1 = direct LR/Ridge on input features
  Setup2 = supervised MLP trained per task
  Setup3 = pretrain, freeze trunk, linear/Ridge probe   <- key eval
  Setup4 = pretrain, then supervised finetune
```

Each experiment below only adds task-specific extras on top of this block.

---

## 3. EXP-0 — d-hat-as-conversion baseline + oracle  [HIGHEST PRIORITY, no retrain]

**Purpose:** decide whether the conversion advantage comes from *contrastive
geometry* or simply from *dense continuous supervision* replacing sparse
conversion labels. This determines the paper's central claim. Do this FIRST.

```
GOAL
Test whether a simple Ridge regression to d_mod3 (no contrastive, no MLP
pretrain) predicts conversion as well as the contrastive latent, and where the
ceiling is.

PROCEDURE (no model training beyond a Ridge fit)
1. Ridge-d-hat predictor:
     raw 768 -> Ridge (fit on TRAIN to predict d_mod3) -> d_hat on TEST
     Use d_hat directly as the conversion SCORE -> conversion AUC.
     (Repeat on combat features.)
2. Oracle predictor (upper bound):
     Use the TRUE test d_mod3 as the conversion score -> conversion AUC.
     This is the ceiling that any d_mod3-derived method can reach.
3. Sparse-label baseline (lower bound):
     direct Logistic on raw 768 trained on the (sparse) conversion labels
     -> conversion AUC. (Already ~0.644 / 0.317; recompute under censoring.)
4. Pull in the existing contrastive z Setup3 conversion AUCs for comparison.

OUTPUT (conversion tasks only; both raw + combat; 5 Ridge seeds for stability):
  A single table, rows = {oracle true-d_mod3, Ridge d_hat, direct logistic,
  contrastive z Setup3}, columns = MCI->AD AUC [CI], CN->MCI AUC [CI],
  n converters. Plus d_mod3 R^2/Spearman for the Ridge d_hat row.
  File: exp0_dhat_vs_contrastive_conversion.csv

INTERPRETATION TO PRINT:
  - contrastive >> Ridge d_hat  -> geometry adds value, "contrastive" claim.
  - contrastive ~= Ridge d_hat >> logistic -> the win is DENSE SUPERVISION,
    not contrastive; reframe headline to "dense progression supervision beats
    sparse conversion labels", contrastive is one implementation.
  - all ~= logistic -> conversion advantage is noise; story collapses.
```

## 4. EXP-2 — Statistical power hardening  [no retrain — it is an eval-protocol fix]

> Numbered EXP-2 but logically a prerequisite. Implement once as shared eval
> utilities, then EXP-0/1/3/4 all call it. Nothing here trains a model.

```
GOAL
Make every conversion number defensible: correct labels, CIs on both tasks,
seed stability.

1. FOLLOW-UP CENSORING (label fix — the reviewer-critical one):
   A subject labeled "non-converter" is only valid if they have >= 2 years
   of adequate follow-up after baseline. Subjects who dropped out before the
   2y window must be EXCLUDED, not counted as non-converters.
   - Recompute MCI->AD 2y and CN->MCI 2y cohorts under this rule.
   - Report new n / n-converters / n-excluded per task, train and test.
   - Save the censored cohort definition to:
       conversion_cohorts_censored.csv
   - ALL conversion metrics in every other experiment use these cohorts.

2. BOOTSTRAP CI ON BOTH TASKS:
   - Add the bootstrap (same routine as diagnose_cn_mci_conversion.py) to
     MCI->AD 2y as well (currently only CN->MCI has it).
   - Report paired delta CI vs the Setup1 baseline for each method.

3. SEED STABILITY:
   - >= 5 seeds for every trained config; report conversion AUC mean +/- std
     and point-estimate range, so we can see whether a 0.77-vs-0.77 gap is
     within seed noise.

OUTPUT:
  exp2_censoring_report.csv (cohort counts before/after censoring)
  Updated CIs folded into every downstream table.
  A note stating, for each conversion task, whether ANY method pair is
  statistically separable after censoring + CI (likely: NOT, given 14/5).
```

## 5. EXP-1 — Method 2 dx-pretrain baseline  [CRITICAL, decision-tree input, 1-2 day]

**Purpose:** the missing fair comparison. NOT the "pure supervised MLP" already
in the table (that was per-task direct supervision). This is *pretrain with
diagnosis CE, then freeze + probe* — same protocol as contrastive, swapping
only the pretrain objective. Answers: does the contrastive objective beat a
supervised-dx objective on conversion?

```
GOAL
Pretrain the SAME trunk with a 3-way CN/MCI/AD cross-entropy objective (instead
of Euclidean Y-Aware contrastive), then evaluate under the identical Stage-2
protocol, so contrastive vs supervised-dx is an apples-to-apples comparison.

ARCHITECTURE (identical to the contrastive trunk):
  768 -> 384 -> 256 (BatchNorm, no final ReLU) -> z
  Pretrain head: Linear(256, 3) -> CN/MCI/AD logits, CE loss.

PROTOCOL:
  - Pretrain 150 epochs, same optimizer as global rules.
  - Setup3-analog: freeze trunk, linear/Ridge probe per task (KEY).
  - Setup4-analog: supervised finetune from dx-pretrain init.
  - raw + combat, >= 5 seeds.

OUTPUT (full unified metric block, section 2), plus a head-to-head table:
  exp1_method2_dx_pretrain.csv
  Direct comparison table: rows = {Euclidean contrastive (existing),
  dx-pretrain (this), Pure ML direct LR (existing)}, columns = full metric set.

INTERPRETATION TO PRINT (this is the decision tree):
  - contrastive conversion >> dx-pretrain conversion (+0.05+) -> Scenario A:
    contrastive genuinely adds value over supervised pretrain; claim solid;
    LoRA push justified; aim MICCAI main venue.
  - contrastive ~= dx-pretrain on conversion -> Scenario B: method adds value
    vs raw features but not specifically over supervised pretrain; workshop
    venue; no LoRA needed.
  - dx-pretrain > contrastive -> Scenario C: contrastive does not add value;
    pivot to honest benchmark framing.
```

## 6. EXP-3 — Hybrid geometry-vs-progression trade-off  [SECONDARY / parallel, ~1 day]

**Purpose:** characterize the collapse<->conversion tension and test whether
DECOUPLING the rank head from the euclidean trunk breaks the monotone
trade-off. Method-characterization run, NOT a leaderboard run. Selection on
d_mod3 Spearman, never on conversion AUC.

```
ARCHITECTURE — two variants, shared front 768 -> 384 (BN, ReLU):
  COUPLED (control):
    384 -> 256 (BN, no final ReLU) -> z         # euclidean Y-Aware on z
    Linear(256,1) -> s                           # soft Kendall on the SAME z
  DECOUPLED (the test):
    Branch A: 384 -> 256 (BN, no final ReLU) -> z_geo   # euclidean only
    Branch B: 384 -> 64  (BN, no final ReLU) -> z_rank
              Linear(64,1) -> s                         # rank only
    euclidean loss touches z_geo ONLY; rank loss touches branch B ONLY.

LOSS:
  loss = euclidean_yaware(z or z_geo, d_mod3)
         + lambda * soft_kendall(s, d_mod3, alpha)
  soft_kendall(s,d,alpha): off-diagonal mean of
      tanh(alpha*(s_i - s_j)) * sign(d_i - d_j), negated. No grad through d.

SWEEP:
  variant : {coupled, decoupled}
  lambda  : {0.0, 0.1, 0.3, 0.6, 1.0, 2.0}   # 0.0 = pure euclidean control
  alpha   : {10}   (add 20 only if time permits)
  5 seeds, raw + combat.

STAGE-2 PROBES:
  DECOUPLED: (a) z_geo alone  (b) concat[z_geo, z_rank]  (c) concat[z_geo, s]
  COUPLED:   z alone + concat[z, s]

OUTPUT:
  exp3_hybrid_tradeoff_all_results.csv   (config x seed x raw/combat)
  exp3_hybrid_tradeoff_summary.csv       (mean +/- std over seeds)
  PLOT exp3_hybrid_tradeoff_curve.(png/pdf):
     x = z.std (geometry-health axis)
     y = d_mod3 Spearman (PRIMARY), MCI->AD AUC, CN->MCI AUC (directional, CIs)
     coupled vs decoupled as two line styles; mark lambda=0 (euclidean) and
     rank-only (from rank_sweep_with_ml_baselines.csv) as anchors.
  Consolidated table: best decoupled vs best coupled vs euclidean(lambda=0).

INTERPRETATION TO PRINT:
  - trade-off MONOTONE (more z.std -> worse conversion) -> clean negative
    result; collapse is incidental-but-harmless; euclidean stays main method.
  - DECOUPLED finds a config with HEALTHY z.std AND d_mod3 Spearman >=
    euclidean AND conversion not worse -> flag loudly; candidate main-method
    upgrade. (Prior ~25% odds.)
```

## 7. EXP-4 — Conversion task suite expansion  [no retrain, 1-2 day]

**Purpose:** the originally-planned variants that were never run correctly
(last attempt accidentally produced current-dx tasks). Longer windows also
*increase converter counts*, which directly helps the EXP-2 power problem.
Run all on the existing trained models (euclidean Setup3 + best hybrid),
under EXP-2 censoring.

```
ADD these tasks (eval-only, reuse trained latents):
  1. MCI->AD 1y / 3y / 4y conversion   (more converters in longer windows)
  2. CN->MCI 3y / 4y conversion         (mitigates the N=5 problem)
  3. Stable MCI vs Progressive MCI (3y binary)  -- standard clinical task
  4. Time-to-conversion regression (continuous): among converters only,
     predict days-baseline-to-AD-dx; report R^2 / Spearman. (Not limited by
     the converter-count problem the same way binary AUC is.)

FOR EACH new task report (under censoring, both raw + combat):
  - For binary tasks: AUC + bootstrap CI + n converters (Setup1 vs Setup3).
  - For time-to-conversion: R^2, Pearson, Spearman.
  - Whether the method's conversion advantage holds ACROSS windows (robustness)
    or only at one window (report which, with caveat).

OUTPUT:
  exp4_conversion_task_suite.csv
  A robustness note: is the contrastive>baseline gap consistent across
  1y/2y/3y/4y, or window-specific?
```

## 8. ACTION (non-compute) — OASIS-3 access  [do TODAY, 1 hour]

```
- Apply at https://www.oasis-brains.org/  (free, 1-2 month wait).
- This is the ONLY thing that solves the root power problem (5 test
  converters for CN->MCI). EXP-2/4 make the existing numbers honest but
  cannot manufacture statistical power. Start the clock now; it is the
  longest pole and zero compute cost.
- Downstream (after access): apply the same frozen-Swin pipeline to OASIS-3,
  extract 768-d, run the frozen contrastive method (no retrain), do
  cross-cohort CN->MCI external validation.
```

---

## 9. Execution order (GPU scheduling + gating)

```
TODAY, zero compute:
  [ ] Submit OASIS-3 application (section 8). Do not delay.

WAVE 1 — claim-defining, mostly no-retrain, run first:
  [ ] EXP-2 censoring utilities (no train) -- prerequisite for all conversion #s
  [ ] EXP-0 d-hat baseline (no train, Ridge only) -- defines paper claim
  [ ] EXP-1 Method 2 dx-pretrain (TRAINS; ~1-2 day) -- decision-tree input
  These three answer: (a) is the win geometry or dense supervision? (b) does
  contrastive beat supervised-dx? Paper claim + venue scenario fall out of
  EXP-0 + EXP-1 together.

WAVE 2 — runs in PARALLEL with Wave 1 if GPU available, else after:
  [ ] EXP-4 conversion task suite (no retrain; reuses trained latents)
  [ ] EXP-3 hybrid trade-off (TRAINS; ~1 day) -- secondary characterization

GATED (do NOT start until Wave 1 results are in):
  [ ] LoRA on Swin -- ONLY if EXP-0 shows contrastive > d_hat AND EXP-1 shows
      contrastive > dx-pretrain on conversion (Scenario A). Otherwise the
      finding is not strong enough to justify 2-3 weeks + overfit risk on
      1709 effective images. Order matters: confirm the finding before
      polishing the number.

If only one GPU-day is available: EXP-0 + EXP-1 first. EXP-3/4 can wait until
the claim is fixed.
```

## 10. Standing decisions (do NOT redo)

```
- Main contrastive loss = Euclidean Y-Aware. Locked. (cosine / LayerNorm /
  uniformity all tried and worse; rank/hybrid ablated, comparable-not-better.)
- Collapse line is CLOSED: rank-only gives z.std ~0.94 / 256 active dims but
  WORSE conversion (CN->MCI 0.501). "Not collapsed" != "more useful". Do not
  spend more effort fixing z.std. In the paper, reframe small z.std as a
  feature (info compressed into a progression subspace), not a weakness.
- No more loss variants, no more hyperparameter tuning on frozen features,
  no PCA reduction (disease signal lives in low-variance dims).
```
