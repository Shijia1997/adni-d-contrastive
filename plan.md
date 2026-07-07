# ADNI d_contrastive — Research Plan (d-internalization + 3-way conversion study)

Last updated: 2026-07-01

This document is the single source of truth for (a) what was changed in the code
and why, (b) how to run it on the cluster, and (c) the research plan going
forward. Everything here is CPU-runnable (small MLP on **frozen** Swin features);
the local laptop has no GPU/torch, so all runs happen on JHPCE.

> **2026-07-01 update — NEW MAIN: Workstream 2 (Section 8), "internalizing d into
> the image encoder."** The 3-way conversion study (Section 3b) established the
> key empirical fact: a Ridge `d_hat` (frozen-embedding → `d_mod3`) *beats* the
> contrastive and task-trained baselines. So `d` carries strong, transferable
> signal. The new question is whether an image encoder can **internalize** that
> signal — image-only at inference, no `d` recomputed — and improve both
> **diagnosis (CN/MCI/AD)** and **conversion**. Sections 3b/1 remain the frozen
> baselines this workstream must beat.
>
> **2026-06-27 update.** The focus was the **3-way-split conversion study**
> (Section 3b): contrastive / finetune / test splits, contrastive learning as the
> main method, and two Ridge `d_hat` baselines (downstream-only vs all-train).
> **RASPER (Section 4) is shelved** — see the note there for why.

---

## 0. What we have (verified from data + code)

**Imaging features.** Frozen 3D **Swin UNETR** embeddings, **768-d**, in two
versions (`raw`, `combat`). The Swin backbone is frozen — every experiment is a
light head on top of these 768-d vectors. We do *not* touch raw images.

**Granularity = visit-level.**
- Feature matrix: **2408 train visits**, **611 test visits** (`swin_combat_*.npy`).
- Subjects: ~954 train / 239 test, ~3.5 visits per subject.
- Diagnosis / `d_mod3` regression tasks use one row per visit.
- **Conversion** tasks are subject-level labels on the **baseline visit's**
  feature: `build_conversion_labels` (`minimal_v0_contrastive.py:400`) takes each
  subject's first visit and checks whether dx converts within the horizon.

**Conversion cohort sizes (2y, after censoring — the binding constraint).**

| task | train n | train conv | test n | test conv |
|---|---|---|---|---|
| MCI→AD 1y | 324 | 6 | 85 | 4 |
| MCI→AD 2y | 268 | 29 | 65 | 14 |
| MCI→AD 3y | 205 | 58 | 51 | 19 |
| MCI→AD 4y | 159 | 64 | 45 | 22 |
| CN→MCI 2y | 244 | 14 | 59 | 5 |
| CN→MCI 3y | 201 | 25 | 49 | 9 |
| CN→MCI 4y | 169 | 33 | 42 | 9 |

Converters are very few (CN→MCI 2y test has **5**). This is why we (1) borrow
external ranking to stabilize small models, and (2) treat `d_mod3` regression
(n=2408, continuous) as the statistical workhorse.

**Model.** Contrastive learning is a **small MLP** (`ContrastiveMLP_v2`,
`minimal_v0_contrastive.py:705`):
`768 → Linear(256) → ReLU → Dropout(0.2) → Linear(128) → BatchNorm` encoder, plus
a 1-D `progression_head` (`Linear(128→1)`) that produces the rank score `s`.

---

## 1. The rank-loss problem (boss's note: "not pure rank")

### What was there (original)
`soft_kendall_loss` (`minimal_v0_contrastive.py:768`) minimizes

```
L = - mean_{i!=j}  tanh(alpha * (s_i - s_j)) * sign(d_i - d_j),   alpha = 10
```

The most-basic Kendall's τ is `mean sign(s_i - s_j) * sign(d_i - d_j)`.

### What was "added" / why it is not pure rank
1. **`tanh(alpha·Δs)` with a hand-picked `alpha=10`.** A smooth surrogate for the
   order indicator is necessary for differentiability — but with finite `alpha`,
   `tanh` also responds to the **magnitude** of the score gap, not just its order.
   Small `alpha` ⇒ the loss rewards making concordant pairs have *larger* score
   gaps (margin/regression flavor). So `alpha` is a knob that leaks magnitude into
   what should be a purely ordinal loss. Only `alpha → ∞` approaches pure rank.
2. **Reported "rank" numbers often came from `hybrid`** (`= euclidean + λ·rank`),
   which is rank mixed with the Y-Aware Euclidean contrastive loss — also not pure
   rank.

### The fix (writeup A — chosen)
Add the canonical smoothed Kendall from Henderson (2026), eq. (11):

```
maximize  mean_{i!=j}  g_nu(s_i - s_j) * ( I(d_i > d_j) - 1/2 )
          g_nu(x) = 1 / (1 + exp(-x / nu))                 # sigmoid surrogate
          nu = 0.1 * std(score)        # principled default (paper: 0.1 * ||beta||)
```

The key difference from the original is the **principled `nu`**: it is calibrated
so `g_nu` closely approximates the indicator (`g_nu(+0.25 sd) >= 0.99`,
`g_nu(-0.25 sd) <= 0.01`), instead of an arbitrary sharpness that mixes in
magnitude. This is as close to pure rank as a differentiable surrogate allows.

> Sign convention verified (numpy mirror): concordant `s,d` → most-negative loss
> (best), anti-concordant → positive (worst).

---

## 2. Code changes made (all in `d_contrastive/minimal_v0_contrastive.py`)

All changes are additive and backward-compatible; the original `soft_kendall_loss`
is kept as the ablation baseline.

| # | Location | Change | Why |
|---|---|---|---|
| 1 | `soft_kendall_loss` | Added docstring noting it leaks magnitude (kept as baseline) | Make the non-pure-rank behavior explicit |
| 2 | new `kendall_loss_basic(s, d, nu=None)` | Principled smoothed Kendall (sigmoid + nu) | The pure-rank fix |
| 3 | `run_setup3(...)` signature | Added `rank_nu=None` | Plumb the new knob |
| 4 | training loop loss selection | New modes `rank_kendall_basic`, `hybrid_basic` route to `kendall_loss_basic`; existing modes unchanged | Switch surrogates without touching old runs |
| 5 | `loss_history` + both result dicts | Log `rank_nu` | Traceability of every run |
| 6 | argparse | `--loss_mode` gains `rank_kendall_basic`, `hybrid_basic`; new `--rank_nu`; help text on `--rank_alpha` | CLI control |

New `loss_mode` values:
- `euclidean` — Y-Aware Euclidean contrastive (unchanged)
- `rank_kendall` — ORIGINAL tanh+alpha (not pure rank; baseline)
- `rank_kendall_basic` — **NEW** principled pure-rank Kendall
- `hybrid` — euclidean + λ·(original rank)
- `hybrid_basic` — euclidean + λ·(basic rank)

> Note: `kendall_loss_basic` is on a different numeric scale than the tanh version
> (`±0.5` label term vs `±1` sign). For `hybrid_basic`, `--lambda_rank` may need
> re-tuning relative to `hybrid`. For pure-rank modes the scale is irrelevant.

---

## 3. Workstream 0 — fix + freeze the pure-rank baseline (run first, ~1 wk)

**Goal:** report what *pure* rank alone achieves, separated from euclidean/hybrid,
and quantify how much the original `alpha` distorted things.

**Run (cluster):**
```bash
cd /dcs07/zwang/data/adni_d/d_contrastive
sbatch run_w0_rank_ablation.sbatch
```
This does two things (see the script):
- **Part 1 — loss geometry:** runs all 5 modes (`euclidean`, `rank_kendall`,
  `rank_kendall_basic`, `hybrid`, `hybrid_basic`) on `raw`+`combat` into
  `results_w0/mode_<MODE>/`.
- **Part 2 — sharpness/smoothing sweep:** `rank_kendall` over
  `alpha ∈ {1,2,5,10,20,50}` and `rank_kendall_basic` over
  `nu ∈ {auto,0.05,0.1,0.25,0.5,1.0}` into `results_w0/sweep_*`.

**What to read:** the **setup3** rows of each `results_w0/*/results_<version>.json`
(and `setup3_*` CSVs). Setup1/2 are recomputed identically across modes — ignore
their duplication, only setup3 depends on the loss.

**Deliverables:**
- One "loss-geometry ablation" table: pure-rank vs +magnitude vs +euclidean, on
  CN/AD AUC, conversion AUC, and `d_mod3` Spearman/R².
- One alpha/nu curve showing how approaching pure order changes conversion —
  the quantitative answer to the boss's concern.

**Consolidation:** `consolidate_rank_results.py` / `make_new_results_summary.py`
already aggregate setup3 outputs; point them at `results_w0/` (adjust the input
glob if needed).

---

## 3b. Workstream 0 (3-way split) — conversion study (CURRENT MAIN)

**Why a 3-way split.** The old 2-way (train/test) mixes representation learning
and downstream task fitting on the same `train`. To cleanly separate them — and to
give the conversion probe its own held-out fit set — the data is re-partitioned
into **three RID-disjoint splits** (no patient/image leakage; verified at load):

| split | RIDs | unique images | role |
|---|---|---|---|
| `contrastive` | 477 | 1206 | pretrain the contrastive encoder on `d_mod3` (main method) |
| `finetune` | 298 | 754 | train the downstream conversion head |
| `test` | 418 | 1059 | **evaluation only** |

Splits live on the cluster at `../data/splits_3way_20260627_v2/` as
`{contrastive,finetune,test}_image_ids.npy` (+ matched CSVs). Conversion follow-up
works within a split because all of a subject's visits stay together.

**HARD CONSTRAINT — test never uses `d_mod3`.** The test split contributes only
(a) imaging features and (b) conversion outcome labels (for AUC). Its `d_mod3` is
never an input to any fit, scaler, or hyper-parameter selection. Enforced 3 ways:
(1) by construction (`mt["d_mod3"]` is never read; `StandardScaler` is fit on
contrastive+finetune features only); (2) a runtime audit that asserts no method's
d-source is `test`; (3) the `.sh` self-check (RID-disjointness + cohort counts
before any training).

**Two Ridge `d_hat` baselines (the boss's request):** Ridge predicting `d_mod3`
from imaging, then `d_hat` used directly as the conversion score —
- `ridge_dhat_finetune` — fit on the **finetune split only** (downstream-only),
- `ridge_dhat_all` — fit on **contrastive+finetune** (all-train); the Δ baseline.

**Methods compared** (per task × horizon × version), AUC + bootstrap CI +
paired Δ vs `ridge_dhat_all`:

| method | what it is |
|---|---|
| `ridge_dhat_finetune` / `ridge_dhat_all` | borrow score, two training pools |
| `direct_logistic` | internal-only logistic on raw imaging (finetune labels) |
| `contrastive_<mode>_probe` | frozen-encoder linear probe on `z` (finetune labels) |
| `contrastive_<mode>_finetune` | end-to-end encoder fine-tune on conversion labels |
| `contrastive_<mode>_s` | the 1-D progression score `s`, no downstream training |

`<mode> ∈ {euclidean, rank_kendall_basic, hybrid_basic}`. The frozen-probe vs
fine-tune pair answers "freeze or fine-tune the encoder for conversion?".

**Code (all additive in `minimal_v0_contrastive.py`, no existing fn modified):**
- `train_contrastive_encoder(...)` — Stage-1 contrastive loop factored out verbatim
  from `run_setup3` (single source of truth for the loss math).
- `encode_features(model, X)` — frozen `(z, s)`.
- `finetune_encoder_classifier(...)` — deep-copies the pretrained encoder + BCE head,
  trains end-to-end; returns test P(convert).
- `load_features_3way(features_dir, split_dir, d_csv, version)` — loads the 3 splits
  by image-id list, builds longitudinal meta, **raises on any cross-split RID overlap**.
- `run_w0_conversion_3way.py` — the driver; `run_w0_conversion_3way.sbatch` — one-click.

**Run (cluster):**
```bash
cd /dcs07/zwang/data/adni_d/d_contrastive
sbatch run_w0_conversion_3way.sbatch       # self-check -> raw+combat full run
# or directly:
python run_w0_conversion_3way.py \
  --features_dir ../data/embeddings_128_05152016 \
  --split_dir   ../data/splits_3way_20260627_v2 \
  --d_csv       ../data/master_smri_05152016/D_with_image_paths_full.csv \
  --versions raw combat --horizons 2 3 4 \
  --loss_modes euclidean rank_kendall_basic hybrid_basic \
  --epochs 150 --device cpu --out_dir results_w0_conv_3way
```
Outputs: `results_w0_conv_3way/w0_conversion_3way.{csv,md}`. Use `--no_finetune`
to skip the fine-tune variant (frozen probe only).

---

## 4. Workstream 1 — RASPER for conversion (SHELVED 2026-06-27)

> **Shelved.** Quick runs showed RASPER could not beat `ridge_d_hat` under the
> option-A mapping: the internal covariates (PCA-16 of Swin) are a *degraded
> subset* of the same Swin-768 information the external Ridge `d_hat` ranker
> already uses, so borrowing that ranking adds nothing the score didn't carry — a
> structural, not a tuning, problem. A fair test needs option B (an *independent*
> external ranker, e.g. clinical), which we don't have a usable model for yet. The
> code below stays in the repo and passes its self-check; revisit only if an
> independent external ranker becomes available. Current focus = Section 3b.

### (archived) Original W1 plan

RASPER (Henderson 2026) = penalized regression that borrows an **external risk
*ranking*** instead of external *scores/coefficients*. This is the principled
version of our existing finding that `ridge_d_hat` (an external progression score)
beats `direct_logistic` on conversion. RASPER is **complementary** to the
contrastive method (W0), not a replacement: W0 is the main representation learner;
RASPER consumes a representation + an external ranking to build the small model.

**Files (all CPU-only, pure numpy/scipy/sklearn — no torch):**
- `rasper.py` — the estimator. `RASPER` class (objective eq. 14, analytic
  gradient, L-BFGS-B), `kendall_concordance` / `spearman_concordance` (eqs. 11/9,
  the same smooth Kendall as W0), `select_lambda_alpha_cv` (LOO or stratified
  k-fold, paper §3.3). `nu` default = `0.1 * ||beta_MLE||`.
- `run_w1_rasper_conversion.py` — the experiment driver.
- `test_rasper_sim.py` — self-check (gradient check + reproduces the paper's
  "high rank corr, large score gap" win). **Verified locally: passes, RASPER
  wins 20/20.**
- `run_w1_rasper.sbatch` — cluster job (runs the self-check then the experiment).

**Role mapping (decided: external ranker = option A, the d_mod3 model):**

| RASPER concept | Our object | In code |
|---|---|---|
| Internal study (small) | conversion task (MCI→AD / CN→MCI), `Y` = conversion label | censored cohort via `build_censored_conversion_cohort` |
| External risk model `f_E` | **Ridge `d_hat` predicting `d_mod3` on the large train split** | `external_dhat()` |
| External ranking `r^E` | rank of `d_hat` on the internal cohort | `ranks(dhat_tr[tri])` |
| Internal covariates `x` | low-dim imaging block (**default: PCA-16 of frozen Swin 768**) | `build_internal_features()` |
| Penalty `λ` | strength of forcing internal risk order to match `r^E` | `RASPER(lam=...)` |

`d_mod3` is a related/surrogate outcome to conversion, so its **ranking** is the
transportable signal — exactly RASPER's assumption.

**Upgrade path:** `build_internal_features()` currently uses PCA-16 of the Swin
features so W1 runs standalone *before* W0 finishes. Once W0 is done, swap in the
W0 contrastive latent `z` as the internal covariate block (the "novel imaging
biomarker") — this is the W0→W1 hand-off.

**How to run (cluster):**
```bash
cd /dcs07/zwang/data/adni_d/d_contrastive
sbatch run_w1_rasper.sbatch          # self-check + full run
# or directly:
python run_w1_rasper_conversion.py --versions raw combat --horizons 2 3 4 \
  --n_pca 16 --penalties kendall --cv_splits 5 --out_dir results_w1_rasper
```
Outputs: `results_w1_rasper/w1_rasper_conversion.{csv,md}` — one row per
method × task × horizon × version, with **bootstrap AUC CIs** and **paired Δ +
p-value vs `ridge_d_hat`**.

**Comparator matrix (per task):** `direct_logistic` (internal only),
`ridge_d_hat` (borrow score), **`rasper_kendall` (borrow ranking)**,
`oracle_true_d_mod3` (upper bound). Headline contrast: **`rasper_*` vs
`ridge_d_hat`** — does borrowing the *ranking* beat borrowing the *score*?

**Scope note — conversion now, d_mod3 regression next:** the driver covers the
small conversion cohorts (RASPER's natural small-n setting; exact O(n²) pairwise
is fine here). The continuous `d_mod3` regression (n=2408) is too large for exact
pairwise — validate that arena with the paper-style simulation in
`test_rasper_sim.py` (and add pair-subsampling to the estimator later if a full
d_mod3 RASPER fit is wanted).

**(Optional, later) Workstream 2 — option B external ranker:** a published AD/MCI
risk score (age/APOE/MMSE/hippocampal volume) as a black-box ranker — the truest
RASPER setting and most publishable, pending a usable external model. Just replace
`external_dhat()` with that model's score.

---

## 5. Evaluation & statistical rigor (applies throughout)

- Tiny converter counts → every conversion AUC needs **bootstrap CIs** and
  **paired Δ + p-value** (helpers already in `experiment_utils.py`).
- Always report `n` and `n_converters`; never claim from a single point AUC.
- Add a **permutation / random-external-ranking control** to confirm gains come
  from real ranking information, not extra degrees of freedom.

---

## 6. Open decisions & risks

- **Risk:** conversion `n` too small → RASPER gains may sit inside CI noise.
  *Mitigation:* aggregate across horizons/tasks; lead with `d_mod3` regression;
  conversion as downstream confirmation.
- **Decided:** external ranker = **A** (self-contained `d_mod3` model) first.
- **Pending:** whether a suitable published external AD model exists for the
  option-B truest-RASPER result.

---

## 7. How to reproduce / smoke test

```bash
source /users/szhang1/fsl/bin/activate optuna_env
cd /dcs07/zwang/data/adni_d/d_contrastive

# fast wiring check of the new mode (2 epochs)
python minimal_v0_contrastive.py \
  --features_dir ../data/embeddings_128_05152016 \
  --d_csv ../data/master_smri_05152016/D_with_image_paths_full.csv \
  --versions raw --loss_mode rank_kendall_basic --smoke \
  --output_dir results_smoke_basic

# full Workstream 0
sbatch run_w0_rank_ablation.sbatch
```

---

## 8. Workstream 2 — internalizing `d` into the image encoder (NEW MAIN, 2026-07-01)

### 8.0 Thesis / central question

`d_mod3` (Wang's continuous AD-progression score) is a strong **dense** supervisor:
the 3-way study (Section 3b) showed a Ridge `d_hat` (frozen 768-d → `d`) *beats*
contrastive-geometry and task-trained baselines on conversion. That means `d`
holds transferable disease signal that the current pipeline only reaches by
explicitly regressing `d` at test time.

> **Central question.** Can an image encoder *internalize* `d` — so that at
> inference we use **image only, without recomputing `d`** — and still get the
> benefit on **diagnosis (CN/MCI/AD)** and **conversion (MCI→AD, CN→MCI)**?
> And, as a ceiling: **how much of `d`'s value survives image-only vs. having
> `d` explicitly available?**

### 8.1 Two inference regimes (kept explicit — this is the crux)

- **R1 — image-only (primary goal).** `d` is used *only during training* to shape
  the encoder. Test path: `image → encoder → task head`. No `d` computed at test.
  This is the deployment-realistic setting ("不需要重新算 d").
- **R2 — `d`-as-support (ceiling / additional analysis).** `d` (from the test
  image via Wang's model) is concatenated with the imaging representation at test.
  Answers "if you *do* have `d`, how much does it help." Mainly a support/upper-bound.

**Leakage rule (unchanged, enforced 3 ways as in 3b).** The **test** split's
`d_mod3` never enters any fit, scaler, or hyper-parameter selection. In **R2**,
`d` is a *test-image-derived input feature* fed to a model whose parameters were
fit on train only — this is allowed, but it **changes the deployment assumption**
(you must run Wang's model at test), so R2 rows are always labeled as such and
never mixed into the R1 headline.

### 8.1b Split mode — 2-way (PRIMARY for W2) vs 3-way  *(decided 2026-07-03)*

W2 runs on the **2-way** `train`/`test` partition (the previous split, from
`matched_TRAIN.csv` / `matched_TEST.csv`): the encoder pretrains on `train`'s `d`
**and** the downstream head trains on `train`'s task labels; `test` is held out.
Rationale: the strict 3-way split starved the downstream (finetune ≈ 298 subjects,
few converters → high variance), which was a main reason contrastive looked weak.
2-way gives the methods the full train pool with **no test leakage** — it only
drops the clean representation-vs-head data separation, which is a fairness nicety,
not a leakage requirement.

Every W2 driver takes `--mode {2way,3way}` (default `3way` for back-compat; the W2
sbatches pass `--mode 2way`). Loader `load_features_2way` returns `train`/`test`
and aliases `contrastive`/`finetune`→`train` with `_mode='2way'`, so the drivers
run unchanged; in 2-way the two ridge `d_hat` configs collapse to one fit on
`train` (`ridge_dhat_finetune` == `ridge_dhat_all`). `test` `d` is still never in
any fit.

### 8.2 Design axes

| Axis | Levels |
|---|---|
| **A. How `d` supervises the encoder** | (a) none / task-only baseline · (b) contrastive geometry on `d` (euclidean / rank — current W0) · (c) **direct `d`-regression** (encoder trained to predict `d`, MSE) — NEW |
| **B. How much the encoder adapts** | (i) frozen Swin + light head (current) · (ii) **LoRA** on `swinViT` · (iii) full fine-tune |
| **C. Downstream task** | diagnosis 3-class CN/MCI/AD (+ CN/AD, CN/MCI binaries) · conversion MCI→AD, CN→MCI (2/3/4 y) |
| **D. Scheme** | two-stage (pretrain-`d` → freeze rep → train task head) vs. joint / auxiliary (`task loss + λ·d loss`) |
| **E. Covariates** | +age, +APOE — *additional analysis, added last* |

The scientific spine is **A × B**: does *direct `d`-regression* (c) beat
*contrastive geometry* (b), and does *adapting the backbone* (ii/iii) beat
*frozen* (i)? Everything else is held fixed for fairness (§8.7).

### 8.3 Phase 0 — characterize `d` ↔ diagnosis ↔ conversion (analysis, CPU, cheap)

Pure analysis, no training. Establishes the **ceiling every later method chases**:
- correlation / ordinal-AUC of true `d_mod3` vs diagnosis, and vs conversion;
- AUC of **true `d`** and of **frozen `d_hat`** for each task (the borrow-score ceiling);
- how much diagnosis/conversion signal is `d`-explained vs. imaging residual.

Deliverable: a "`d`-value" table — the number each internalization method is trying
to recover from image alone.

### 8.4 Phase 1 — frozen encoder, image-only (reuse existing infra, CPU)

On the frozen 768-d embeddings (`data/embeddings_128_05152016`, raw + combat),
compare **supervision × scheme** for both tasks:
- (a) no-`d` baseline: task head directly on 768-d (logistic/BCE; reuse `binary_dx_auc`, `logistic_score`).
- (b) contrastive-geometry pretrain (euclidean / rank) → probe / fine-tune head (current W0 code).
- (c) **NEW direct-`d`-regression pretrain**: encoder + `progression_head` trained with `MSE(head(z), d)`, then downstream task on the learned rep. Implement as a `loss_mode="regress_d"` in `train_contrastive_encoder`, or an added `--aux_mse_lambda` for the joint scheme — additive, does not touch existing modes.
- schemes: two-stage vs. joint/auxiliary.

This is the cheap core and directly tests the earlier hypothesis: *a direct-`d`
regression objective should be a cleaner internalization of `d` than the
relative-only contrastive geometry.* Baselines `ridge_dhat_finetune` /
`ridge_dhat_all` / `direct_logistic` ride along in every table.

> **Implemented (2026-07-03).** `regress_d` mode added to
> `train_contrastive_encoder` (per-sample `MSE(s, d)`, additive — existing modes
> untouched). Drivers: `run_w0_phase0_dvalue.py` (Phase 0 ceiling),
> `run_w0_phase1_diagnosis.py` (diagnosis 3-class + binaries, arms a/b/c),
> conversion via `run_w0_conversion_3way.py` with `regress_d` added to
> `--loss_modes`. **Scheme = two-stage** (pretrain `d` → freeze → task head), which
> is exactly the "先学 d 再学 task" ask; the joint/auxiliary scheme is a later
> extension. One-click: `run_w2_phase01.sbatch` (CPU/`shared`).

### 8.4b Phase 1 — `d`-as-support (regime R2 ceiling)  *(implemented 2026-07-03)*

Directly answers "把 `d` 加进去和 image 一起,会更好". `run_w2_phase1_dsupport.py`
compares, per diagnosis/conversion task: `image_only` vs `image+dhat` (R1,
deployable, `d_hat` = frozen Ridge) vs `image+dtrue` (R2, feeds the test image's
true `d` as an input feature at test time — params still fit on train only; the
`d` column is z-scored by finetune stats so no test stats leak into the transform).
R2 rows are labelled non-deployable and never enter an R1 headline. Frozen, CPU;
runs from the same `run_w2_phase01.sbatch` bucket if desired.

### 8.5 Phase 2 — unfreeze the Swin: LoRA + full fine-tune  **[code written 2026-07-03; run gated on infra]**

**Gate (confirm on cluster on first run — the sbatch step 0 does this):** raw NIfTI
volumes reachable via `sMRI_path`, the BTCV `SwinUnetrModelForInference` package
(`/dcs07/zwang/data/pmrc/SwinUNETR/BTCV`) importable, HF weights
`darragh/swinunetr-btcv-base` cached, GPU node reachable. **This laptop has no GPU
— every Swin/LoRA job runs on the cluster via `sbatch` only.**

- (ii) **LoRA** (self-contained `LoRALinear`, no `peft` dep) injected into `swinViT`
  Linear modules matching `--lora_targets` (default `(qkv|proj|fc1|fc2)$`); backbone
  frozen, adapters + head trained. (iii) **full fine-tune** of `swinViT` + head.
- **Design decision (reuse):** the trainer (`train_w2_phase2_swin.py`) trains
  `Swin(adapt) → ContrastiveMLP_v2 head` on contrastive `d`, then **exports the
  adapted 768-d pooled backbone features** in the frozen on-disk layout
  (`swin_latent.npy` + `image_id_order.npy`). Downstream then reuses the Phase-0/1
  CPU drivers **byte-for-byte** on that export — so any delta is attributable to the
  backbone adapting: a clean **adapted-768 vs original-768** comparison, holding the
  downstream identical.
- Compute (§8.8): physical batch 8 + grad-accum 16 (eff 128) + gradient
  checkpointing; GPU node `compute-126`. Orchestrated by `run_w2_phase2_swin.sbatch`
  (step 0 = module-tree dump + LoRA forward/backward+export smoke; step 1 = adapt ×
  loss-geometry, then downstream on each adapted export).
- **Cannot be validated off-cluster** (needs GPU + volumes + BTCV pkg). Locally only
  AST/`bash -n` checked; the sbatch step 0 is the real first-run verification.

### 8.6 Phase 3 — covariates + R2 ceiling  **[additional analysis, last]**

- **age / APOE — DEFERRED (2026-07-03): data not available yet.** Revisit once the
  covariate data lands. Columns `age`/`ageori`/`apoe` already exist in the master
  CSVs but the values/coverage we need are pending. Question when done: *given `d`
  is already modelled, does age/APOE add anything?* (fine if not — an ablation).
- **R2 `d`-as-support** ceiling: **done** in §8.4b (`run_w2_phase1_dsupport.py`).

### 8.7 Fairness protocol (applies to every arm)

1. **Same 3-way RID-disjoint split** (contrastive / finetune / test) as 3b; same
   fixed seeds; test `d` never in any fit.
2. **Same metrics + CIs**: diagnosis = 3-class macro-AUC + CN/AD + CN/MCI AUC;
   conversion = per-task/horizon AUC with **bootstrap CIs** and **paired Δ vs
   `ridge_dhat_all`** (helpers in `experiment_utils.py`).
3. **Same downstream head capacity + epochs** when comparing supervision arms, so a
   difference reflects the *representation*, not head budget.
4. **Matched optimizer cadence across B-levels**: frozen uses batch 128; LoRA /
   full-FT use physical batch 8 + gradient accumulation to **effective 128** (§8.8).
5. **Contrastive pairwise-batch caveat (important).** Gradient accumulation does
   **not** enlarge the in-batch pair set: a pairwise contrastive loss over 16
   micro-batches of 8 sees 16 independent 8-sample pair problems, *not* one
   128-sample pair problem. For per-sample losses (regression MSE, BCE) accumulation
   is exact; for the pairwise geometry it is **not**. To keep b (contrastive) fair:
   - enable **gradient checkpointing** on `swinViT` to push the *physical* batch as
     high as memory allows (LoRA has few trainable params → activations dominate →
     checkpointing buys the most headroom);
   - optionally add a **feature memory-bank / queue** (MoCo-style) so the pairwise
     loss sees a large effective pool at small physical batch;
   - always report a **matched frozen-batch-8 contrastive control** so frozen vs
     LoRA are compared at *equal pairwise batch*, not 128-vs-8.
   The direct-`d`-regression arm (c) is immune to this — another reason it is the
   cleaner primary internalization objective.
6. **Baselines in every table**: `ridge_dhat_finetune`, `ridge_dhat_all`,
   `direct_logistic` (task-only). A method "wins" only if it beats these image-only
   / borrow-score references.

### 8.8 Compute / infra standard (all Swin-based jobs)

- **Cluster-only, GPU.** No local GPU; author/py_compile locally, run via `sbatch`.
- **GPU node:** `compute-126` for **all** Swin / Swin-LoRA / full-FT experiments
  (`#SBATCH --nodelist=compute-126`, `--gres=gpu:1`; confirm the partition that
  owns `compute-126` on first submit).
- **Batching:** Swin & Swin-LoRA contrastive → **physical batch 8** + gradient
  accumulation to **effective 128** (accum steps = 16) + **gradient checkpointing
  on**. Effective 128 is chosen to match the current frozen contrastive batch.
- **Frozen-embedding jobs stay CPU** (`shared` partition, batch 128) — Phase 0/1.
- **Current contrastive batch size = 128** (`train_contrastive_encoder` default;
  `run_w0_conversion_3way.py --batch_size 128`) — this is the "effective 128" target.

### 8.9 Execution order & deliverables

`P0 (ceiling) → P1 (frozen, direct-d vs contrastive, both tasks) → P2 (LoRA/full-FT,
gated) → P3 (age/APOE + R2)`. P0/P1 start now (no new infra); P2 starts once §8.5
gate is confirmed.

Deliverables:
- **Ceiling table** (P0): true-`d` and frozen-`d_hat` AUC per task.
- **Master table**: rows = supervision × adaptation × scheme, cols = diagnosis +
  conversion metrics, `ridge_dhat_*` baseline rows always present, R1 vs R2 labeled.
- **Gain-from-adaptation table**: frozen vs LoRA vs full-FT, per supervision arm.

### 8.10 Open items to confirm

- Exact partition owning `compute-126` (and whether it needs `--account` / QOS).
- LoRA target modules inside `swinViT` (attention `qkv`/`proj`, MLP `fc1`/`fc2`).
- Whether raw volumes are already resampled/cached or must be re-read per epoch
  (I/O budget for full-FT vs LoRA).
