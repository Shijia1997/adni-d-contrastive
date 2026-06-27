# ADNI d_contrastive — Research Plan (pure-rank fix + 3-way conversion study)

Last updated: 2026-06-27

This document is the single source of truth for (a) what was changed in the code
and why, (b) how to run it on the cluster, and (c) the research plan going
forward. Everything here is CPU-runnable (small MLP on **frozen** Swin features);
the local laptop has no GPU/torch, so all runs happen on JHPCE.

> **2026-06-27 update.** The focus is now the **3-way-split conversion study**
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
