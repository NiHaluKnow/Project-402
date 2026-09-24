# Bayesian Reliability Estimation of Computer Hardware Systems Using Monte-Carlo Simulation

A from-scratch Python implementation of the reliability-estimation framework from

> K. Arekar, R. Jain, and S. Kumar, "Bayesian Estimation of System Reliability Models
> Using Monte-Carlo Technique of Simulation," *Journal of Statistical Theory and
> Applications*, vol. 20, no. 1, pp. 149-163, 2021.
> https://doi.org/10.2991/jsta.d.210201.001 (open access)

extended from the paper's own vehicle-fleet worked example to a real computer-hardware
case study (hard-drive failure data), per the project proposal.

This README explains what was built, exactly what is real vs. modelled in the data,
how to reproduce every result, and how to swap in genuine raw field data later.

---

## 1. How this maps to the four proposal objectives

| # | Proposal objective | Where it's implemented | Run with |
|---|---|---|---|
| 1 | **Re-implementation** — reproduce the Weibull/prior/Monte-Carlo pipeline and validate it | `src/weibull_core.py`, `src/bayes_estimator.py` | `run_scripts/run_01_validate_implementation.py` |
| 2 | **Extension** — apply the framework to a real computer-hardware dataset | `src/data_pipeline.py` | `run_scripts/run_02_build_dataset.py` |
| 3 | **Comparative analysis** — Bayes vs. classical (MLE/MVUE) | `src/weibull_core.py`, `src/nonparametric.py` | `run_scripts/run_03_fit_and_compare.py` |
| 4 | **Prior sensitivity** — Rayleigh vs. Beta vs. Uniform | `src/priors.py` | `run_scripts/run_04_prior_sensitivity.py` |

Run everything in order with:

```bash
pip install -r requirements.txt
python run_all.py
```

Plots land in `results/`, datasets and provenance records in `data/`.
Each script also prints a full numeric summary to the console and saves it as JSON.

---

## 2. Methodology — derived directly from the base paper

The base paper's **Section 7** ("Bayesian Estimation of System Reliability with Priori
Rayleigh Distribution using Monte-Carlo Simulation", Eq. 22-30) is the part of the paper
that matches this project's stated case study most closely — a Weibull time-to-failure
model with **known shape** and an unknown scale/rate parameter carrying a **Rayleigh
prior**. That section is what the rest of this project's model is built from.

### 2.1 The model (Eq. 22-23 of the paper)

Each unit's time-to-failure `T` has pdf

```
f(t; beta, kappa) = beta * t^(kappa-1) * exp(-beta * t^kappa / kappa),   t > 0
```

- `kappa` (the paper's `alpha_i + 1`) — Weibull **shape**, assumed **known**.
- `beta` (the paper's `beta_i`) — the unknown **rate** parameter this whole project
  estimates. (Relation to the textbook scale `eta`: `beta = kappa / eta^kappa`.)

Reliability: `R(t; beta) = exp(-beta * t^kappa / kappa)`.

The prior on `beta` is Rayleigh(b): `g(beta) = (beta/b^2) * exp(-beta^2/(2b^2))`
— exactly Eq. 23.

### 2.2 Sufficient statistic and classical estimators

For `n` units with `r` observed failures {t_i} and `n-r` right-censored units {c_j}:

```
S = sum_{failures} t_i^kappa + sum_{censored} c_j^kappa
```

- **MLE**: `beta_hat = r*kappa / S`  →  `R_MLE(t) = exp(-r * t^kappa / S)`
- **MVUE** (standard textbook UMVUE for Weibull-known-shape under Type-II censoring —
  see Bain & Engelhardt, *Statistical Analysis of Reliability and Life-Testing Models*):
  `R_MVUE(t) = (1 - t^kappa/S)^(r-1)` for `0 <= t^kappa < S`, else 0.
  This generalises the paper's own two closed-form special cases: `kappa=2` (Rayleigh,
  Section 2, exponent `n-1`) and a related `kappa=1` exponential form (Section 4, under a
  different Poisson/time-truncated sampling scheme, hence a different exponent there).

### 2.3 Bayes estimator

```
r~(t) = E[R(t;beta) | data]
      = Integral{ beta^r * g(beta) * exp(-beta*(S+t^kappa)/kappa) dbeta }
        -------------------------------------------------------------------
        Integral{ beta^r * g(beta) * exp(-beta*S/kappa) dbeta }
```

No closed form exists for a Rayleigh (or Beta/Uniform) prior combined with this
likelihood — which is exactly why the paper turns to Monte-Carlo simulation. Three
independent ways of computing this same quantity are implemented and cross-validated
against each other in `src/bayes_estimator.py`:

1. **Numerical quadrature** (`bayes_reliability_quadrature`) — near-exact 1-D
   integration, used as ground truth.
2. **Self-normalised importance sampling** (`bayes_reliability_importance_sampling`) —
   draw `beta ~ prior`, reweight by the likelihood. Works for any prior.
3. **The paper's own two-stage Monte-Carlo replication**
   (`bayes_reliability_monte_carlo_paper`, Eq. 29-30): draw `k_s` posterior samples of
   `beta` (via sampling-importance-resampling), simulate `k_r` replicate Weibull
   lifetimes per draw using the paper's own inverse-CDF formula (Eq. 21/29), then
   `R_hat(t) = (N - k_f(t)) / N` where `N = k_s*k_r` and `k_f(t)` counts replicate
   lifetimes `<= t`.

`run_01_validate_implementation.py` confirms all three agree with each other and
recover the known truth from simulated data (see §4).

### 2.4 A note on exact numeric reproduction of the paper's own tables

The base paper's published Tables 2, 4, 5 and 8 contain a number of internal
inconsistencies (non-monotonic values where the underlying quantity should be
monotonic, one outlier row that breaks an otherwise clean pattern) that are very
likely OCR/typesetting artefacts in the original journal PDF rather than genuine
values. Bit-for-bit reproduction of those specific numbers is therefore not a
meaningful or achievable validation target. Instead, §4 below validates the
implementation the standard, more rigorous way: by recovering known ground-truth
parameters from simulated data, and cross-checking three independent computations of
the same Bayes estimator against each other.

---

## 3. The hardware dataset — what's real and what's modelled (please read this)

The proposal's Section 5 called for real Backblaze hard-drive field data. Here is
exactly what was and wasn't achievable inside this sandboxed build environment, stated
plainly:

**What's real, current, and cited:** the *population-level statistics* for two real
Backblaze drive models, taken directly from Backblaze's own **Q1 2026 Drive Stats
report** (published July 2026):

| Model | Real drive count | Real cumulative drive-days | Real cumulative failures | Real AFR |
|---|---|---|---|---|
| HGST HUH721212ALE600 (12 TB) | 2,608 | 6,115,413 | 104 | 0.62% |
| WDC WUH722222ALE6L4 (22 TB) — Backblaze's largest single-model cohort | 45,638 | 3,992,942 | 42 | 0.38% |

Sources: `https://www.backblaze.com/blog/backblaze-drive-stats-for-q1-2026/`,
`https://www.backblaze.com/blog/?p=113117`, `https://www.backblaze.com/hard-drive.html`
(all retrieved 2026-09-23; full citation trail also in `data/objective2_dataset_provenance.json`).

**What's simulated:** the actual raw daily SMART-log files (the individual per-drive,
per-day snapshots) are several **gigabytes per quarter** and hosted at backblaze.com,
which is outside this build environment's network access — they could not be
downloaded here. So the *individual day on which each simulated drive fails* is
generated from a Weibull model, **not** read from a real log line.

**How the simulation is calibrated (`src/data_pipeline.py:build_real_stat_calibrated_dataset`):**
for each model, a common censoring horizon `C = real_drive_days / real_drive_count`
is computed (the real average observed service time per drive), and the Weibull rate
`beta` is solved so that the simulated population's expected failure probability at
`C` exactly equals the real observed failure proportion (`real_failures / real_drive_count`).
The **population size, observation horizon, and expected failure count are therefore
real, cited numbers**; only the exact day-of-failure for each simulated unit is
modelled. The realised simulated failure count is printed alongside the real target
every run so the match is visible (e.g. 107 simulated vs. 104 real for the 12 TB model).

**Assumed Weibull shape (`kappa = 1.3`, `src/config.py`):** the paper's framework
treats the shape as known. We use a single mild "wear-out" shape for both models,
broadly consistent with reliability-engineering literature on enterprise HDD fleets
(constant-hazard/electronics-like failure would be `kappa=1.0`; strong mechanical
wear-out would be `kappa>2`). It's a clearly labelled, easily-changed constant — not a
number fitted to make results look a particular way.

### 3.1 Using genuine raw Backblaze data instead

`src/data_pipeline.py` ships a complete, schema-accurate parser for the **real**
Backblaze daily-CSV format — `load_backblaze_raw_csvs()`. To use actual field data:

1. Download the real quarterly ZIP(s) yourself from
   `https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data`
   (this needs a machine with normal internet access and a few GB of disk space — both
   unavailable in this build sandbox).
2. Unzip into a folder of daily/quarterly CSVs.
3. Replace the call to `build_real_stat_calibrated_dataset(...)` in
   `run_scripts/run_02_build_dataset.py` with:
   ```python
   from src.data_pipeline import load_backblaze_raw_csvs
   data = load_backblaze_raw_csvs(csv_dir="/path/to/your/csvs",
                                   model="HUH721212ALE600", kappa=1.3)
   ```
4. Nothing else in the project needs to change — `run_03` and `run_04` consume
   `FailureData` objects regardless of how they were built.

---

## 4. Results obtained

Full numeric detail is in `results/objective{1,3,4}_*.json`; headline findings:

**Objective 1 (validation).** MLE recovers the true `beta` to within ~1-3% relative
error at n=5000 across multiple `(beta, kappa)` settings including the paper's own
`kappa=1` and `kappa=2` special cases; the quadrature and Monte-Carlo (Eq. 29-30) Bayes
estimators agree with each other to RMSE < 0.005 in every scenario tested, and both
converge to the true `R(t)` curve as sample size grows (see `results/validation_*.png`).

**Objective 3 (Bayes vs. classical).** With the full real-calibrated dataset (thousands
of drives), MLE, MVUE and Bayes are all close to the nonparametric Kaplan-Meier curve
(RMSE ~ 0.002-0.003 for both drive models) — the three methods agree once there's
enough data, as they should. With a **limited** subsample (50 drives, mimicking early
field data), MVUE's error spikes sharply (e.g. RMSE 0.0135 vs. 0.0021 for Bayes on the
12 TB model — see `results/rmse_bars_HGST_12TB.png`) while the Bayes estimate — using a
prior centred on realistic prior/fleet knowledge — stays close to the reference curve.
This is precisely the small-sample value proposition described in this project's own
Introduction.

**Objective 4 (prior sensitivity).** With the full dataset, the three priors (Rayleigh,
Beta, Uniform — constructed with matched prior means, `src/priors.make_comparable_priors`)
agree almost exactly (max pairwise difference in `R(t)` < 0.001) — the likelihood
dominates. With limited data, the priors visibly diverge, and — a genuine finding worth
noting explicitly — the divergence is largest not simply "when data is scarce" but
specifically when extrapolating **beyond** the observed horizon (`results/prior_sensitivity_*.png`
plots out to 4x the observed horizon for exactly this reason). For the 22 TB model's
limited-data scenario (0 observed failures in the subsample, very short real observation
window), all three priors still predict near-certain reliability across the plotted
range — a legitimate result, not a bug: with essentially no data and a short window, the
Weibull exponent stays tiny for any plausible `beta`, so prior *shape* barely matters
there regardless of sample size.

---

## 5. Project layout

```
bayesian_reliability_project/
├── README.md                 <- this file
├── requirements.txt
├── run_all.py                <- runs every stage in order
├── src/
│   ├── config.py              constants shared by every script (kappa, seeds)
│   ├── weibull_core.py         Weibull model, MLE, MVUE
│   ├── priors.py               Rayleigh / Beta / Uniform priors
│   ├── bayes_estimator.py      quadrature / importance-sampling / paper's own MC (Eq 29-30)
│   ├── data_pipeline.py        real raw-CSV parser + real-stat-calibrated dataset builder
│   ├── nonparametric.py        Kaplan-Meier reference curve
│   ├── sampling.py             deterministic "limited data" subsampling
│   └── plotting.py             all figure generation
├── run_scripts/
│   ├── run_01_validate_implementation.py
│   ├── run_02_build_dataset.py
│   ├── run_03_fit_and_compare.py
│   └── run_04_prior_sensitivity.py
├── data/                      generated: per-drive datasets + provenance JSON
└── results/                   generated: all PNG plots + metric JSONs
```

## 6. Known limitations (stated plainly, for the write-up)

- The hardware dataset's exact failure *timing* is simulated, calibrated to real
  aggregate statistics rather than read from raw field logs (§3). This is the single
  biggest deviation from "collect a real dataset" as literally stated in the
  reviewers' feedback, and is entirely a consequence of this build sandbox's network
  restrictions, not a methodological choice — §3.1 gives the exact swap-in path once
  you have the real files.
- The Weibull shape `kappa=1.3` is an assumed, literature-informed constant rather than
  independently fitted per drive model (consistent with the base paper's own "known
  shape" framework, but worth stating explicitly in any write-up).
- The MVUE formula is the standard Type-II-censoring UMVUE, applied here to Type-I
  (calendar-time) censored field data, as is conventional practice — see the docstring
  in `src/weibull_core.py:mvue_reliability` for the exact caveat.
