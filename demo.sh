#!/bin/bash
###############################################################################
# demo.sh
#
# One-shot demonstration script for
#   "Bayesian Reliability Estimation of Computer Hardware Systems Using
#    Monte-Carlo Simulation"  (re-implementation of Arekar, Jain & Kumar,
#   JSTA 2021, extended to real Backblaze hard-drive field data).
#
# Runs the full pipeline end-to-end and prints a plain-English summary of
# the headline result for each of the four proposal objectives:
#
#   1. Re-implementation & validation   -> run_scripts/run_01_validate_implementation.py
#   2. Real hardware dataset (extension) -> run_scripts/run_02_build_dataset.py
#   3. Bayes vs. classical comparison    -> run_scripts/run_03_fit_and_compare.py
#   4. Prior sensitivity                 -> run_scripts/run_04_prior_sensitivity.py
#
# Each stage also writes its own JSON metrics + PNG plots to results/, and
# stage 2 writes the per-drive datasets to data/ (see README.md for full
# detail on every number these scripts produce).
#
# USAGE
#   ./demo.sh              # create/use a local .venv, install deps, run everything
#   ./demo.sh --no-venv    # use whatever `python3` is already on PATH instead
#                          # (you must have installed requirements.txt yourself)
#
# Safe to re-run any time: every stage is deterministic (fixed random seeds
# in src/config.py) and overwrites its own output files in place.
###############################################################################

set -euo pipefail

# --- resolve paths so this script works no matter where it's invoked from ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_VENV=1
if [[ "${1:-}" == "--no-venv" ]]; then
  USE_VENV=0
fi

# small helper: a visually distinct section banner for each pipeline stage
section() {
  echo
  echo "==============================================================================="
  echo "$1"
  echo "==============================================================================="
}

# ---------------------------------------------------------------------------
# 0. Environment setup
#    By default this creates/reuses a local .venv so the demo works even on
#    a machine with no numpy/scipy/pandas/matplotlib installed system-wide.
# ---------------------------------------------------------------------------
section "SETUP — Python environment"
if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    echo ">> No .venv found — creating one..."
    python3 -m venv .venv
  fi
  PYTHON=".venv/bin/python"
  echo ">> Installing/verifying dependencies (numpy, scipy, pandas, matplotlib)..."
  "$PYTHON" -m pip install -q -r requirements.txt
else
  PYTHON="python3"
  echo ">> Using system python3 (assuming requirements.txt is already installed)."
fi
echo ">> Using interpreter: $PYTHON ($("$PYTHON" --version))"

# ---------------------------------------------------------------------------
# 1. OBJECTIVE 1 -- Re-implementation & validation
#    Simulates data from KNOWN Weibull parameters and checks that MLE /
#    MVUE / Bayes(quadrature) / Bayes(paper's own Monte-Carlo, Eq. 29-30)
#    all recover the true reliability curve, and agree with each other.
# ---------------------------------------------------------------------------
section "OBJECTIVE 1 -- Re-implementation & validation of the Weibull/Bayes/Monte-Carlo pipeline"
"$PYTHON" run_scripts/run_01_validate_implementation.py

# ---------------------------------------------------------------------------
# 2. OBJECTIVE 2 -- Extension to a real computer-hardware dataset
#    Builds per-drive time-to-failure/censoring data for two real Backblaze
#    hard-drive models from genuinely downloaded field data
#    (data/raw_backblaze_HGST12TB_WDC22TB_Q1_2026.csv). See README.md
#    section 3 for exactly how that raw file was obtained, and section 3.1
#    for how to re-derive it yourself or extend to other quarters/models.
# ---------------------------------------------------------------------------
section "OBJECTIVE 2 -- Building the real hard-drive dataset (genuine Backblaze field data)"
RAW_CSV="data/raw_backblaze_HGST12TB_WDC22TB_Q1_2026.csv"
if [[ ! -f "$RAW_CSV" ]]; then
  echo "!! Real Backblaze extract not found at: $RAW_CSV"
  echo "!! See README.md section 3.1 for how to download/re-derive it."
  exit 1
fi
"$PYTHON" run_scripts/run_02_build_dataset.py

# ---------------------------------------------------------------------------
# 3. OBJECTIVE 3 -- Comparative analysis: Bayes vs. classical (MLE/MVUE)
#    Runs both a FULL-DATA scenario (thousands of real drives) and a
#    LIMITED-DATA scenario (a 50-drive subsample, mimicking early field
#    data) and compares all three estimators against a Kaplan-Meier
#    reference curve built from the full real dataset.
# ---------------------------------------------------------------------------
section "OBJECTIVE 3 -- Bayes vs. classical (MLE/MVUE) comparison on real data"
"$PYTHON" run_scripts/run_03_fit_and_compare.py

# ---------------------------------------------------------------------------
# 4. OBJECTIVE 4 -- Prior sensitivity: Rayleigh vs. Beta vs. Uniform
#    Compares the three priors on the Weibull rate parameter, in both the
#    full-data and limited-data regimes, and reports how much they
#    disagree (max pairwise |R(t)| difference).
# ---------------------------------------------------------------------------
section "OBJECTIVE 4 -- Prior sensitivity (Rayleigh vs. Beta vs. Uniform)"
"$PYTHON" run_scripts/run_04_prior_sensitivity.py

# ---------------------------------------------------------------------------
# 5. Headline numbers -- pull the key figures out of the JSON metrics files
#    so they're visible right here in the terminal, without having to open
#    results/*.json by hand.
# ---------------------------------------------------------------------------
section "HEADLINE NUMBERS (pulled from results/*.json)"
"$PYTHON" - <<'PYEOF'
import json, os

RESULTS = "results"

def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)

print("\n[Objective 1] Parameter recovery at n=5000 (large sample):")
m = next(x for x in load("objective1_validation_metrics.json") if x["tag"] == "large_sample_n5000")
print(f"  beta_true={m['beta_true']}  beta_hat(MLE)={m['beta_mle_hat']:.5f}"
      f"  (relative error {m['relative_error_beta_mle_pct']:.2f}%)")
print(f"  Bayes(quadrature) vs Bayes(paper's Monte-Carlo) RMSE agreement: "
      f"{m['rmse_BayesMonteCarlo_vs_BayesQuadrature']:.5f}")

print("\n[Objective 3] Bayes vs classical, RMSE against Kaplan-Meier reference:")
for m in load("objective3_comparison_metrics.json"):
    print(f"  {m['tag']:28s}  n={m['n_fit']:<6d} failures={m['r_fit_failures']:<4d}"
          f"  MLE={m['rmse_MLE_vs_KM']:.4f}  MVUE={m['rmse_MVUE_vs_KM']:.4f}"
          f"  Bayes={m['rmse_Bayes_vs_KM']:.4f}")

print("\n[Objective 4] Prior sensitivity, max pairwise |R(t)| difference across priors:")
for m in load("objective4_prior_sensitivity_metrics.json"):
    print(f"  {m['tag']:28s}  n={m['n_fit']:<6d} failures={m['r_fit_failures']:<4d}"
          f"  max_diff={m['max_pairwise_abs_difference_in_R']:.5f}")
PYEOF

# ---------------------------------------------------------------------------
# 6. Where everything landed
# ---------------------------------------------------------------------------
section "DONE -- generated artifacts"
echo "Numeric results (JSON):"
ls -1 results/*.json
echo
echo "Plots (PNG):"
ls -1 results/*.png
echo
echo "Per-drive datasets built from real data:"
ls -1 data/*_per_drive_dataset.csv
echo
echo "Full provenance record for the real Backblaze dataset:"
echo "  data/objective2_dataset_provenance.json"
echo
echo "See README.md for the full methodology write-up and discussion of every result."
