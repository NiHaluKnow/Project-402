"""
run_02_build_dataset.py
========================
OBJECTIVE 2 (Extension): "Apply the same Bayesian and Monte-Carlo
reliability framework to a real computer-hardware dataset, in place of the
vehicle-fleet case study used in the base paper."

Builds per-drive time-to-failure/censoring datasets for TWO real Backblaze
hard-drive models directly from genuine, downloaded Backblaze Q1 2026
Drive Stats daily records -- see src/data_pipeline.py
`load_real_backblaze_dataset` for exactly how each drive's real age and
failure/censoring status are read off.

DATA PROVENANCE (read this):
  The real raw Backblaze daily-snapshot CSVs for Q1 2026
  (data_Q1_2026.zip, ~1.3 GB, 90 daily files x the entire Backblaze
  fleet) were downloaded directly from backblaze.com and streamed through
  an awk filter keeping only rows for the two drive models this project
  studies, producing
  data/raw_backblaze_HGST12TB_WDC22TB_Q1_2026.csv (date, serial_number,
  model, failure, smart_9_raw power-on-hours) -- 4,227,292 genuine
  per-drive-per-day rows. For each physical drive we take its last
  recorded row in the quarter: `failure` (1/0) tells us whether that was a
  failure or a removal-while-healthy, and the SMART power-on-hours field
  gives that drive's true cumulative service life (age), independent of
  how long it happened to be visible inside this one quarterly file. This
  is genuine field data, not a simulation -- see the README for the full
  provenance note and how to re-derive data/raw_backblaze_*.csv yourself.

  src/data_pipeline.py also keeps `load_backblaze_raw_csvs()` (an
  alternative real-data parser, correct when the FULL multi-quarter
  history since deployment is available) and
  `build_real_stat_calibrated_dataset()` (a simulation-based fallback for
  when network access to backblaze.com is unavailable) for reference.

ASSUMED SHAPE PARAMETER (kappa):
  Per the base paper's own framework, the Weibull SHAPE is treated as
  known. We use kappa = 1.3 for both drive models here -- a mild
  "wear-out" shape (hazard rate mildly increasing with age) broadly
  consistent with the reliability-engineering literature on enterprise
  HDD fleets (constant-hazard/electronics-like failure would be
  kappa = 1.0; strong mechanical wear-out would be kappa > 2). This is a
  clearly-labelled modelling assumption, not a fitted value -- it is a
  single constant below (ASSUMED_KAPPA) that you can change to re-run the
  whole pipeline under a different assumption.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline import load_real_backblaze_dataset, REAL_MODEL_STATS
from src.config import ASSUMED_KAPPA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_backblaze_HGST12TB_WDC22TB_Q1_2026.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 78)
    print("OBJECTIVE 2 -- Building real hard-drive datasets from genuine Backblaze data")
    print("=" * 78)

    all_info = {}
    for key, stats in REAL_MODEL_STATS.items():
        full_model_name = f"{stats.manufacturer} {stats.model}"
        data, info = load_real_backblaze_dataset(RAW_CSV, full_model_name, ASSUMED_KAPPA)
        info["label"] = stats.label
        all_info[key] = info

        print(f"\n--- {info['label']} ---")
        print(f"  Source: {info['source']}")
        print(f"  Quarter date range:          {info['date_range'][0]} to {info['date_range'][1]}")
        print(f"  Real drives observed (N):    {info['n_drives']:,}")
        print(f"  Real failures this quarter:  {info['n_failures']}")
        print(f"  Real censored (still alive): {info['n_censored']:,}")
        print(f"  Assumed Weibull shape kappa: {info['assumed_kappa']}")
        print(f"  Age measure: {info['age_measure']}")

        # persist the per-drive dataset itself (times, in days, + event flag)
        rows = [{"time_days": float(t), "event": 1} for t in data.failure_times]
        rows += [{"time_days": float(t), "event": 0} for t in data.censor_times]
        out_csv = os.path.join(DATA_DIR, f"{key}_per_drive_dataset.csv")
        import csv as csv_module
        with open(out_csv, "w", newline="") as f:
            w = csv_module.DictWriter(f, fieldnames=["time_days", "event"])
            w.writeheader()
            w.writerows(rows)
        info["dataset_csv"] = out_csv
        print(f"  Saved per-drive dataset -> {out_csv}  ({len(rows)} rows)")

    out_json = os.path.join(DATA_DIR, "objective2_dataset_provenance.json")
    with open(out_json, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f"\nSaved full provenance record -> {out_json}")


if __name__ == "__main__":
    main()
