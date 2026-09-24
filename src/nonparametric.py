"""
nonparametric.py
=================
Kaplan-Meier (product-limit) estimator of the reliability / survival curve.

Used as a MODEL-FREE reference curve when comparing MLE / MVUE / Bayes
reliability estimates on the real-calibrated hard-drive dataset, where
(unlike the pure-synthetic validation in Objective 1) the "true" beta is not
known and so accuracy must be judged against a nonparametric benchmark
rather than a known ground truth.
"""

from __future__ import annotations
import numpy as np

from .weibull_core import FailureData


def kaplan_meier(data: FailureData):
    """Return (times, survival) step-function points for the Kaplan-Meier
    estimator built from a FailureData object's failure and censoring
    times.
    """
    events = np.concatenate(
        [
            np.column_stack([data.failure_times, np.ones_like(data.failure_times)]),
            np.column_stack([data.censor_times, np.zeros_like(data.censor_times)]),
        ]
    )
    order = np.argsort(events[:, 0])
    events = events[order]

    n_at_risk = data.n
    survival = 1.0
    times = [0.0]
    surv = [1.0]

    i = 0
    N = len(events)
    while i < N:
        t = events[i, 0]
        # process all ties at this exact time together
        deaths = 0
        removed = 0
        while i < N and events[i, 0] == t:
            if events[i, 1] == 1:
                deaths += 1
            else:
                removed += 1
            i += 1
        if deaths > 0:
            survival *= 1.0 - deaths / n_at_risk
            times.append(t)
            surv.append(survival)
        n_at_risk -= deaths + removed

    return np.array(times), np.array(surv)


def km_at(t_query, km_times, km_surv):
    """Step-function evaluation of a Kaplan-Meier curve at arbitrary query
    times (right-continuous step function, KM convention)."""
    t_query = np.asarray(t_query, dtype=float)
    idx = np.searchsorted(km_times, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return km_surv[idx]
