"""
plotting.py
===========
Matplotlib helpers for reliability curves, MLE/MVUE/Bayes comparisons, and
prior-sensitivity plots. All functions save a PNG to `outpath` and also
return the Figure in case the caller wants to display it inline (e.g. in a
notebook).
"""

from __future__ import annotations
import textwrap
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _wrap(title, width=68):
    return "\n".join(textwrap.wrap(title, width=width, break_long_words=False))


def plot_validation_recovery(t, R_true, R_mle, R_mvue, R_bayes_quad, R_bayes_mc, title, outpath):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, R_true, "k-", lw=2.6, label="True R(t)  (known simulation parameters)")
    ax.plot(t, R_mle, "--", lw=1.8, label="MLE plug-in")
    ax.plot(t, R_mvue, ":", lw=2.0, label="MVUE")
    ax.plot(t, R_bayes_quad, "-", lw=1.8, alpha=0.9, label="Bayes (quadrature)")
    ax.plot(t, R_bayes_mc, "o", ms=3.5, alpha=0.7, label="Bayes (Monte-Carlo, Eq. 29-30)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reliability R(t)")
    ax.set_title(_wrap(title), fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def plot_estimator_comparison(t, curves: dict, km_times, km_surv, title, outpath, xlabel="Time (days)"):
    """curves: dict of {label: array-like R(t)} to overlay, e.g.
    {'MLE': ..., 'MVUE': ..., 'Bayes (Rayleigh)': ...}."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.step(km_times, km_surv, where="post", color="k", lw=1.6, alpha=0.55, label="Kaplan-Meier (nonparametric)")
    styles = ["-", "--", "-.", ":"]
    for i, (label, R) in enumerate(curves.items()):
        ax.plot(t, R, styles[i % len(styles)], lw=2.0, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reliability R(t)")
    ax.set_title(_wrap(title), fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def plot_prior_sensitivity(t, prior_curves: dict, prior_intervals: dict, title, outpath, xlabel="Time (days)"):
    """prior_curves: {prior_name: R(t) array}
    prior_intervals: {prior_name: (lower, upper) arrays} for shaded credible bands.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, R) in enumerate(prior_curves.items()):
        c = colors[i % len(colors)]
        ax.plot(t, R, lw=2.2, color=c, label=f"{name} prior")
        if name in prior_intervals:
            lo, hi = prior_intervals[name]
            ax.fill_between(t, lo, hi, color=c, alpha=0.15, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bayes reliability estimate  r~(t)")
    ax.set_title(_wrap(title), fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def plot_rmse_bar(labels, rmse_values, title, outpath):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, rmse_values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)])
    ax.set_ylabel("RMSE vs Kaplan-Meier")
    ax.set_title(_wrap(title), fontsize=10)
    for b, v in zip(bars, rmse_values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath
