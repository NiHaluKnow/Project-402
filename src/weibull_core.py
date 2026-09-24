"""
weibull_core.py
================
Core Weibull time-to-failure model and classical (MLE / MVUE) reliability
estimators, following the parameterisation used in the base paper:

    Arekar, K., Jain, R., and Kumar, S. (2021). "Bayesian Estimation of
    System Reliability Models Using Monte-Carlo Technique of Simulation."
    Journal of Statistical Theory and Applications, 20(1), 149-163.
    https://doi.org/10.2991/jsta.d.210201.001

PARAMETERISATION (matches the paper's Eq. 22, Section 7)
----------------------------------------------------------
Component / unit time-to-failure T has pdf

    f(t; beta, kappa) = beta * t**(kappa - 1) * exp(-beta * t**kappa / kappa),   t > 0

    kappa  = SHAPE parameter (paper's alpha_i + 1)  -- assumed KNOWN
    beta   = intensity / "rate" parameter (paper's beta_i) -- UNKNOWN, the
             quantity the whole project places priors on / estimates.

This is an ordinary Weibull distribution; if you prefer the textbook
(shape kappa, scale eta) parameterisation, the two are related by

    beta = kappa / eta**kappa        <=>        eta = (kappa / beta) ** (1 / kappa)

Reliability (survival) function:

    R(t; beta) = P(T > t) = exp(-beta * t**kappa / kappa)

SUFFICIENT STATISTIC
---------------------
For a sample of n units observed under (Type-I or Type-II) right censoring,
with r observed failures at times {t_i} and (n - r) units censored (removed
from observation while still working) at times {c_j}, the Weibull-known-shape
likelihood is

    L(beta) = beta**r * (prod t_i**(kappa-1)) * exp(-beta * S / kappa)

    where S = sum_{failures} t_i**kappa + sum_{censored} c_j**kappa

S (and r) are jointly sufficient for beta. Every estimator below is a
function of (r, S, kappa) alone -- this is exactly the "s" statistic that
appears throughout the base paper (Eqs. 1-19 there use "s = sum x_i" in
precisely this role for their special cases kappa=1 and kappa=2).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FailureData:
    """A right-censored sample of time-to-failure observations for units
    that all share the same (known) Weibull shape kappa and (unknown) rate
    beta.

    failure_times : times at which a unit was observed to fail
    censor_times  : times at which a unit was last observed still working
                    (removed from service / study ended / still alive)
    kappa         : known Weibull shape parameter
    """

    failure_times: np.ndarray
    censor_times: np.ndarray
    kappa: float

    def __post_init__(self):
        self.failure_times = np.asarray(self.failure_times, dtype=float)
        self.censor_times = np.asarray(self.censor_times, dtype=float)
        if np.any(self.failure_times <= 0) or np.any(self.censor_times <= 0):
            raise ValueError("All times must be strictly positive.")
        if self.kappa <= 0:
            raise ValueError("kappa (shape) must be positive.")

    # ---- sufficient statistics -------------------------------------------------
    @property
    def r(self) -> int:
        """Number of observed failures."""
        return len(self.failure_times)

    @property
    def n(self) -> int:
        """Total number of units (failed + censored)."""
        return len(self.failure_times) + len(self.censor_times)

    @property
    def S(self) -> float:
        """Sufficient statistic S = sum t_i^kappa over ALL units (failed and
        censored)."""
        k = self.kappa
        return float(
            np.sum(self.failure_times**k) + np.sum(self.censor_times**k)
        )

    def summary(self) -> dict:
        return {
            "n_total": self.n,
            "n_failed": self.r,
            "n_censored": len(self.censor_times),
            "kappa": self.kappa,
            "S": self.S,
        }


# --------------------------------------------------------------------------- #
# Weibull pdf / reliability (given beta, kappa)
# --------------------------------------------------------------------------- #
def weibull_pdf(t, beta, kappa):
    t = np.asarray(t, dtype=float)
    return beta * t ** (kappa - 1.0) * np.exp(-beta * t**kappa / kappa)


def weibull_reliability(t, beta, kappa):
    """R(t; beta, kappa) = exp(-beta * t^kappa / kappa)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-beta * t**kappa / kappa)


def beta_to_scale(beta, kappa):
    """Convert the paper's rate parameter beta to the textbook Weibull scale
    eta, for reporting purposes only."""
    return (kappa / beta) ** (1.0 / kappa)


def scale_to_beta(eta, kappa):
    """Inverse of beta_to_scale."""
    return kappa / eta**kappa


# --------------------------------------------------------------------------- #
# Classical estimators
# --------------------------------------------------------------------------- #
def mle_beta(data: FailureData) -> float:
    """Maximum-likelihood estimate of beta.

    Derivation: log L(beta) = r*log(beta) - beta*S/kappa + const.
    d/d(beta) = r/beta - S/kappa = 0  =>  beta_hat = r*kappa / S.
    """
    if data.r == 0:
        return 0.0
    return data.r * data.kappa / data.S


def mle_reliability(t, data: FailureData):
    """Plug-in MLE of the reliability curve: R_hat(t) = exp(-beta_hat*t^k/k)
                                                        = exp(-r * t^k / S).
    This matches the structural form of the paper's own MLE[r(t)] results
    (e.g. Eq. in Section 2, "MLE[r(t)] = exp(-m^2/s^2)" for the kappa=2
    special case).
    """
    t = np.asarray(t, dtype=float)
    if data.r == 0:
        return np.ones_like(t)
    return np.exp(-data.r * t**data.kappa / data.S)


def mvue_reliability(t, data: FailureData):
    """Uniform minimum variance unbiased estimator (UMVUE) of R(t) for a
    Weibull sample with known shape kappa, under (approximate) Type-II
    censoring with r observed failures and sufficient statistic S.

        R_MVUE(t) = (1 - t^kappa / S) ** (r - 1),   0 <= t^kappa < S
                   = 0,                              otherwise

    This is the standard textbook UMVUE for this model (see e.g. Bain &
    Engelhardt, "Statistical Analysis of Reliability and Life-Testing
    Models", 2nd ed., Ch. 4; Lawless, "Statistical Models and Methods for
    Lifetime Data"). It generalises the two closed-form special cases given
    explicitly in the base paper:

      * kappa = 1 (exponential):     the paper's Sec. 4 gives a related
        closed form under a *time-truncated Poisson* sampling scheme,
        r_hat(t) = (1 - t/t_s)^x  (note: exponent x, not x-1, because that
        section uses a different -- Type-I / counting-process -- sampling
        scheme to ours).
      * kappa = 2 (Rayleigh):        the paper's Sec. 2 gives exactly this
        formula with exponent (n-1) for a fully-observed (r = n) sample:
        r_hat(t) = (1 - t^2/s^2)^(n-1).

    Real, randomly-censored field data (like the Backblaze drive-days used
    in this project) follow Type-I censoring rather than the Type-II scheme
    the UMVUE is derived under; we use it here as the conventional
    "plug-in" classical estimator, exactly as reliability-engineering
    practice normally does, and flag this modelling assumption in the
    project README.
    """
    t = np.asarray(t, dtype=float)
    r = data.r
    S = data.S
    out = np.zeros_like(t)
    if r >= 1:
        base = 1.0 - t**data.kappa / S
        valid = base > 0
        out[valid] = base[valid] ** (r - 1)
    return out


def mse_mvue_analytic(t, data: FailureData):
    """Approximate analytic MSE curve for the MVUE, using the same
    large-r Gamma-based approximation style used for the paper's own
    negative-exponential MSE formula (Eq. 16), adapted to general kappa.
    Provided for diagnostic / plotting purposes only (an exact closed form
    is not tractable for general kappa); we fall back to nan when r is too
    small for the approximation to be meaningful.
    """
    t = np.asarray(t, dtype=float)
    r = data.r
    if r < 2:
        return np.full_like(t, np.nan)
    beta_hat = mle_beta(data)
    x = t**data.kappa
    mean_x = data.kappa / beta_hat  # E[T^kappa] under the MLE plug-in
    # crude variance approximation of R_hat via delta method around beta_hat
    var_beta_hat = beta_hat**2 / r
    dR_dbeta = -x / data.kappa * np.exp(-beta_hat * x / data.kappa)
    return (dR_dbeta**2) * var_beta_hat
