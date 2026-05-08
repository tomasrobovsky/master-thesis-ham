"""
Moment functions for SMM estimation.

Provides the two moment sets used in the thesis:
  - **FWCHL17** (17 moments) — main analysis (H1, H2)
  - **FWCHL16** (16 moments) — H3 short sub-periods (drops ABS-AC50)

Both sets follow the taxonomy of Kukacka & Zila (2023, JEBO) and combine
the Chen, He & Lux (2018) autocorrelation structure with the Franke &
Westerhoff (2012) tail and long-memory coverage. ABS-MEAN is excluded
because it is highly collinear with RAW-VAR and ill-conditions the
weighting matrix.

All autocorrelation-type moments for |r| and r² are *smoothed* over
neighbouring lags following Franke & Westerhoff (2011, 2012, 2016):
  lag k > 1 → mean of lags {k−1, k, k+1}
  lag k = 1 → mean of lags {1, 2}
Raw-return autocorrelations are NOT smoothed.

Moments use standardised autocorrelation coefficients ρ(k),
matching the Julia reference implementation (StatsBase.autocor).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Moment-set definitions
# ---------------------------------------------------------------------------

# FWCHL17: thesis default for H1 and H2.
_FWCHL17_LABELS: list[str] = [
    "RAW-VAR", "RAW-KURT", "RAW-AC1",
    "ABS-HILL5",
    "ABS-AC1", "ABS-AC5", "ABS-AC10", "ABS-AC15", "ABS-AC20", "ABS-AC25",
    "ABS-AC50",
    "SQR-AC1", "SQR-AC5", "SQR-AC10", "SQR-AC15", "SQR-AC20", "SQR-AC25",
]

# FWCHL16: FWCHL17 without ABS-AC50. Used for short samples (e.g., H3
# pre/post-ETF sub-periods with T ≈ 363–496) where the lag-50
# autocorrelation cannot be reliably preserved by the block bootstrap.
_FWCHL16_LABELS: list[str] = [
    "RAW-VAR", "RAW-KURT", "RAW-AC1",
    "ABS-HILL5",
    "ABS-AC1", "ABS-AC5", "ABS-AC10", "ABS-AC15", "ABS-AC20", "ABS-AC25",
    "SQR-AC1", "SQR-AC5", "SQR-AC10", "SQR-AC15", "SQR-AC20", "SQR-AC25",
]

_SETS: dict[str, list[str]] = {
    "FWCHL16": _FWCHL16_LABELS,
    "FWCHL17": _FWCHL17_LABELS,
}

# Long-memory moments (ACF at lags >= 10): computed from a separate bootstrap
# series in the weighting matrix, following Kukacka & Zila (2023).
_LONGMEM_LABELS: set[str] = {
    "ABS-AC10", "ABS-AC15", "ABS-AC20", "ABS-AC25", "ABS-AC50",
    "SQR-AC10", "SQR-AC15", "SQR-AC20", "SQR-AC25",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray, lag: int) -> float:
    """Standardised autocorrelation ρ(k), matching Julia StatsBase.autocor.

    ρ(k) = Σ(x_t - μ)(x_{t-k} - μ) / Σ(x_t - μ)²
    """
    T = len(x)
    if lag >= T:
        return np.nan
    mu = np.mean(x)
    x_dm = x - mu
    numer = np.sum(x_dm[lag:] * x_dm[: T - lag])
    denom = np.sum(x_dm ** 2)
    if denom == 0:
        return np.nan
    return float(numer / denom)


def _smoothed_acf(x: np.ndarray, lag: int) -> float:
    """Smoothed autocorrelation: average ρ over neighbouring lags.

    lag = 1  → mean of lags {1, 2}
    lag > 1  → mean of lags {lag-1, lag, lag+1}
    """
    if lag == 1:
        neighbours = [1, 2]
    else:
        neighbours = [lag - 1, lag, lag + 1]
    vals = [_autocorrelation(x, k) for k in neighbours if k < len(x)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _hill_estimator(abs_r: np.ndarray, quantile: float) -> float:
    """Hill tail-index estimator α̂ for top *quantile* fraction of |r|.

    Returns the tail index α̂ = 1/γ̂, matching the Julia reference:
        α̂ = ((1/k) Σ_{i=1}^{k} [ ln v_{(i)} − ln v_{(k+1)} ])^{-1}
    where v_{(1)} ≥ … ≥ v_{(T)} are the order statistics of |r|.
    """
    sorted_abs = np.sort(abs_r)[::-1]  # descending
    k = max(1, int(np.floor(quantile * len(abs_r))))
    if k >= len(sorted_abs):
        k = len(sorted_abs) - 1
    if sorted_abs[k] <= 0:
        return np.nan
    log_threshold = np.log(sorted_abs[k])
    gamma = float(np.mean(np.log(sorted_abs[:k]) - log_threshold))
    if gamma <= 0:
        return np.nan
    return 1.0 / gamma  # tail index α̂


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_moments(
    r: np.ndarray,
    moment_set: str = "FWCHL17",
    return_names: bool = False,
    r_longmem: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Compute a moment vector from a return series.

    Parameters
    ----------
    r : np.ndarray
        1-D array of log-returns.
    moment_set : str
        ``"FWCHL17"`` (17 moments, thesis default for H1 and H2) or
        ``"FWCHL16"`` (16 moments, FWCHL17 without ABS-AC50; used for
        short H3 sub-periods).
    return_names : bool
        If *True*, also return the list of moment labels.
    r_longmem : np.ndarray or None
        Optional separate return series for long-memory moments (ACF at
        lags >= 10).  Following Kukacka & Zila (2023), the weighting-matrix
        bootstrap draws two independent block-bootstrap samples: one for
        short-memory moments and one for long-memory moments.  When
        *r_longmem* is provided, moments in ``_LONGMEM_LABELS`` are
        computed from this series instead of *r*.

    Returns
    -------
    m : np.ndarray   (or tuple (m, names) if *return_names*)
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]

    if r.ndim != 1:
        raise ValueError("Returns r must be a 1-D array.")
    if len(r) < 5:
        raise ValueError("Returns series too short to compute moments.")

    if moment_set not in _SETS:
        raise ValueError(
            f"Unknown moment_set '{moment_set}'. Choose from {list(_SETS)}."
        )

    # Prepare long-memory series if provided
    if r_longmem is not None:
        r_lm = np.asarray(r_longmem, dtype=float)
        r_lm = r_lm[np.isfinite(r_lm)]
        abs_r_lm = np.abs(r_lm)
        sqr_r_lm = r_lm ** 2
    else:
        r_lm = None

    labels = _SETS[moment_set]
    abs_r = np.abs(r)
    sqr_r = r ** 2

    # Cache of already-computed moments
    _cache: dict[str, float] = {}

    def _get(label: str) -> float:
        if label in _cache:
            return _cache[label]
        # Long-memory moments use separate series when available
        if r_lm is not None and label in _LONGMEM_LABELS:
            val = _compute_one(label, r_lm, abs_r_lm, sqr_r_lm)
        else:
            val = _compute_one(label, r, abs_r, sqr_r)
        _cache[label] = val
        return val

    m_list = [_get(lab) for lab in labels]
    m = np.array(m_list, dtype=float)

    if return_names:
        return m, list(labels)
    return m


def _compute_one(
    label: str,
    r: np.ndarray,
    abs_r: np.ndarray,
    sqr_r: np.ndarray,
) -> float:
    """Compute a single named moment."""

    # --- raw return moments (no smoothing) --------------------------------
    if label == "RAW-VAR":
        return float(np.var(r, ddof=1))
    if label == "RAW-KURT":
        mu = np.mean(r)
        sigma2 = np.var(r, ddof=0)
        if sigma2 == 0:
            return np.nan
        return float(np.mean((r - mu) ** 4) / sigma2 ** 2)
    if label == "RAW-AC1":
        return _autocorrelation(r, 1)

    # --- absolute return moments ------------------------------------------
    if label == "ABS-HILL5":
        return _hill_estimator(abs_r, 0.05)

    # Smoothed standardised ACF of absolute returns
    _ABS_AC_LAGS = {
        "ABS-AC1": 1, "ABS-AC5": 5, "ABS-AC10": 10,
        "ABS-AC15": 15, "ABS-AC20": 20, "ABS-AC25": 25,
        "ABS-AC50": 50,
    }
    if label in _ABS_AC_LAGS:
        return _smoothed_acf(abs_r, _ABS_AC_LAGS[label])

    # Smoothed standardised ACF of squared returns
    _SQR_AC_LAGS = {
        "SQR-AC1": 1, "SQR-AC5": 5, "SQR-AC10": 10,
        "SQR-AC15": 15, "SQR-AC20": 20, "SQR-AC25": 25,
    }
    if label in _SQR_AC_LAGS:
        return _smoothed_acf(sqr_r, _SQR_AC_LAGS[label])

    raise ValueError(f"Unknown moment label: {label}")
