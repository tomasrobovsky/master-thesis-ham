"""
Diagonal bootstrap weighting matrix for SMM.

Follows Kukacka & Zila (2023), Section 2.2: for each bootstrap iteration, draw
two independent overlapping-block samples from the empirical returns — one for
short-memory moments, one for long-memory moments (ACF at lags >= 10) — and
estimate the bootstrap variance of each empirical moment. The weighting matrix
is then

    W = diag(1/σ̂²₁, …, 1/σ̂²_D),

i.e. each moment is weighted by the inverse of its bootstrap variance. The
17-dimensional FWCHL17 set contains blocks of highly correlated moments
(absolute-return ACFs at neighbouring lags exceed 0.95), so the full inverse
covariance matrix Σ̂⁻¹ is severely ill-conditioned and produces poor
moment-level fits despite low aggregate loss; the diagonal specification avoids
this entirely. See Section 5.1.4 of the thesis.
"""

from __future__ import annotations

import numpy as np

from moments import _LONGMEM_LABELS, _SETS, compute_moments


def _overlapping_block_bootstrap(
    r: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one bootstrap sample of length T using overlapping block bootstrap.

    Randomly selects ceil(T / block_size) starting indices from the
    T − L + 1 possible overlapping blocks, concatenates them, and trims
    to the original series length T.
    """
    T = len(r)
    n_available = T - block_size + 1
    n_needed = int(np.ceil(T / block_size))

    starts = rng.integers(0, n_available, size=n_needed)
    blocks = [r[s : s + block_size] for s in starts]
    return np.concatenate(blocks)[:T]


def _has_longmem_moments(moment_set: str) -> bool:
    """Check if the moment set contains any long-memory moments."""
    labels = _SETS.get(moment_set, [])
    return bool(set(labels) & _LONGMEM_LABELS)


def compute_weighting_matrix(
    r: np.ndarray,
    moment_set: str = "FWCHL17",
    n_bootstrap: int = 5000,
    block_size: int = 250,
    seed: int = 42,
) -> np.ndarray:
    """Compute the diagonal bootstrap weighting matrix W = diag(1/σ̂²).

    Parameters
    ----------
    r : np.ndarray
        Empirical returns (1-D, length T).
    moment_set : str
        Moment set name — must match ``compute_moments`` / ``SMMConfig``.
    n_bootstrap : int
        Number of bootstrap replications B (default 5,000, as in the thesis).
    block_size : int
        Overlapping-block length L (default 250 trading days).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    W : np.ndarray, shape (D, D)
        Positive-definite diagonal weighting matrix.
    """
    r = np.asarray(r, dtype=float)
    T = len(r)

    if block_size >= T:
        raise ValueError(f"block_size ({block_size}) must be < T ({T}).")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    rng = np.random.default_rng(seed)
    use_two_samples = _has_longmem_moments(moment_set)

    bootstrap_moments: list[np.ndarray] = []

    for _ in range(n_bootstrap):
        r_boot_short = _overlapping_block_bootstrap(r, block_size, rng)
        r_boot_long = (
            _overlapping_block_bootstrap(r, block_size, rng)
            if use_two_samples
            else None
        )

        try:
            m_boot = compute_moments(
                r_boot_short,
                moment_set=moment_set,
                r_longmem=r_boot_long,
            )
            if np.all(np.isfinite(m_boot)):
                bootstrap_moments.append(m_boot)
        except Exception:
            continue  # skip rare degenerate samples

    B_valid = len(bootstrap_moments)
    if B_valid < 50:
        raise RuntimeError(
            f"Only {B_valid} valid bootstrap samples out of {n_bootstrap}. "
            "Check data or reduce block_size."
        )

    M = np.vstack(bootstrap_moments)  # (B_valid, D)

    variances = np.var(M, axis=0, ddof=1)
    if np.any(variances <= 0):
        raise RuntimeError(
            "Zero or negative variance encountered in bootstrap moments."
        )

    return np.diag(1.0 / variances)
