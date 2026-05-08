"""
Synchronous block bootstrap for cross-market Wald tests (H2).

Generates pseudo-return series for two markets simultaneously using
identical block indices — ensuring the cross-covariance structure of
parameter estimates is captured (Taylor & McGuire, 2005).

Usage
-----
From a calibration script (e.g., on Metacentrum)::

    from block_bootstrap import bootstrap_pair

    r_eq, r_cr = bootstrap_pair(r_equity, r_crypto, boot_id=42)
    # ... calibrate model on r_eq and r_cr independently ...

The same boot_id always produces the same block indices, so bootstrap
data does not need to be stored — it is regenerated deterministically.
"""

from __future__ import annotations

import numpy as np


# ── core functions ──────────────────────────────────────────────────

def _block_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw contiguous-block indices that tile a series of length T.

    Randomly selects ceil(T / block_size) block-start positions,
    concatenates the resulting ranges, and trims to exactly T.
    """
    n_blocks = int(np.ceil(T / block_size))
    max_start = T - block_size          # inclusive upper bound
    if max_start < 0:
        raise ValueError(
            f"block_size ({block_size}) exceeds series length ({T})"
        )
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    indices = np.concatenate(
        [np.arange(s, s + block_size) for s in starts]
    )
    return indices[:T]


def bootstrap_single(
    r: np.ndarray,
    boot_id: int,
    block_size: int = 40,
    master_seed: int = 20260403,
) -> np.ndarray:
    """Generate one block bootstrap replication for a single market.

    Uses identical index logic as bootstrap_pair — given the same T,
    block_size, master_seed and boot_id, the block indices are the same.
    This allows calibrating markets independently while preserving
    synchronous bootstrap structure for the Wald test.
    """
    r = np.asarray(r, dtype=float)
    T = len(r)
    rng = np.random.default_rng(master_seed + boot_id)
    idx = _block_indices(T, block_size, rng)
    return r[idx]


def bootstrap_pair(
    r_a: np.ndarray,
    r_b: np.ndarray,
    boot_id: int,
    block_size: int = 40,
    master_seed: int = 20260403,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synchronous bootstrap replication for two markets.

    Parameters
    ----------
    r_a, r_b : np.ndarray
        Aligned return series of equal length T.
    boot_id : int
        Replication index (1 … B).  Deterministically sets the RNG seed.
    block_size : int
        Length of contiguous blocks (default 40 ≈ 2 trading months).
    master_seed : int
        Base seed.  Effective seed = master_seed + boot_id.

    Returns
    -------
    (r_a_star, r_b_star) : tuple of np.ndarray
        Bootstrap pseudo-return series, each of length T.
    """
    r_a = np.asarray(r_a, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    if r_a.shape != r_b.shape:
        raise ValueError(
            f"Series lengths differ: {len(r_a)} vs {len(r_b)}"
        )
    T = len(r_a)

    rng = np.random.default_rng(master_seed + boot_id)
    idx = _block_indices(T, block_size, rng)

    return r_a[idx], r_b[idx]


def bootstrap_all(
    r_a: np.ndarray,
    r_b: np.ndarray,
    B: int = 500,
    block_size: int = 40,
    master_seed: int = 20260403,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all B bootstrap replications at once.

    Useful for local inspection / diagnostics.  On Metacentrum, use
    ``bootstrap_pair`` with a single boot_id per job instead.

    Returns
    -------
    (R_a, R_b) : tuple of np.ndarray, each shape (B, T)
    """
    r_a = np.asarray(r_a, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    T = len(r_a)

    R_a = np.empty((B, T), dtype=float)
    R_b = np.empty((B, T), dtype=float)

    for b in range(B):
        R_a[b], R_b[b] = bootstrap_pair(
            r_a, r_b,
            boot_id=b + 1,
            block_size=block_size,
            master_seed=master_seed,
        )
    return R_a, R_b


# ── diagnostics ─────────────────────────────────────────────────────

def check_bootstrap_properties(
    r_orig: np.ndarray,
    R_boot: np.ndarray,
    label: str = "",
) -> dict:
    """Compare original vs bootstrap moments (quick sanity check).

    Parameters
    ----------
    r_orig : (T,) original returns
    R_boot : (B, T) bootstrap replications

    Returns
    -------
    dict with mean, std, kurtosis for original and bootstrap (mean ± std across B).
    """
    from scipy.stats import kurtosis as sp_kurt

    orig_mean = float(np.mean(r_orig))
    orig_std = float(np.std(r_orig, ddof=1))
    orig_kurt = float(sp_kurt(r_orig, fisher=True))

    boot_means = np.mean(R_boot, axis=1)
    boot_stds = np.std(R_boot, axis=1, ddof=1)
    boot_kurts = np.array([float(sp_kurt(row, fisher=True)) for row in R_boot])

    result = {
        "label": label,
        "T": len(r_orig),
        "B": R_boot.shape[0],
        "orig_mean": orig_mean,
        "orig_std": orig_std,
        "orig_kurt": orig_kurt,
        "boot_mean": f"{boot_means.mean():.6f} ± {boot_means.std():.6f}",
        "boot_std": f"{boot_stds.mean():.6f} ± {boot_stds.std():.6f}",
        "boot_kurt": f"{boot_kurts.mean():.2f} ± {boot_kurts.std():.2f}",
    }
    return result


# ── CLI for quick testing ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Generate & inspect synchronous block bootstrap replications."
    )
    parser.add_argument("--data-a", required=True, help="CSV path for market A (cols: date, r)")
    parser.add_argument("--data-b", required=True, help="CSV path for market B (cols: date, r)")
    parser.add_argument("--B", type=int, default=10, help="Number of replications (default 10)")
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20260403)
    args = parser.parse_args()

    r_a = pd.read_csv(args.data_a)["r"].to_numpy(dtype=float)
    r_b = pd.read_csv(args.data_b)["r"].to_numpy(dtype=float)

    print(f"T = {len(r_a)}, B = {args.B}, block_size = {args.block_size}")
    print()

    R_a, R_b = bootstrap_all(r_a, r_b, B=args.B,
                              block_size=args.block_size,
                              master_seed=args.master_seed)

    for label, r_orig, R_boot in [("Market A", r_a, R_a), ("Market B", r_b, R_b)]:
        info = check_bootstrap_properties(r_orig, R_boot, label)
        print(f"--- {label} ---")
        print(f"  Original:  mean={info['orig_mean']:.6f}  std={info['orig_std']:.6f}  kurt={info['orig_kurt']:.2f}")
        print(f"  Bootstrap: mean={info['boot_mean']}  std={info['boot_std']}  kurt={info['boot_kurt']}")
        print()
