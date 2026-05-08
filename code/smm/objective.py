from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np

from moments import compute_moments


@dataclass(frozen=True)
class SMMConfig:
    T: int
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    moment_set: str = "FWCHL17"
    burn_in: int = 0


def _sim_one_seed(model, theta, T_total, burn_in, moment_set, seed):
    """Simulate one seed and return moment vector (top-level for pickling)."""
    r = model.simulate(theta, T=T_total, seed=seed)
    if burn_in > 0:
        if len(r) <= burn_in:
            return None
        r = r[burn_in:]
    m = compute_moments(r, moment_set=moment_set, return_names=False)
    if not np.all(np.isfinite(m)):
        return None
    return m


def simulated_moments(model, theta, cfg: SMMConfig, return_names: bool = False,
                      parallel_seeds: int = 0):
    """
    For given theta: simulate S times (fixed seeds), compute moments each time,
    return average moments across seeds.

    Parameters
    ----------
    parallel_seeds : int
        0 or 1: sequential (default).  -1: use all CPU cores.
        N > 1: use N parallel workers for seed evaluation.
    """
    T_total = int(cfg.T) + int(cfg.burn_in)
    if T_total <= 0:
        raise ValueError("SMMConfig.T + burn_in must be positive.")
    if cfg.burn_in < 0:
        raise ValueError("SMMConfig.burn_in must be >= 0.")

    use_parallel = parallel_seeds != 0 and parallel_seeds != 1 and len(cfg.seeds) > 1

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor
        n_workers = None if parallel_seeds == -1 else parallel_seeds
        worker = partial(_sim_one_seed, model, theta, T_total, cfg.burn_in, cfg.moment_set)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker, cfg.seeds))
        ms = [m for m in results if m is not None]
    else:
        ms = []
        for s in cfg.seeds:
            m = _sim_one_seed(model, theta, T_total, cfg.burn_in, cfg.moment_set, s)
            if m is not None:
                ms.append(m)

    if not ms:
        if return_names:
            _, names = compute_moments(np.zeros(10), moment_set=cfg.moment_set, return_names=True)
            return np.full(len(names), np.nan), names
        return None

    m_bar = np.mean(np.vstack(ms), axis=0)

    if return_names:
        _, names = compute_moments(
            np.zeros(max(T_total, 10)), moment_set=cfg.moment_set, return_names=True
        )
        return m_bar, names
    return m_bar


def loss_quadratic(
    m_sim: np.ndarray,
    m_data: np.ndarray,
    W: Optional[np.ndarray] = None,
) -> float:
    diff = m_sim - m_data
    if W is None:
        return float(diff @ diff)
    return float(diff @ W @ diff)


def objective(
    theta,
    model,
    m_data: np.ndarray,
    cfg: SMMConfig,
    W: Optional[np.ndarray] = None,
    parallel_seeds: int = 0,
) -> float:
    """SMM objective function.

    Returns the quadratic loss ``(m_sim - m_data)' W (m_sim - m_data)`` averaged
    over ``cfg.seeds``. Returns ``1e18`` (rather than raising) on any numerical
    failure so the optimiser can proceed.

    Parameters
    ----------
    parallel_seeds : int
        Passed to ``simulated_moments()``.  0 = sequential, -1 = all cores.
    """
    BIG = 1e18
    try:
        m_sim = simulated_moments(model, theta, cfg, return_names=False,
                                  parallel_seeds=parallel_seeds)
        if (m_sim is None) or (not np.all(np.isfinite(m_sim))):
            return BIG

        val = loss_quadratic(m_sim, m_data, W=W)
        if not np.isfinite(val):
            return BIG

        return float(val)
    except Exception:
        return BIG
