from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

from smm.objective import SMMConfig, objective


@dataclass(frozen=True)
class OptimizeResult:
    theta_hat: np.ndarray
    loss_hat: float


def run_de(model, m_data: np.ndarray, cfg: SMMConfig, W: Optional[np.ndarray] = None,
           maxiter: int = 50, popsize: int = 15, seed: int = 123,
           polish: bool = True, workers: int = 1,
           parallel_seeds: int = 0) -> OptimizeResult:
    bounds = getattr(model, "bounds", None)
    if bounds is None:
        raise ValueError("Model must define `bounds` for DE.")

    # Use functools.partial instead of a closure so the callable is
    # picklable — required when workers != 1 (multiprocessing).
    f = partial(objective, model=model, m_data=m_data, cfg=cfg, W=W,
                parallel_seeds=parallel_seeds)

    res = differential_evolution(
        f,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=polish,
        updating="deferred",
        workers=workers,
    )

    return OptimizeResult(theta_hat=res.x, loss_hat=float(res.fun))


def refine_nelder_mead(model, theta0: np.ndarray, m_data: np.ndarray, cfg: SMMConfig,
                       W: Optional[np.ndarray] = None,
                       maxiter: int = 300,
                       parallel_seeds: int = 0) -> OptimizeResult:
    bounds = getattr(model, "bounds", None)

    def penalty(th):
        if bounds is None:
            return 0.0
        pen = 0.0
        for x, (lo, hi) in zip(th, bounds):
            if x < lo:
                pen += (lo - x) ** 2
            elif x > hi:
                pen += (x - hi) ** 2
        return 1e6 * pen

    def f(th):
        return objective(th, model=model, m_data=m_data, cfg=cfg, W=W,
                         parallel_seeds=parallel_seeds) + penalty(th)

    res = minimize(
        f,
        x0=np.asarray(theta0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6},
    )

    return OptimizeResult(theta_hat=res.x, loss_hat=float(res.fun))
