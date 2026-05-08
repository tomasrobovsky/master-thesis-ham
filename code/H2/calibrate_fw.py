"""
H2 driver: calibrate the FW model on one bootstrap replication of one market.

Designed for independent parallel execution on Metacentrum:

    python calibrate_fw.py --market sp500 --boot-id 42

Each invocation:
  1. Loads the original rescaled returns for the market.
  2. Computes the diagonal bootstrap weighting matrix W from those returns.
  3. Generates the bootstrap pseudo-series for the given boot_id (deterministic
     block indices via master_seed + boot_id, so the same boot_id produces the
     SAME indices on every market — synchronous bootstrap).
  4. Calibrates FW on the bootstrap series (DE + Nelder-Mead).
  5. Saves the result as fw_{market}_b{boot_id:03d}.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from moments import compute_moments
from smm.objective import SMMConfig, simulated_moments
from smm.optimize import run_de, refine_nelder_mead
from smm.weighting import compute_weighting_matrix
from models.fw_dca_hpm import FrankeWesterhoffDCAHPM
from block_bootstrap import bootstrap_single


DATA_FILES = {
    "sp500": "sp500_returns_common_rescaled.csv",
    "btc":   "btc_returns_common_rescaled.csv",
    "eth":   "eth_returns_common_rescaled.csv",
}


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate FW on one synchronous-bootstrap replication for H2."
    )
    parser.add_argument("--model", default="fw", choices=["fw"],
                        help="kept for backwards compatibility; only FW is supported")
    parser.add_argument("--market", required=True, choices=["sp500", "btc", "eth"])
    parser.add_argument("--boot-id", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=60)
    parser.add_argument("--master-seed", type=int, default=20260403)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--de-maxiter", type=int, default=100)
    parser.add_argument("--de-popsize", type=int, default=30)
    parser.add_argument("--nm-maxiter", type=int, default=500)
    parser.add_argument("--parallel-seeds", type=int, default=0)
    parser.add_argument("--w-bootstrap", type=int, default=5000)
    parser.add_argument("--w-block-size", type=int, default=250)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    ps = args.parallel_seeds
    n_cores = os.cpu_count() if ps == -1 else ps

    # Load original data
    data_path = os.path.join(data_dir, DATA_FILES[args.market])
    r_orig = pd.read_csv(data_path)["r"].to_numpy(dtype=float)
    T = len(r_orig)

    print(f"=== H2 calibrate_fw.py ===")
    print(f"Model: FW, Market: {args.market}")
    print(f"Boot ID: {args.boot_id}, Block size: {args.block_size}")
    print(f"T: {T}, Seeds: {args.seeds}, Parallel: {n_cores} cores")
    print(f"DE: maxiter={args.de_maxiter}, popsize={args.de_popsize}")

    model = FrankeWesterhoffDCAHPM(pstar=0.0, n0=0.5, beta=1.0)
    burn_in = 200

    cfg = SMMConfig(
        T=T,
        seeds=tuple(range(1, args.seeds + 1)),
        moment_set="FWCHL17",
        burn_in=burn_in,
    )

    # W from the ORIGINAL data — same across all bootstrap replications
    print(f"\nComputing W from original data...")
    t0 = time.time()
    W = compute_weighting_matrix(
        r_orig, moment_set=cfg.moment_set,
        n_bootstrap=args.w_bootstrap, block_size=args.w_block_size,
        seed=42,
    )
    print(f"W computed in {time.time() - t0:.1f}s")

    # Generate bootstrap pseudo-data (deterministic in boot_id and master_seed)
    r_boot = bootstrap_single(
        r_orig,
        boot_id=args.boot_id,
        block_size=args.block_size,
        master_seed=args.master_seed,
    )

    # Calibrate
    m_data = compute_moments(r_boot, moment_set=cfg.moment_set)

    print(f"\n--- DE ---")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_de = run_de(
            model, m_data=m_data, cfg=cfg, W=W,
            maxiter=args.de_maxiter, popsize=args.de_popsize,
            seed=42, polish=True, workers=-1,
            parallel_seeds=0,
        )
    time_de = time.time() - t0
    print(f"DE: {time_de:.1f}s, loss={res_de.loss_hat:.4f}")

    print(f"\n--- NM ---")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_nm = refine_nelder_mead(
            model, theta0=res_de.theta_hat,
            m_data=m_data, cfg=cfg, W=W,
            maxiter=args.nm_maxiter,
            parallel_seeds=0,
        )
    time_nm = time.time() - t0
    print(f"NM: {time_nm:.1f}s, loss={res_nm.loss_hat:.4f}")

    m_sim = simulated_moments(model, res_nm.theta_hat, cfg, parallel_seeds=ps)

    # Save
    result = {
        "model": type(model).__name__,
        "model_short": "fw",
        "market": args.market,
        "theta_names": model.theta_names,
        "theta_hat": res_nm.theta_hat.tolist(),
        "loss_hat": float(res_nm.loss_hat),
        "boot_id": args.boot_id,
        "block_size": args.block_size,
        "master_seed": args.master_seed,
        "cfg": {
            "T": cfg.T,
            "seeds": list(cfg.seeds),
            "moment_set": cfg.moment_set,
            "burn_in": cfg.burn_in,
        },
        "optimization": {
            "de_maxiter": args.de_maxiter,
            "de_popsize": args.de_popsize,
            "nm_maxiter": args.nm_maxiter,
            "parallel_seeds": ps,
        },
        "moments_data": m_data.tolist(),
        "moments_sim": m_sim.tolist(),
        "time_de_s": round(time_de, 1),
        "time_nm_s": round(time_nm, 1),
    }

    fname = f"fw_{args.market}_b{args.boot_id:03d}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {fpath}")
    print(f"Total: {time_de + time_nm:.1f}s ({(time_de + time_nm)/3600:.2f}h)")


if __name__ == "__main__":
    main()
