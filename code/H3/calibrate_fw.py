"""
H3 driver: calibrate FW on one BTC sub-period (pre-ETF or post-ETF).

Designed for independent parallel execution on Metacentrum:

    python calibrate_fw.py --period pre  --boot-id 0    # original (headline)
    python calibrate_fw.py --period post --boot-id 0    # original
    python calibrate_fw.py --period pre  --boot-id 42   # bootstrap rep 42
    python calibrate_fw.py --period post --boot-id 42   # bootstrap rep 42

Each invocation:
  1. Loads sub-period data (btc_pre_h3_rescaled.csv or btc_post_h3_rescaled.csv).
  2. Computes the period-specific diagonal weighting matrix W from that
     sub-period's ORIGINAL data (re-used across all bootstrap replications of
     the same period).
  3. If boot_id == 0: calibrates on the ORIGINAL sub-period returns.
     If boot_id >= 1: generates a block-bootstrap pseudo-series and calibrates
     on it.
  4. Saves one JSON file: fw_btc_{period}_{orig_run|b{boot_id}}.json

Bootstrap RNG uses a period-specific master seed so pre- and post-ETF
bootstraps are INDEPENDENT — unlike the H2 synchronous bootstrap, which shares
block indices across markets. This is appropriate because the two H3
sub-samples are non-overlapping in time, so the cross-covariance term in the
Wald variance vanishes:

    V_delta = Cov(bootstrap_pre) + Cov(bootstrap_post)
    W = d' V_delta^{-1} d  ~  chi^2(K)   under H0: theta_pre == theta_post
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
from block_bootstrap import bootstrap_single


DATA_FILES = {
    "pre":  "btc_pre_h3_rescaled.csv",
    "post": "btc_post_h3_rescaled.csv",
}

# Period-specific master seeds → independent bootstrap RNG streams.
# Distinct bases guarantee that boot_id=b produces different block indices
# for pre vs post, even when T_pre == T_post.
MASTER_SEEDS = {
    "pre":  20260501,
    "post": 20260601,
}


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate FW on one H3 sub-period for one bootstrap replication."
    )
    parser.add_argument("--period", required=True, choices=["pre", "post"])
    parser.add_argument("--boot-id", type=int, required=True,
                        help="0 = original data (point estimate); >=1 = bootstrap replication")
    parser.add_argument("--orig-run-id", type=int, default=1,
                        help="When boot_id=0, selects seed block: 1 -> seeds 1..N, 2 -> seeds N+1..2N, etc. Ignored for boot_id>=1.")
    parser.add_argument("--block-size", type=int, default=30)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--de-maxiter", type=int, default=100)
    parser.add_argument("--de-popsize", type=int, default=30)
    parser.add_argument("--nm-maxiter", type=int, default=500)
    parser.add_argument("--parallel-seeds", type=int, default=0)
    parser.add_argument("--moment-set", default="FWCHL16")
    parser.add_argument("--w-bootstrap", type=int, default=5000)
    parser.add_argument("--w-block-size", type=int, default=125)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "H3", "results")
    os.makedirs(out_dir, exist_ok=True)

    ps = args.parallel_seeds
    n_cores = os.cpu_count() if ps == -1 else ps

    # Load sub-period data (original returns for this period)
    data_path = os.path.join(data_dir, DATA_FILES[args.period])
    r_orig = pd.read_csv(data_path)["r"].to_numpy(dtype=float)
    T = len(r_orig)

    print(f"=== calibrate_fw.py ===")
    print(f"Model: FW, Market: BTC, Period: {args.period}")
    print(f"Boot ID: {args.boot_id} ({'ORIGINAL' if args.boot_id == 0 else 'bootstrap'})")
    print(f"T: {T}, Block size: {args.block_size}")
    print(f"Seeds: {args.seeds}, Parallel: {n_cores} cores")
    print(f"DE: maxiter={args.de_maxiter}, popsize={args.de_popsize}")
    print(f"Moment set: {args.moment_set}")

    # Model and config
    from models.fw_dca_hpm import FrankeWesterhoffDCAHPM
    model = FrankeWesterhoffDCAHPM(pstar=0.0, n0=0.5, beta=1.0)
    burn_in = 200

    # Seed range:
    #   boot_id == 0  → use orig_run_id to pick block of args.seeds seeds
    #                   run 1: 1..N, run 2: N+1..2N, ..., run 10: 9N+1..10N
    #   boot_id >= 1  → always use seeds 1..N
    if args.boot_id == 0:
        seed_start = (args.orig_run_id - 1) * args.seeds + 1
    else:
        seed_start = 1
    seed_end = seed_start + args.seeds - 1

    cfg = SMMConfig(
        T=T,
        seeds=tuple(range(seed_start, seed_end + 1)),
        moment_set=args.moment_set,
        burn_in=burn_in,
    )
    print(f"Seed range: {seed_start}..{seed_end}")

    # W from ORIGINAL sub-period data (same for original + all bootstraps of this period)
    print(f"\nComputing W from original {args.period} data...")
    t0 = time.time()
    W = compute_weighting_matrix(
        r_orig, moment_set=cfg.moment_set,
        n_bootstrap=args.w_bootstrap, block_size=args.w_block_size,
        seed=42,
    )
    print(f"W computed in {time.time() - t0:.1f}s")

    # Choose return series for calibration
    if args.boot_id == 0:
        r_fit = r_orig
        print(f"\nCalibrating on ORIGINAL {args.period} data")
    else:
        master_seed = MASTER_SEEDS[args.period]
        r_fit = bootstrap_single(
            r_orig,
            boot_id=args.boot_id,
            block_size=args.block_size,
            master_seed=master_seed,
        )
        print(f"\nCalibrating on bootstrap replication {args.boot_id} "
              f"(master_seed={master_seed})")

    # Calibrate
    m_data = compute_moments(r_fit, moment_set=cfg.moment_set)

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
        "market": "btc",
        "period": args.period,
        "theta_names": model.theta_names,
        "theta_hat": res_nm.theta_hat.tolist(),
        "loss_hat": float(res_nm.loss_hat),
        "boot_id": args.boot_id,
        "is_original": args.boot_id == 0,
        "orig_run_id": args.orig_run_id if args.boot_id == 0 else None,
        "block_size": args.block_size,
        "master_seed": MASTER_SEEDS[args.period],
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

    if args.boot_id == 0:
        fname = f"fw_btc_{args.period}_orig_run{args.orig_run_id:02d}.json"
    else:
        fname = f"fw_btc_{args.period}_b{args.boot_id:03d}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {fpath}")
    print(f"Total: {time_de + time_nm:.1f}s ({(time_de + time_nm)/3600:.2f}h)")


if __name__ == "__main__":
    main()
