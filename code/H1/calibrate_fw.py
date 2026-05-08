"""
Calibrate FW (DCA-HPM) model on standardised log-returns — H1 grid.

Each run uses a DIFFERENT block of 100 seeds:
  run 1: seeds    1 –  100
  run 2: seeds  101 –  200
  ...
  run 10: seeds 901 – 1000

DE_SEED = 42 for all runs.

Usage:
    python calibrate_fw.py --market sp500 --run-id 1
"""

from __future__ import annotations

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

DATA_FILES = {
    "sp500":    "sp500_returns_common_rescaled.csv",
    "btc":      "btc_returns_common_rescaled.csv",
    "eth":      "eth_returns_common_rescaled.csv",
    "btc_7day": "btc_returns_common_7day_rescaled.csv",
}

N_SEEDS_PER_RUN = 100
DE_SEED = 42


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate FW on rescaled returns (common variance target)")
    parser.add_argument("--market", required=True, choices=["sp500", "btc", "eth", "btc_7day"])
    parser.add_argument("--run-id", required=True, type=int, help="1..10")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out-dir",  default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--parallel-seeds", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Configuration ─────────────────────────────────────────────────
    MOMENT_SET = "FWCHL17"
    BURN_IN = 200
    DE_MAXITER = 100
    DE_POPSIZE = 30
    NM_MAXITER = 500
    W_BOOTSTRAP = 5000
    W_BLOCK_SIZE = 250

    seed_start = (args.run_id - 1) * N_SEEDS_PER_RUN + 1
    seed_end = args.run_id * N_SEEDS_PER_RUN
    SEEDS = tuple(range(seed_start, seed_end + 1))

    # ── Load standardised data ────────────────────────────────────────
    data_path = os.path.join(args.data_dir, DATA_FILES[args.market])
    df = pd.read_csv(data_path)
    r = df["r"].to_numpy(dtype=float)
    T = len(r)
    dates = df["date"].tolist()

    print(f"Market: {args.market}  Run: {args.run_id}")
    print(f"Data: {data_path}, T={T}, {dates[0]} to {dates[-1]}")
    print(f"Standardised: mean={r.mean():.4f}, std={r.std(ddof=1):.4f}")
    print(f"Seeds: {seed_start}-{seed_end} ({len(SEEDS)} seeds)")

    # ── W matrix ──────────────────────────────────────────────────────
    print(f"\nComputing W...")
    t0 = time.time()
    W = compute_weighting_matrix(
        r, moment_set=MOMENT_SET,
        n_bootstrap=W_BOOTSTRAP, block_size=W_BLOCK_SIZE,
        seed=42,
    )
    cond_W = float(np.linalg.cond(W))
    print(f"W done in {time.time()-t0:.1f}s, cond={cond_W:.2e}")

    # ── Moments + model ───────────────────────────────────────────────
    m_data, moment_names = compute_moments(r, moment_set=MOMENT_SET, return_names=True)
    model = FrankeWesterhoffDCAHPM(pstar=0.0, n0=0.5, beta=1.0)

    cfg = SMMConfig(T=T, seeds=SEEDS, moment_set=MOMENT_SET, burn_in=BURN_IN)

    ps = args.parallel_seeds
    n_cores = os.cpu_count() if ps == -1 else ps
    print(f"\nModel: {type(model).__name__}")
    print(f"  theta_names = {model.theta_names}")
    print(f"  bounds      = {model.bounds}")
    print(f"Seeds: {len(SEEDS)}, parallel_seeds: {ps}")
    print(f"DE: maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, workers=-1 (DE-level parallelism)")

    # ── DE ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStage 1: DE\n{'='*60}")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_de = run_de(
            model, m_data=m_data, cfg=cfg, W=W,
            maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            seed=DE_SEED, polish=True, workers=-1,
            parallel_seeds=ps,
        )
    time_de = time.time() - t0
    print(f"DE: {time_de:.1f}s ({time_de/3600:.2f}h), loss={res_de.loss_hat:.6f}")
    print(f"theta: {dict(zip(model.theta_names, res_de.theta_hat))}")

    # ── NM ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStage 2: NM (maxiter={NM_MAXITER})\n{'='*60}")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_nm = refine_nelder_mead(
            model, theta0=res_de.theta_hat,
            m_data=m_data, cfg=cfg, W=W,
            maxiter=NM_MAXITER, parallel_seeds=ps,
        )
    time_nm = time.time() - t0
    print(f"NM: {time_nm:.1f}s ({time_nm/3600:.2f}h), loss={res_nm.loss_hat:.6f}")
    print(f"theta: {dict(zip(model.theta_names, res_nm.theta_hat))}")

    # ── Moments comparison ────────────────────────────────────────────
    m_sim = simulated_moments(model, res_nm.theta_hat, cfg, parallel_seeds=ps)
    print(f"\n{'Moment':12s}  {'Data':>12s}  {'Sim':>12s}  {'Diff%':>8s}")
    for name, md, ms in zip(moment_names, m_data, m_sim):
        pct = 100 * (ms - md) / abs(md) if abs(md) > 1e-15 else 0
        print(f"{name:12s}  {md:12.6f}  {ms:12.6f}  {pct:+7.1f}%")

    # ── J-test (descriptive only — diagonal W) ────────────────────────
    from scipy.stats import chi2
    D, K = len(m_data), len(model.theta_names)
    df_j = D - K
    J_stat = float(res_nm.loss_hat)
    p_value = 1.0 - chi2.cdf(J_stat, df_j)
    print(f"\nJ-test (indicative, diagonal W): J={J_stat:.4f}, df={df_j}, p={p_value:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    from datetime import date
    result = {
        "model": type(model).__name__,
        "theta_names": model.theta_names,
        "theta_hat":   res_nm.theta_hat.tolist(),
        "loss_hat":    float(res_nm.loss_hat),
        "method": "DE+polish+NM",
        "cfg": {"T": cfg.T, "seeds": list(cfg.seeds),
                "moment_set": cfg.moment_set, "burn_in": cfg.burn_in},
        "optimization": {"de_maxiter": DE_MAXITER, "de_popsize": DE_POPSIZE,
                         "de_seed": DE_SEED, "de_polish": True,
                         "de_workers": -1,
                         "nm_maxiter": NM_MAXITER, "parallel_seeds": ps},
        "weighting_matrix": "diagonal_bootstrap",
        "bootstrap_config": {"n_bootstrap": W_BOOTSTRAP, "block_size": W_BLOCK_SIZE},
        "condition_number_W": cond_W,
        "j_test": {"J_stat": J_stat, "df": df_j, "p_value": p_value},
        "data_file": os.path.abspath(data_path),
        "data_rescaled": True,
        "data_rescaling": "common variance target (geometric mean of market sigmas)",
        "date_range": [dates[0], dates[-1]],
        "moment_names": list(moment_names),
        "moments_data": m_data.tolist(),
        "moments_sim":  m_sim.tolist(),
        "time_de_s": round(time_de, 2),
        "time_nm_s": round(time_nm, 2),
    }

    fname = (f"fw_{args.market}_resc_run{args.run_id:03d}_"
             f"de{DE_SEED}_{date.today().strftime('%Y%m%d')}.json")
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {fpath}")
    print(f"Total: {time_de+time_nm:.1f}s ({(time_de+time_nm)/3600:.2f}h)")


if __name__ == "__main__":
    main()
