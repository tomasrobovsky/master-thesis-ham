"""
Wald test for H2: cross-market parameter equality (BTC vs SP500).

Usage:
    python wald_test.py \
        --orig-a results/fw_btc_common_diag_20260403.json \
        --orig-b results/fw_sp500_common_diag_20260403.json \
        --boot-dir H2/results \
        --model fw \
        --market-a btc \
        --market-b sp500

Method
------
H0: theta_BTC == theta_SP500

Point estimate:
    d = theta_hat_BTC - theta_hat_SP500   (from original data calibrations)

Variance of d via synchronous block bootstrap:
    d^b = theta_hat_BTC^b - theta_hat_SP500^b   for b = 1..B
    V_delta = sample covariance of {d^b}

The synchronous bootstrap preserves the cross-covariance between markets,
so V_delta automatically accounts for it:
    V_delta = Var(theta_BTC) + Var(theta_SP500) - 2*Cov(theta_BTC, theta_SP500)

Joint Wald statistic:
    W = d' V_delta^{-1} d  ~  chi^2(K)   under H0

Individual parameter tests:
    W_i = d_i^2 / (V_delta)_ii  ~  chi^2(1)   under H0

Reference: Ghysels & Hall (1990), Andrews (1993), Hall & Sen (1999).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── helpers ─────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_bootstrap_thetas(
    boot_dir: str, model: str, market: str
) -> tuple[np.ndarray, list[int]]:
    """
    Load all bootstrap theta_hat arrays for given model and market.

    Returns
    -------
    thetas : ndarray, shape (B, K)
    boot_ids : list of int, sorted
    """
    pattern = re.compile(rf"^{re.escape(model)}_{re.escape(market)}_b(\d+)\.json$")
    entries = []
    for fname in os.listdir(boot_dir):
        m = pattern.match(fname)
        if m:
            boot_id = int(m.group(1))
            path = os.path.join(boot_dir, fname)
            d = load_json(path)
            entries.append((boot_id, np.array(d["theta_hat"])))

    if not entries:
        raise FileNotFoundError(
            f"No bootstrap files found for model={model}, market={market} in {boot_dir}"
        )

    entries.sort(key=lambda x: x[0])
    boot_ids = [e[0] for e in entries]
    thetas = np.array([e[1] for e in entries])  # (B, K)
    return thetas, boot_ids


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Wald test ────────────────────────────────────────────────────────

def wald_test(
    theta_a: np.ndarray,         # (K,) original estimate, market A
    theta_b: np.ndarray,         # (K,) original estimate, market B
    boot_a: np.ndarray,          # (B, K) bootstrap estimates, market A
    boot_b: np.ndarray,          # (B, K) bootstrap estimates, market B
    theta_names: list[str],
    alpha: float = 0.05,
) -> dict:
    """
    Compute joint and individual Wald tests.

    Returns dict with full results.
    """
    K = len(theta_names)
    B = boot_a.shape[0]
    assert boot_a.shape == boot_b.shape, "Bootstrap arrays must have same shape"
    assert boot_a.shape[1] == K

    # Point estimate of difference
    d = theta_a - theta_b                        # (K,)

    # Bootstrap differences (captures cross-covariance automatically)
    d_boot = boot_a - boot_b                     # (B, K)
    d_bar  = d_boot.mean(axis=0)                 # (K,)

    # Sample covariance matrix of d^b
    centered = d_boot - d_bar                    # (B, K)
    V_delta  = (centered.T @ centered) / (B - 1) # (K, K)

    # --- Joint Wald ---
    try:
        V_inv = np.linalg.inv(V_delta)
        W_joint = float(d @ V_inv @ d)
        singular = False
    except np.linalg.LinAlgError:
        W_joint = np.nan
        singular = True

    df_joint = K
    p_joint  = float(1 - stats.chi2.cdf(W_joint, df_joint)) if not np.isnan(W_joint) else np.nan

    # --- Individual Wald (chi^2(1)) ---
    diag_V = np.diag(V_delta)
    W_ind  = d**2 / diag_V                        # (K,)
    p_ind  = 1 - stats.chi2.cdf(W_ind, df=1)      # (K,)
    se     = np.sqrt(diag_V)                       # bootstrap SE of d_i

    # --- Bootstrap correlation between markets ---
    corr_matrix = np.corrcoef(boot_a.T, boot_b.T)  # (2K, 2K)
    # cross-market correlations: top-right KxK block
    cross_corr = corr_matrix[:K, K:]

    return {
        "d": d,
        "V_delta": V_delta,
        "se": se,
        "W_joint": W_joint,
        "df_joint": df_joint,
        "p_joint": p_joint,
        "singular": singular,
        "W_ind": W_ind,
        "p_ind": p_ind,
        "theta_names": theta_names,
        "B": B,
        "K": K,
        "alpha": alpha,
        "cross_corr": cross_corr,
    }


def print_results(
    res: dict,
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    market_a: str,
    market_b: str,
) -> None:
    names  = res["theta_names"]
    d      = res["d"]
    se     = res["se"]
    W_ind  = res["W_ind"]
    p_ind  = res["p_ind"]
    alpha  = res["alpha"]
    B      = res["B"]
    K      = res["K"]

    print_section("Setup")
    print(f"  Markets:     {market_a} (A)  vs  {market_b} (B)")
    print(f"  Parameters:  K = {K}")
    print(f"  Bootstrap B: {B}")
    print(f"  Significance: alpha = {alpha}")

    print_section("Point estimates and bootstrap SE")
    print(f"  {'param':10} {market_a:>12} {market_b:>12} {'diff (A-B)':>12} {'SE(diff)':>10} {'z':>8}")
    print(f"  {'-'*66}")
    for i, name in enumerate(names):
        z = d[i] / se[i]
        print(f"  {name:10} {theta_a[i]:>12.5f} {theta_b[i]:>12.5f} {d[i]:>12.5f} {se[i]:>10.5f} {z:>8.2f}")

    print_section("Individual Wald tests  [W_i ~ chi^2(1)]")
    print(f"  {'param':10} {'W_i':>10} {'p-value':>10} {'sig':>6}")
    print(f"  {'-'*40}")
    for i, name in enumerate(names):
        sig = "***" if p_ind[i] < 0.01 else ("**" if p_ind[i] < 0.05 else ("*" if p_ind[i] < 0.1 else ""))
        print(f"  {name:10} {W_ind[i]:>10.3f} {p_ind[i]:>10.4f} {sig:>6}")

    print_section(f"Joint Wald test  [W ~ chi^2({K})]")
    if res["singular"]:
        print("  WARNING: V_delta is singular — cannot compute joint test.")
        print("  Consider using pseudo-inverse (--use-pinv) or checking for")
        print("  collinear parameters.")
    else:
        print(f"  W statistic: {res['W_joint']:.4f}")
        print(f"  df:          {res['df_joint']}")
        print(f"  p-value:     {res['p_joint']:.6f}")
        sig = ("***" if res["p_joint"] < 0.01
               else ("**" if res["p_joint"] < 0.05
               else ("*" if res["p_joint"] < 0.1 else "not significant")))
        print(f"  Result:      {sig}")

    print(f"\n  Significance codes: *** p<0.01  ** p<0.05  * p<0.10")

    # Cross-market correlation summary (diagonal = same-param cross-market corr)
    cross_diag = np.diag(res["cross_corr"])
    print_section("Bootstrap cross-market correlations (per parameter)")
    print(f"  (rho between theta_hat_{market_a}^b and theta_hat_{market_b}^b)")
    print(f"  {'param':10} {'rho':>8}")
    for i, name in enumerate(names):
        print(f"  {name:10} {cross_diag[i]:>8.3f}")


def save_results(res: dict, theta_a: np.ndarray, theta_b: np.ndarray,
                 market_a: str, market_b: str, model: str, out_path: str) -> None:
    output = {
        "model": model,
        "market_a": market_a,
        "market_b": market_b,
        "B": res["B"],
        "K": res["K"],
        "theta_names": res["theta_names"],
        "theta_hat_a": theta_a.tolist(),
        "theta_hat_b": theta_b.tolist(),
        "d": res["d"].tolist(),
        "se_d": res["se"].tolist(),
        "V_delta": res["V_delta"].tolist(),
        "W_joint": res["W_joint"],
        "df_joint": res["df_joint"],
        "p_joint": res["p_joint"],
        "singular": res["singular"],
        "W_individual": res["W_ind"].tolist(),
        "p_individual": res["p_ind"].tolist(),
        "cross_corr_diagonal": np.diag(res["cross_corr"]).tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wald test for H2: cross-market parameter equality."
    )
    parser.add_argument("--orig-a", required=True,
                        help="JSON with original calibration for market A (e.g. BTC)")
    parser.add_argument("--orig-b", required=True,
                        help="JSON with original calibration for market B (e.g. SP500)")
    parser.add_argument("--boot-dir", required=True,
                        help="Directory containing bootstrap JSON files")
    parser.add_argument("--model", required=True, help="Model short name (e.g. fw)")
    parser.add_argument("--market-a", required=True, help="Market A name (e.g. btc)")
    parser.add_argument("--market-b", required=True, help="Market B name (e.g. sp500)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default 0.05)")
    parser.add_argument("--out", default=None,
                        help="Save JSON results to this path (optional)")
    args = parser.parse_args()

    # Load original estimates
    orig_a = load_json(args.orig_a)
    orig_b = load_json(args.orig_b)
    theta_a = np.array(orig_a["theta_hat"])
    theta_b = np.array(orig_b["theta_hat"])
    theta_names = orig_a["theta_names"]

    assert orig_a["theta_names"] == orig_b["theta_names"], \
        "theta_names mismatch between orig-a and orig-b"

    # Load bootstrap thetas
    print(f"Loading bootstrap results from: {args.boot_dir}")
    boot_a, ids_a = load_bootstrap_thetas(args.boot_dir, args.model, args.market_a)
    boot_b, ids_b = load_bootstrap_thetas(args.boot_dir, args.model, args.market_b)

    if ids_a != ids_b:
        # Keep only common boot_ids
        common = sorted(set(ids_a) & set(ids_b))
        print(f"WARNING: boot_ids differ. Using {len(common)} common replications.")
        idx_a = [ids_a.index(i) for i in common]
        idx_b = [ids_b.index(i) for i in common]
        boot_a = boot_a[idx_a]
        boot_b = boot_b[idx_b]
    else:
        print(f"Bootstrap replications: B = {len(ids_a)}")

    # Run test
    res = wald_test(theta_a, theta_b, boot_a, boot_b, theta_names, alpha=args.alpha)

    # Print
    print_results(res, theta_a, theta_b, args.market_a, args.market_b)

    # Optionally save
    if args.out:
        save_results(res, theta_a, theta_b, args.market_a, args.market_b,
                     args.model, args.out)


if __name__ == "__main__":
    main()
