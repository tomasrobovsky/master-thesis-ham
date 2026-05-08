"""
Wald test for H3: pre-ETF vs post-ETF parameter equality (BTC, FW model).

Usage:
    python wald_test_h3.py \
        --results-dir H3/results \
        --out H3/wald_fw_btc_preEtf_vs_postEtf.json

Method
------
H0: theta_pre == theta_post

Point estimate:
    d = theta_hat_pre - theta_hat_post    (from fw_btc_pre_b000.json / fw_btc_post_b000.json)

Variance:
    The pre-ETF and post-ETF samples are non-overlapping in time, so the
    estimators are INDEPENDENT.  Therefore:

        V_delta = Var(theta_hat_pre) + Var(theta_hat_post)

    Each variance is estimated from an independent block bootstrap
    (boot_id = 1..B, master_seed differs per period).

Joint Wald statistic:
    W = d' V_delta^{-1} d  ~  chi^2(K)   under H0

Individual parameter tests:
    W_i = d_i^2 / (V_delta)_ii  ~  chi^2(1)   under H0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import stats


# ── helpers ─────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_period_thetas(
    results_dir: str, period: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load original + bootstrap theta_hat arrays for one sub-period.

    Returns
    -------
    theta_orig : ndarray, shape (K,)
    thetas_boot : ndarray, shape (B, K)
    boot_ids : list[int]    (sorted, 1..B)
    """
    pattern = re.compile(rf"^fw_btc_{re.escape(period)}_b(\d+)\.json$")
    entries_boot = []
    theta_orig = None
    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if not m:
            continue
        boot_id = int(m.group(1))
        path = os.path.join(results_dir, fname)
        d = load_json(path)
        theta = np.array(d["theta_hat"])
        if boot_id == 0:
            theta_orig = theta
        else:
            entries_boot.append((boot_id, theta))

    if theta_orig is None:
        raise FileNotFoundError(
            f"Missing original calibration fw_btc_{period}_b000.json in {results_dir}"
        )
    if not entries_boot:
        raise FileNotFoundError(
            f"No bootstrap results for period={period} in {results_dir}"
        )
    entries_boot.sort(key=lambda x: x[0])
    boot_ids = [e[0] for e in entries_boot]
    thetas = np.array([e[1] for e in entries_boot])
    return theta_orig, thetas, boot_ids


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Wald test ────────────────────────────────────────────────────────

def wald_test_h3(
    theta_pre: np.ndarray,         # (K,) original, pre period
    theta_post: np.ndarray,        # (K,) original, post period
    boot_pre: np.ndarray,          # (B_pre,  K)
    boot_post: np.ndarray,         # (B_post, K)
    theta_names: list[str],
    alpha: float = 0.05,
) -> dict:
    """
    Compute joint and individual Wald tests for H3 structural break.

    V_delta = Cov(boot_pre) + Cov(boot_post), exploiting the independence
    of pre- and post-ETF sub-samples.
    """
    K = len(theta_names)
    assert boot_pre.shape[1] == K and boot_post.shape[1] == K

    # Point estimate of difference
    d = theta_pre - theta_post                      # (K,)

    # Per-period bootstrap covariances (sample covariance of theta_hat^(b))
    V_pre  = np.cov(boot_pre.T,  ddof=1)             # (K, K)
    V_post = np.cov(boot_post.T, ddof=1)             # (K, K)
    V_delta = V_pre + V_post                          # independence

    # --- Joint Wald ---
    try:
        V_inv = np.linalg.inv(V_delta)
        W_joint = float(d @ V_inv @ d)
        singular = False
    except np.linalg.LinAlgError:
        W_joint = np.nan
        singular = True

    df_joint = K
    p_joint  = (float(1 - stats.chi2.cdf(W_joint, df_joint))
                if not np.isnan(W_joint) else np.nan)

    # --- Individual Wald (chi^2(1)) ---
    diag_V = np.diag(V_delta)
    W_ind  = d**2 / diag_V
    p_ind  = 1 - stats.chi2.cdf(W_ind, df=1)
    se     = np.sqrt(diag_V)                          # SE of difference
    se_pre  = np.sqrt(np.diag(V_pre))
    se_post = np.sqrt(np.diag(V_post))

    return {
        "d": d,
        "V_pre": V_pre,
        "V_post": V_post,
        "V_delta": V_delta,
        "se": se,
        "se_pre": se_pre,
        "se_post": se_post,
        "W_joint": W_joint,
        "df_joint": df_joint,
        "p_joint": p_joint,
        "singular": singular,
        "W_ind": W_ind,
        "p_ind": p_ind,
        "theta_names": theta_names,
        "B_pre": boot_pre.shape[0],
        "B_post": boot_post.shape[0],
        "K": K,
        "alpha": alpha,
    }


def print_results(
    res: dict,
    theta_pre: np.ndarray,
    theta_post: np.ndarray,
) -> None:
    names   = res["theta_names"]
    d       = res["d"]
    se      = res["se"]
    se_pre  = res["se_pre"]
    se_post = res["se_post"]
    W_ind   = res["W_ind"]
    p_ind   = res["p_ind"]
    alpha   = res["alpha"]
    B_pre   = res["B_pre"]
    B_post  = res["B_post"]
    K       = res["K"]

    print_section("Setup (H3: pre-ETF vs post-ETF, BTC, FW)")
    print(f"  Parameters:   K = {K}")
    print(f"  Bootstrap:    B_pre = {B_pre}, B_post = {B_post}")
    print(f"  Significance: alpha = {alpha}")

    print_section("Point estimates and bootstrap SE")
    print(f"  {'param':10} {'pre':>12} {'post':>12} {'SE_pre':>10} {'SE_post':>10} "
          f"{'diff':>12} {'SE(diff)':>10} {'z':>8}")
    print(f"  {'-'*95}")
    for i, name in enumerate(names):
        z = d[i] / se[i] if se[i] > 0 else np.nan
        print(f"  {name:10} {theta_pre[i]:>12.5f} {theta_post[i]:>12.5f} "
              f"{se_pre[i]:>10.5f} {se_post[i]:>10.5f} "
              f"{d[i]:>12.5f} {se[i]:>10.5f} {z:>8.2f}")

    print_section("Individual Wald tests  [W_i ~ chi^2(1)]")
    print(f"  {'param':10} {'W_i':>10} {'p-value':>10} {'sig':>6}")
    print(f"  {'-'*40}")
    for i, name in enumerate(names):
        sig = "***" if p_ind[i] < 0.01 else ("**" if p_ind[i] < 0.05 else
              ("*" if p_ind[i] < 0.1 else ""))
        print(f"  {name:10} {W_ind[i]:>10.3f} {p_ind[i]:>10.4f} {sig:>6}")

    print_section(f"Joint Wald test  [W ~ chi^2({K})]")
    if res["singular"]:
        print("  WARNING: V_delta is singular — cannot compute joint test.")
    else:
        print(f"  W statistic: {res['W_joint']:.4f}")
        print(f"  df:          {res['df_joint']}")
        print(f"  p-value:     {res['p_joint']:.6f}")
        sig = ("***" if res["p_joint"] < 0.01
               else ("**" if res["p_joint"] < 0.05
               else ("*" if res["p_joint"] < 0.1 else "not significant")))
        print(f"  Result:      {sig}")

    print(f"\n  Significance codes: *** p<0.01  ** p<0.05  * p<0.10")


def save_results(res: dict, theta_pre: np.ndarray, theta_post: np.ndarray,
                 out_path: str) -> None:
    output = {
        "model": "fw",
        "market": "btc",
        "test": "H3_structural_break_preETF_vs_postETF",
        "B_pre": res["B_pre"],
        "B_post": res["B_post"],
        "K": res["K"],
        "theta_names": res["theta_names"],
        "theta_hat_pre": theta_pre.tolist(),
        "theta_hat_post": theta_post.tolist(),
        "d": res["d"].tolist(),
        "se_pre": res["se_pre"].tolist(),
        "se_post": res["se_post"].tolist(),
        "se_d": res["se"].tolist(),
        "V_pre": res["V_pre"].tolist(),
        "V_post": res["V_post"].tolist(),
        "V_delta": res["V_delta"].tolist(),
        "W_joint": res["W_joint"],
        "df_joint": res["df_joint"],
        "p_joint": res["p_joint"],
        "singular": res["singular"],
        "W_individual": res["W_ind"].tolist(),
        "p_individual": res["p_ind"].tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wald test for H3: pre-ETF vs post-ETF parameter equality."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing fw_btc_{pre,post}_b{000..500}.json files")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    theta_pre,  boot_pre,  ids_pre  = load_period_thetas(args.results_dir, "pre")
    theta_post, boot_post, ids_post = load_period_thetas(args.results_dir, "post")
    print(f"  pre : B = {len(ids_pre)}, theta_orig shape = {theta_pre.shape}")
    print(f"  post: B = {len(ids_post)}, theta_orig shape = {theta_post.shape}")

    # Load names from original pre file
    orig_pre = load_json(os.path.join(args.results_dir, "fw_btc_pre_b000.json"))
    theta_names = orig_pre["theta_names"]

    res = wald_test_h3(theta_pre, theta_post, boot_pre, boot_post,
                       theta_names, alpha=args.alpha)
    print_results(res, theta_pre, theta_post)

    if args.out:
        save_results(res, theta_pre, theta_post, args.out)


if __name__ == "__main__":
    main()
