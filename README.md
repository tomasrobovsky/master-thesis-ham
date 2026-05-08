# master-thesis-ham

Companion repository for the Master's thesis **"Heterogeneous Agent Modelling in Traditional Finance vs. Crypto Markets: Does One Model Rule Them All?"** .

The thesis estimates four heterogeneous agent models — Brock & Hommes (1998), Farmer & Joshi (2002), Alfarano *et al.* (2008), and Franke & Westerhoff (2012) — on the S&P 500, Bitcoin, and Ethereum by simulated method of moments (SMM).

## Repository layout

```
master-thesis-ham/
├── data/                    daily log-returns (rescaled, ready to use)
└── code/
    ├── moments.py           FWCHL17 / FWCHL16 moment sets (smoothed ACFs, Hill tail index)
    ├── block_bootstrap.py   overlapping block-bootstrap primitives (used by H2 and H3)
    ├── smm/                 SMM building blocks
    │   ├── objective.py     SMMConfig, simulated_moments, quadratic loss, objective
    │   ├── optimize.py      DE (global) + Nelder–Mead (local) two-stage estimator
    │   └── weighting.py     diagonal block-bootstrap weighting matrix
    ├── models/              the four HAMs (each implements theta_names, bounds, simulate)
    │   ├── bh.py
    │   ├── fw_dca_hpm.py
    │   ├── alfarano_lux_wagner.py
    │   └── farmer_joshi.py
    ├── H1/                  model comparison (FW on the 3 markets)
    │   ├── calibrate_fw.py  per-(market, run_id) calibration driver
    │   ├── job.sh           PBS job script
    │   └── submit.sh        submits 30 jobs (3 markets × 10 seed blocks)
    ├── H2/                  cross-market Wald test (BTC vs S&P 500)
    │   ├── calibrate_fw.py  per-(market, boot_id) calibration on a synchronous bootstrap replication
    │   ├── wald_test.py     aggregation across the 500 bootstrap pairs
    │   ├── job.sh
    │   └── submit.sh        submits 1,000 jobs (500 boots × 2 markets, interleaved)
    └── H3/                  structural-break Wald test (BTC pre/post-ETF)
        ├── calibrate_fw.py  per-(period, boot_id) calibration on an independent bootstrap replication
        ├── wald_test.py     aggregation across the 500 bootstrap pairs
        ├── job.sh
        └── submit.sh        submits 1,020 jobs (20 originals + 1,000 boots, interleaved)
```

The block bootstrap module (`code/block_bootstrap.py`) is a thin layer over `numpy` that provides `bootstrap_single` and `bootstrap_pair`. The synchronous-vs-independent distinction is purely a calling convention based on how the caller chooses `master_seed`:

- **H2 (synchronous)** uses the *same* `master_seed` across markets → identical block indices for BTC and S&P 500 within a given `boot_id` → cross-market co-movement preserved.
- **H3 (independent)** uses *different* `master_seed`s per period → independent indices for the pre- and post-ETF replications, which is correct because the two sub-samples are non-overlapping in time.

## Data

All series are daily log-returns retrieved from Yahoo Finance and rescaled to a common unconditional standard deviation `σ_target ≈ 0.0306` via a pure multiplicative transformation `r_t^resc = r_t · (σ_target / σ̂_market)`, so all scale-invariant moments (skewness, kurtosis, Hill tail index, autocorrelations) are preserved exactly. Per-market rescaling factors are reported in Table 4.2 of the thesis.

### Common-sample series (B-day alignment, used in H1 and H2)

Cryptocurrency series are aligned to the S&P 500 trading calendar so all three series share identical observation dates. Sample period: **2017-11-10 to 2025-12-31**, *T* = 2,045 daily log-returns per asset.

| File | Asset | Rescaling factor |
|---|---|---|
| `data/sp500_returns_common_rescaled.csv` | S&P 500 | 2.487 |
| `data/btc_returns_common_rescaled.csv` | Bitcoin | 0.724 |
| `data/eth_returns_common_rescaled.csv` | Ethereum | 0.555 |

### 7-day calendar BTC (Appendix B robustness)

Bitcoin returns without business-day aggregation, used in the data-alignment robustness check. *T* = 2,973.

| File | Asset |
|---|---|
| `data/btc_returns_common_7day_rescaled.csv` | Bitcoin (7-day) |

### Pre/post-ETF sub-samples (used in H3)

Each cryptocurrency series is split around its spot ETF approval date. Sub-samples are trimmed to equal length (*T*<sub>pre</sub> = *T*<sub>post</sub>) and rescaled by the same per-market factor as the full series, so the volatility-level shift across the break is preserved.

| File | Asset | Period | *T* |
|---|---|---|---|
| `data/btc_pre_h3_rescaled.csv` | Bitcoin pre-ETF | 2022-01-19 to 2024-01-09 | 496 |
| `data/btc_post_h3_rescaled.csv` | Bitcoin post-ETF | 2024-01-10 to 2025-12-31 | 496 |
| `data/eth_pre_h3_rescaled.csv` | Ethereum pre-ETF | 2023-02-09 to 2024-07-22 | 363 |
| `data/eth_post_h3_rescaled.csv` | Ethereum post-ETF | 2024-07-23 to 2025-12-31 | 363 |

Cutoff dates: **2024-01-10** (SEC approval of the first eleven spot Bitcoin ETFs) and **2024-07-23** (SEC approval of spot Ethereum ETFs).

### File format

Two columns: `date` (ISO `YYYY-MM-DD`) and `r` (rescaled daily log-return).

## SMM estimation framework

The four model classes share a uniform interface:
- `theta_names`: parameter names in the order expected by `simulate`
- `bounds`: per-parameter `(lo, hi)` tuples for Differential Evolution
- `simulate(theta, T, seed) -> np.ndarray`: length-`T` log-return path; returns `np.full(T, np.nan)` on numerical failure

`smm.objective` averages simulated moments over `cfg.seeds` and computes the quadratic loss `(m_sim - m_data)' W (m_sim - m_data)`. Burn-in is handled externally: the model simulates `T + burn_in` and the first `burn_in` observations are discarded before computing moments.

`smm.optimize` runs Differential Evolution (global, `polish=True`) followed by Nelder–Mead (local, with soft bound penalties).

`smm.weighting.compute_weighting_matrix` returns `W = diag(1/σ̂²)` from an overlapping block bootstrap (default `B = 5,000`, `block_size = 250`). Long-memory moments (ACFs at lags ≥ 10) use a second independent block-bootstrap sample, following Kukačka & Žila (2023).

## Running on Metacentrum

The repository is laid out so that the entire tree can be uploaded as a single self-contained job package. Each hypothesis (H1, H2, H3) gets its own working directory on storage, the same code is reused across all three.

### One-time setup

```bash
ssh skirit.metacentrum.cz
module add python/3.10
python -m venv ~/venvs/diplomka
source ~/venvs/diplomka/bin/activate
pip install numpy scipy pandas
```

### Deploying a hypothesis

For each of `h1_fw`, `h2_fw`, `h3_fw`:

```bash
BASE=/storage/brno2/home/$USER/diplomka
mkdir -p $BASE/h1_fw          # or h2_fw, h3_fw
rsync -av master-thesis-ham/code master-thesis-ham/data \
      $USER@skirit.metacentrum.cz:$BASE/h1_fw/
```

After upload, `$BASE/h1_fw/` should contain `code/` and `data/` side by side.

### Submitting

```bash
ssh skirit.metacentrum.cz
cd $BASE/h1_fw                  # or h2_fw, h3_fw
bash code/H1/submit.sh          # or code/H2/submit.sh, code/H3/submit.sh
```

| Hypothesis | Submitter | Total jobs | Per-job resources |
|---|---|---|---|
| H1 — model comparison (FW on 3 markets) | `code/H1/submit.sh` | 30  (3 markets × 10 seed blocks) | 8 cpus, 4 h |
| H2 — cross-market Wald (BTC vs S&P 500) | `code/H2/submit.sh` | 1,000  (500 bootstraps × 2 markets) | 6 cpus, 4 h |
| H3 — structural break (BTC pre/post-ETF) | `code/H3/submit.sh` | 1,020  (20 originals + 1,000 bootstraps) | 6 cpus, 4 h |

All jobs use 100 simulation seeds, `DE maxiter=100, popsize=30`, `NM maxiter=500`. H1 uses FWCHL17; H2 uses FWCHL17; H3 uses FWCHL16 (the lag-50 ACF cannot be reliably preserved by the 30-day block bootstrap on a 496-day sub-sample).

### After the queue drains

JSON results land in `$BASE/<hypothesis>/results/`. Local aggregation:

- **H1**: pick the lowest-loss run per market from the ten files; results enter Table 6.7 of the thesis.
- **H2**: `python code/H2/wald_test.py --results-dir results/ --out wald_h2.json` builds Σ̂_C, Σ̂_E, Σ̂_CE, V̂_Δ and reports the joint and per-parameter Wald statistics.
- **H3**: `python code/H3/wald_test.py --results-dir results/ --out wald_h3.json` does the same with V̂_Δ = V̂_pre + V̂_post (sub-samples are non-overlapping in time so the cross-covariance term vanishes).

## Local replication

The code runs locally just as well — the Metacentrum scripts only handle resource allocation. From the project root:

```bash
python code/H1/calibrate_fw.py --market sp500 --run-id 1 \
    --data-dir data --out-dir results
```

A single H1 run takes roughly 30 minutes on 8 cores; full H2 / H3 grids are designed for cluster-scale parallelism.

## License

MIT (see `LICENSE`).
