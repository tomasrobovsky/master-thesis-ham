import numpy as np


class FarmerJoshi2002:
    """
    Farmer & Joshi (2002) heterogeneous-agent model with two trader types:

      - Value investors: threshold strategy with heterogeneous (T_v, tau_v, c_v, nu_i).
        Mispricing m_i = p_t - (vbar_t + nu_i).
        Overvalued (m > T_v) → short:  X = -c_v * m
        Undervalued (m < -T_v) → long: X = -c_v * m  (positive since m < 0)
        Exit short when m < tau_v; exit long when m > -tau_v.

      - Trend followers: threshold strategy with heterogeneous (T_t, tau_t, c_t, theta_i).
        Trend signal = p_t - p_{t-theta_i}.
        Enter long/short when |signal| > T_t; position upon entry: X = c_t * signal.
        Exit when the signal crosses tau_t.

    Price dynamics:
      p_{t+1} = p_t + (1/lambda) * omega_t + sigma_xi * Z_t

    lambda = 1 (not separately identifiable from the capital scaling a).

    This implementation uses the calibration variant adopted in the thesis:

      - Capital scaling tied across populations:  a_v = a_t = a
      - Fundamental noise tied to pricing noise:  sigma_eta = sigma_xi
        (vbar follows a random walk with std sigma_xi)
      - Fractional re-parametrisation of the exit thresholds: each exit range
        [tau_lo, tau_max] is encoded as (tau_lo, tau_frac) with
            tau_max = tau_lo + tau_frac * (T_min - margin - tau_lo),
        which guarantees tau < T_min for every agent and removes the dead
        region tau_max >= T_min from the search space.
      - Entry-threshold minima T_v_min, T_t_min are anchored at a fixed value
        (T_min = 0.025 in the thesis); the corresponding maxima are bounded
        by (T_min, 10 * T_min).

    Estimated parameters (9):
      [a, T_v_max, T_t_max, nu_max, sigma_xi,
       tau_v_lo, tau_v_frac, tau_t_lo, tau_t_frac]
    """

    theta_names = [
        "a",            # capital scaling (a_v = a_t = a)
        "T_v_max",      # value-investor entry threshold (max)
        "T_t_max",      # trend-follower entry threshold (max)
        "nu_max",       # perceived-value offset range (uniform on [-nu_max, +nu_max])
        "sigma_xi",     # exogenous price noise std (= sigma_eta)
        "tau_v_lo",     # value-investor exit lower bound
        "tau_v_frac",   # value-investor exit fractional position in [0, 1]
        "tau_t_lo",     # trend-follower exit lower bound
        "tau_t_frac",   # trend-follower exit fractional position in [0, 1]
    ]

    def __init__(
        self,
        T_v_min: float = 0.025,
        T_t_min: float = 0.025,
        n_vi: int = 200,
        n_tf: int = 200,
        p0: float = 0.0,
        v0: float = 0.0,
        theta_max: int = 100,
        lambd: float = 1.0,
    ):
        self.T_v_min = float(T_v_min)
        self.T_t_min = float(T_t_min)
        self.n_vi = int(n_vi)
        self.n_tf = int(n_tf)
        self.p0 = float(p0)
        self.v0 = float(v0)
        self.theta_max = int(theta_max)
        self.lambd = float(lambd)

        # Margins guarantee tau < T for every agent
        self._margin_v = 0.05 * self.T_v_min
        self._margin_t = 0.05 * self.T_t_min

        self.bounds = [
            (1e-6, 0.5),                                            # a
            (self.T_v_min, 10.0 * self.T_v_min),                    # T_v_max
            (self.T_t_min, 10.0 * self.T_t_min),                    # T_t_max
            (0.0, 10.0),                                            # nu_max
            (0.001, 0.10),                                          # sigma_xi
            (-self.T_v_min, self.T_v_min - self._margin_v),         # tau_v_lo
            (0.0, 1.0),                                             # tau_v_frac
            (-self.T_t_min, self.T_t_min - self._margin_t),         # tau_t_lo
            (0.0, 1.0),                                             # tau_t_frac
        ]

    @staticmethod
    def _ensure_order(min_val: float, max_val: float, eps: float = 1e-12) -> tuple[float, float]:
        mn = float(min_val)
        mx = float(max_val)
        if not np.isfinite(mn):
            mn = 0.0
        if not np.isfinite(mx):
            mx = mn + 1.0
        if mx < mn + eps:
            mx = mn + eps
        return mn, mx

    @staticmethod
    def _draw_uniform(rng, n: int, lo: float, hi: float) -> np.ndarray:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1e-12
        return rng.uniform(lo, hi, size=int(n))

    @staticmethod
    def _enforce_tau_band(tau: np.ndarray, T: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        T = np.asarray(T, dtype=float)
        margin = 1e-6 + 1e-3 * np.maximum(1.0, np.abs(T))
        tau = np.where(tau >= T, T - margin, tau)
        tau = np.where(tau <= -T, -T + margin, tau)
        return tau

    def simulate(self, theta, T: int, seed: int):
        theta = np.asarray(theta, dtype=float)
        if theta.size != 9:
            raise ValueError(f"theta must have length 9 (got {theta.size})")

        (a, T_v_max, T_t_max, nu_max, sigma_xi,
         tau_v_lo, tau_v_frac, tau_t_lo, tau_t_frac) = map(float, theta)

        # Decode fractional tau parameterisation
        tau_v_frac = min(max(tau_v_frac, 0.0), 1.0)
        tau_t_frac = min(max(tau_t_frac, 0.0), 1.0)
        tau_v_min = tau_v_lo
        tau_v_max = tau_v_lo + tau_v_frac * (self.T_v_min - self._margin_v - tau_v_lo)
        tau_t_min = tau_t_lo
        tau_t_max = tau_t_lo + tau_t_frac * (self.T_t_min - self._margin_t - tau_t_lo)

        # sigma_eta tied to sigma_xi (random-walk fundamental)
        sigma_eta = sigma_xi

        a = max(a, 0.0)
        nu_max = max(nu_max, 0.0)
        sigma_xi = max(sigma_xi, 1e-12)
        sigma_eta = max(sigma_eta, 1e-12)

        T_v_min, T_v_max = self._ensure_order(max(self.T_v_min, 1e-12), max(T_v_max, 1e-12))
        T_t_min, T_t_max = self._ensure_order(max(self.T_t_min, 1e-12), max(T_t_max, 1e-12))
        tau_v_min, tau_v_max = self._ensure_order(tau_v_min, tau_v_max)
        tau_t_min, tau_t_max = self._ensure_order(tau_t_min, tau_t_max)

        rng = np.random.default_rng(int(seed))
        TT = int(T)

        p = np.zeros(TT + 1, dtype=float)
        vbar = np.zeros(TT + 1, dtype=float)
        r = np.zeros(TT, dtype=float)
        p[0] = self.p0
        vbar[0] = self.v0

        # --- VALUE INVESTORS ---
        T_v = self._draw_uniform(rng, self.n_vi, T_v_min, T_v_max)
        tau_v = self._draw_uniform(rng, self.n_vi, tau_v_min, tau_v_max)
        tau_v = self._enforce_tau_band(tau_v, T_v)
        c_v = a * (T_v - tau_v)
        nu_i = self._draw_uniform(rng, self.n_vi, -nu_max, +nu_max)
        x_v = np.zeros(self.n_vi, dtype=float)

        # --- TREND FOLLOWERS ---
        theta_i = rng.integers(1, self.theta_max + 1, size=self.n_tf, endpoint=True)
        T_t = self._draw_uniform(rng, self.n_tf, T_t_min, T_t_max)
        tau_t = self._draw_uniform(rng, self.n_tf, tau_t_min, tau_t_max)
        tau_t = self._enforce_tau_band(tau_t, T_t)
        c_t = a * (T_t - tau_t)
        x_t = np.zeros(self.n_tf, dtype=float)

        for t in range(TT):
            p_t = p[t]

            # Fundamental random walk
            vbar[t + 1] = vbar[t] + rng.normal(0.0, sigma_eta)

            # === VALUE INVESTORS ===
            m = p_t - (vbar[t] + nu_i)
            x_v_next = x_v.copy()

            # Flat agents: check for entry
            flat = (x_v == 0.0)
            enter_short = flat & (m > T_v)    # overvalued → short
            enter_long = flat & (m < -T_v)    # undervalued → long
            x_v_next[enter_short] = -c_v[enter_short] * m[enter_short]
            x_v_next[enter_long] = -c_v[enter_long] * m[enter_long]

            # Holding short (x < 0): exit when m drops below tau
            short_pos = (x_v < 0)
            exit_short = short_pos & (m < tau_v)
            x_v_next[exit_short] = 0.0

            # Holding long (x > 0): exit when m rises above -tau
            long_pos = (x_v > 0)
            exit_long = long_pos & (m > -tau_v)
            x_v_next[exit_long] = 0.0

            # Hold position (no exit)
            hold_short = short_pos & ~exit_short
            hold_long = long_pos & ~exit_long
            x_v_next[hold_short] = x_v[hold_short]
            x_v_next[hold_long] = x_v[hold_long]

            omega_v = np.sum(x_v_next - x_v)
            x_v = x_v_next

            # === TREND FOLLOWERS ===
            idx = np.maximum(t - theta_i, 0)
            trend_signal = p_t - p[idx]
            x_t_next = x_t.copy()

            flat_t = (x_t == 0.0)
            enter_long_t = flat_t & (trend_signal > T_t)
            enter_short_t = flat_t & (trend_signal < -T_t)
            x_t_next[enter_long_t] = c_t[enter_long_t] * trend_signal[enter_long_t]
            x_t_next[enter_short_t] = c_t[enter_short_t] * trend_signal[enter_short_t]

            long_t = (x_t > 0)
            exit_long_t = long_t & (trend_signal < tau_t)
            x_t_next[exit_long_t] = 0.0

            short_t = (x_t < 0)
            exit_short_t = short_t & (trend_signal > -tau_t)
            x_t_next[exit_short_t] = 0.0

            hold_long_t = long_t & ~exit_long_t
            hold_short_t = short_t & ~exit_short_t
            x_t_next[hold_long_t] = x_t[hold_long_t]
            x_t_next[hold_short_t] = x_t[hold_short_t]

            omega_t_val = np.sum(x_t_next - x_t)
            x_t = x_t_next

            # Net order flow → price update
            omega = omega_v + omega_t_val
            p[t + 1] = p_t + (omega / self.lambd) + rng.normal(0.0, sigma_xi)
            r[t] = p[t + 1] - p_t

            if not np.isfinite(p[t + 1]) or abs(p[t + 1]) > 1e6:
                return np.full(TT, np.nan)

        return r[:TT].astype(float, copy=False)
