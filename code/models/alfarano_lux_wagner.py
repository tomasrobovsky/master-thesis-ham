import numpy as np


class AlfaranoLuxWagner2008:
    """
    ALW (2008) herding model with Langevin sentiment dynamics. The
    sentiment-to-return scaling is fixed at gamma = 1, matching the
    original ALW (2008) demand-scale normalisation N_f T_f / (N T_c) = 1.

    Sentiment dynamics (Langevin, Eq. 12 in ALW 2008) with finite-population
    correction term 4*a/N in the diffusion coefficient:
      x_{t+1} = (1 - 2*a*dt)*x_t + sqrt(dt*(2*b*(1-x_t^2) + 4*a/N)) * Z_t

    Returns:
      r_t = sigf * Z2_t + (x_t - x_{t-1})

    where the first term is the fundamental price change and the second
    is the sentiment-driven component.

    Estimated parameters (3): [a, b, sigf]
    """

    theta_names = ["a", "b", "sigf"]

    bounds = [
        (1e-5, 1.0),    # a    — idiosyncratic switching intensity
        (1e-5, 5.0),    # b    — herding intensity
        (1e-4, 5.0),    # sigf — fundamental noise std
    ]

    def __init__(self, x0: float = 0.0, dt: float = 1.0, nx: int = 100):
        self.x0 = float(x0)
        self.dt = float(dt)
        self.nx = int(nx)
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.nx <= 0:
            raise ValueError("nx must be > 0")

    @staticmethod
    def _clamp(x: float) -> float:
        if x > 1.0:
            return 1.0
        if x < -1.0:
            return -1.0
        return float(x)

    def simulate(self, theta, T: int, seed: int):
        a, b, sigf = map(float, theta)

        if a <= 0 or b <= 0 or sigf <= 0:
            return np.full(int(T), np.nan)

        rng = np.random.default_rng(int(seed))
        T = int(T)
        dt = self.dt
        nx = self.nx

        r = np.empty(T, dtype=float)

        shocks_sentiment = rng.normal(0.0, 1.0, size=T)
        shocks_fundamental = rng.normal(0.0, 1.0, size=T)

        x = self._clamp(self.x0)

        for t in range(T):
            x_prev = x

            # Langevin update with finite-population correction.
            # Overshoots beyond [-1, 1] are allowed (no clamp after update);
            # outside the band the dynamics are pure deterministic drift.
            if -1.0 < x < 1.0:
                drift = (1.0 - 2.0 * a * dt) * x
                diff_var = dt * (2.0 * b * (1.0 - x * x) + 4.0 * a / nx)
                diff_var = max(0.0, diff_var)
                x = drift + np.sqrt(diff_var) * shocks_sentiment[t]
            else:
                x = (1.0 - 2.0 * a * dt) * x

            # Return = fundamental shock + sentiment change (gamma = 1)
            r[t] = sigf * shocks_fundamental[t] + (x - x_prev)

            if not np.isfinite(r[t]):
                return np.full(T, np.nan)

        return r
