import numpy as np


class BrockHommes2:
    """
    Two-type Brock & Hommes (1998) model: a fundamentalist (g=0, b=0) and a
    chartist (g_c, b_c). Loop order each period: fitness -> fractions -> pricing.

    Pricing:
      R * x_t = sum_h n_{h,t} * (g_h * x_{t-1} + b_h) + eps_t

    Fitness — profit at t-1 evaluated with the forecast formed at t-2:
      z_{h,t-1} = d * (g_h * x_{t-2} + b_h - R * x_{t-1})
      pi_{h,t}  = (x_t - R * x_{t-1}) * z_{h,t-1}
      U_{h,t}   = pi_{h,t} + eta * U_{h,t-1}

    Switching (multinomial logit):
      n_{h,t} = exp(beta * U_{h,t-1}) / sum_j exp(beta * U_{j,t-1})

    Parameters for SMM: [beta, g_c, b_c, eta, sigma_eps]
    """

    theta_names = [
        "beta",       # intensity of choice
        "g_c",        # chartist trend extrapolation
        "b_c",        # chartist bias
        "eta",        # fitness memory
        "sigma_eps",  # pricing noise std
    ]

    bounds = [
        (0.0, 150.0),   # beta
        (-4.0, 4.0),    # g_c
        (-1.0, 1.0),    # b_c
        (0.0, 1.0),     # eta
        (1e-6, 0.5),    # sigma_eps
    ]

    def __init__(self, R: float = 1.0001, d: float = 1.0, n0: float = 0.5):
        self.R = float(R)
        self.d = float(d)
        self.n0 = float(n0)

    @staticmethod
    def _softmax2(a: float, b: float) -> tuple[float, float]:
        m = max(a, b)
        ea = np.exp(a - m)
        eb = np.exp(b - m)
        s = ea + eb
        return float(ea / s), float(eb / s)

    def simulate(self, theta, T: int, seed: int):
        beta, g_c, b_c, eta, sigma_eps = map(float, theta)
        R = self.R
        d = self.d

        rng = np.random.default_rng(int(seed))

        # x[0..T]: price deviations; r[0..T-1]: log-returns
        x = np.zeros(T + 1, dtype=float)
        r = np.zeros(T, dtype=float)

        noise = rng.normal(0.0, sigma_eps, size=T + 1)

        n_f = self.n0
        n_c = 1.0 - n_f
        U_f = 0.0
        U_c = 0.0

        x[0] = rng.uniform(-0.01, 0.01)

        for t in range(T):
            # 1) Fitness: profit at t-1 from the forecast formed at t-2
            if t >= 2:
                R_ex = x[t] - R * x[t - 1]
                z_f = d * (0.0 - R * x[t - 1])
                z_c = d * (g_c * x[t - 2] + b_c - R * x[t - 1])
                U_f = R_ex * z_f + eta * U_f
                U_c = R_ex * z_c + eta * U_c

            # 2) Update fractions
            n_f, n_c = self._softmax2(beta * U_f, beta * U_c)

            # 3) Pricing: x[t+1] = (1/R) * sum n_h * (g_h*x[t] + b_h) + noise
            f_bar = n_c * (g_c * x[t] + b_c)   # fundamentalist contributes 0
            x[t + 1] = (1.0 / R) * f_bar + noise[t + 1]

            r[t] = x[t + 1] - x[t]

            if not np.isfinite(x[t + 1]):
                return np.full(T, np.nan)

        return r
