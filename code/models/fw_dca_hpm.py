import numpy as np


class FrankeWesterhoffDCAHPM:
    """
    Fixed parameters (not estimated):
      beta = 1.0   (intensity of choice)

    Estimated parameters (8):
      phi, chi, sigma_f, sigma_c, alpha_h, alpha_p, alpha_m, mu

    Switching index:
      A_t = alpha_h * (n_f - n_c) + alpha_p + alpha_m * (p* - P_t)^2

    DCA switching:
      n_f,t = 1 / (1 + exp(-beta * A_{t-1})),  n_c,t = 1 - n_f,t

    Price dynamics:
      P_{t+1} = P_t + mu * ( n_f,t * Df_t + n_c,t * Dc_t )

    Demands:
      Df_t = phi * (p* - P_t) + sigma_f * eps_f,t
      Dc_t = chi * (P_t - P_{t-1}) + sigma_c * eps_c,t

    Returns:
      r_t = P_{t+1} - P_t   (log-returns if P is log-price)
    """

    theta_names = [
        "phi",       # fundamentalist demand strength
        "chi",       # chartist demand strength
        "sigma_f",   # fundamentalist demand noise std
        "sigma_c",   # chartist demand noise std
        "alpha_h",   # herding weight 
        "alpha_p",   # predisposition bias 
        "alpha_m",   # misalignment weight 
        "mu",        # market maker adjustment rate
    ]

    bounds = [
        (0.0, 0.5),    # phi
        (0.0, 4.0),    # chi
        (0.1, 1.0),    # sigma_f
        (2.0, 25.0),   # sigma_c
        (1.0, 6.0),    # alpha_h
        (-1.0, 1.0),   # alpha_p
        (1.0, 35.0),   # alpha_m
        (0.001, 0.1),  # mu
    ]

    def __init__(self, pstar: float = 0.0, n0: float = 0.5,
                 beta: float = 1.0):
        self.pstar = float(pstar)
        self.n0 = float(n0)
        self.beta = float(beta)

    @staticmethod
    def _logistic(z: float) -> float:
        """Numericky stabilní logistická funkce."""
        if z >= 0:
            ez = np.exp(-z)
            return float(1.0 / (1.0 + ez))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))

    def simulate(self, theta, T: int, seed: int):
        phi, chi, sigma_f, sigma_c, alpha_h, alpha_p, alpha_m, mu = map(float, theta)
        beta = self.beta

        rng = np.random.default_rng(int(seed))

        # log-price (nebo log-deviation) path
        P = np.zeros(T + 1, dtype=float)
        r = np.zeros(T, dtype=float)

        # fractions
        n_f = float(self.n0)

        # for chartists' trend term at t=0
        P_prev = P[0]

        # initialize attractiveness A_{-1} effectively via A0
        # (compute A_t based on current P_t and current n_f)
        for t in range(T):
            # current chartist fraction
            n_c = 1.0 - n_f

            # HPM switching index at time t (herding + predisposition + misalignment)
            # (n_f - n_c) = 2*n_f - 1
            A_t = alpha_h * (2.0 * n_f - 1.0) + alpha_p + alpha_m * (self.pstar - P[t]) ** 2

            # DCA: next-period fraction uses A_t (i.e., n_{t+1} depends on A_t)
            n_f_next = self._logistic(beta * A_t)

            # demands (with their own noises)
            eps_f = rng.normal(0.0, sigma_f)
            eps_c = rng.normal(0.0, sigma_c)

            Df = phi * (self.pstar - P[t]) + eps_f
            Dc = chi * (P[t] - P_prev) + eps_c

            # price impact
            P[t + 1] = P[t] + mu * (n_f * Df + n_c * Dc)

            # return
            r[t] = P[t + 1] - P[t]

            # advance
            P_prev = P[t]
            n_f = n_f_next

            if not np.isfinite(P[t + 1]):
                return np.full(T, np.nan)

        return r
