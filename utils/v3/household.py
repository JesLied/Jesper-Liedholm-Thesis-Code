"""Household specific functions."""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize


# ====================================
# Reference point (r_0)
# ====================================
def reference_point(D_hat, delta_K_hat, V_prev):
    """
    Rational expectations reference point (Koszegi & Rabin, 2006).

    Parameters
    ----------
    D_hat       : float  expected dividends = (1 - s) * alpha * Y_hat
    delta_K_hat : float  expected capital appreciation = K_t - K_{t-1}
    V_prev      : float  firm valuation previous period

    Returns
    -------
    float  r_0 = (D_hat + delta_K_hat) / V_prev
    """
    return (D_hat + delta_K_hat) / V_prev


# ====================================
# Deviation from reference point
# ====================================
def outcome_deviation(r, r_0):
    """
    x = r - r_0: gain (x > 0) or loss (x < 0) relative to reference point.

    Parameters
    ----------
    r   : float  realised or evaluated return
    r_0 : float  reference point

    Returns
    -------
    float  x
    """
    return r - r_0


# ====================================
# Prospect theory value function
# ====================================
def prospect_value(x, alpha_v=0.88, beta_v=0.88, lam=2.25):
    """
    S-shaped value function (Tversky & Kahneman, 1992).

    v(x) = x^alpha          if x >= 0  (concave gains)
    v(x) = -lambda(-x)^beta if x <  0  (convex losses, scaled by lambda)

    Parameters
    ----------
    x       : float or np.ndarray  deviation from reference point
    alpha_v : float  gain curvature (default 0.88)
    beta_v  : float  loss curvature (default 0.88)
    lam     : float  loss aversion coefficient (default 2.25)

    Returns
    -------
    float or np.ndarray
    """
    x      = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    gain   = x >= 0
    result[gain]  =  x[gain] ** alpha_v
    result[~gain] = -lam * ((-x[~gain]) ** beta_v)
    return result


# ====================================
# Probability weighting function
# ====================================
def probability_weight(p, delta=0.65):
    """
    Inverse-S probability weighting (Tversky & Kahneman, 1992).

    w(p) = p^delta / (p^delta + (1-p)^delta)^(1/delta)

    Overweights small probabilities, underweights large ones.
    Pass gamma=0.61 for gain side, delta=0.69 for loss side.

    Parameters
    ----------
    p     : float  cumulative probability in [0, 1]
    delta : float  curvature parameter

    Returns
    -------
    float  decision weight
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (p ** delta) / ((p ** delta + (1 - p) ** delta) ** (1 / delta))


# ====================================
# Friction-adjusted perceived volatility
# ====================================
def sigma_tilde(sigma, phi_0, omega):
    """
    Inflation of perceived volatility due to information frictions
    (Van Nieuwerburgh & Veldkamp, 2009).

    sigma_tilde = sigma + phi_0 / (1 + ln(1 + Omega))

    Limits:
        Omega -> 0   : sigma_tilde -> sigma + phi_0  (full ambiguity)
        Omega -> inf : sigma_tilde -> sigma           (no ambiguity)
        Domestic     : Omega = inf, so sigma_tilde = sigma

    Parameters
    ----------
    sigma : float  true return volatility (public information)
    phi_0 : float  maximum ambiguity premium
    omega : float  information level Omega_{i,j,t} >= 0

    Returns
    -------
    float
    """
    return sigma + phi_0 / (1 + np.log(1 + omega))


# ====================================
# Cumulative prospect theory value
# ====================================
def cpt_value(r_implied, r_0, sigma_tilde_val,
              alpha_v=0.88, beta_v=0.88, lam=2.25,
              gamma_plus=0.61, delta_minus=0.69):
    """
    CPT value of a single asset for a single household.

    Integrates over the perceived Normal return distribution N(r_implied, sigma_tilde^2),
    splitting at the reference point r_0 into gain and loss regions.

    Gain region (x > 0):
        integral_0^inf  w+(1 - F(x)) * v(x) * f(x) dx

    Loss region (x < 0):
        integral_-inf^0  w-(F(x)) * v(x) * f(x) dx

    where x = r - r_0, F is the CDF of x, f is the PDF.

    Parameters
    ----------
    r_implied      : float  mean of perceived return distribution
    r_0            : float  reference point
    sigma_tilde_val: float  friction-adjusted perceived std dev
    alpha_v        : float  gain curvature
    beta_v         : float  loss curvature
    lam            : float  loss aversion
    gamma_plus     : float  probability weighting curvature, gain side (0.61)
    delta_minus    : float  probability weighting curvature, loss side (0.69)

    Returns
    -------
    float  CPT value
    """
    mu_x = r_implied - r_0
    s    = sigma_tilde_val

    def gain_integrand(x):
        p = 1 - norm.cdf(x, mu_x, s)
        w = probability_weight(p, gamma_plus)
        v = prospect_value(x, alpha_v, beta_v, lam)
        return w * v * norm.pdf(x, mu_x, s)

    def loss_integrand(x):
        p = norm.cdf(x, mu_x, s)
        w = probability_weight(p, delta_minus)
        v = prospect_value(x, alpha_v, beta_v, lam)
        return w * v * norm.pdf(x, mu_x, s)

    # Truncate at 8 sigma — density negligible beyond this
    lower = mu_x - 8 * s
    upper = mu_x + 8 * s

    gain, _ = quad(gain_integrand, 0,     upper, limit=100)
    loss, _ = quad(loss_integrand, lower, 0,     limit=100)

    return gain + loss


# ====================================
# CPT portfolio optimisation
# ====================================
def cpt_portfolio(r_implied_vec, r_0_vec, sigma_tilde_vec,
                  w_prev, rho,
                  alpha_v=0.88, beta_v=0.88, lam=2.25,
                  gamma_plus=0.61, delta_minus=0.69):
    """
    Optimise CPT portfolio weights across N countries, then apply inertia.

    Maximises sum_j w_j * V_CPT_j
    subject to: sum(w) = 1, w_j >= 0

    Then applies portfolio inertia:
        w = rho * w_star + (1 - rho) * w_prev

    Parameters
    ----------
    r_implied_vec   : (N,) array  market-implied expected returns
    r_0_vec         : (N,) array  reference points per asset
    sigma_tilde_vec : (N,) array  friction-adjusted perceived vols
    w_prev          : (N,) array  previous period weights
    rho             : float       rebalancing speed in (0, 1)

    Returns
    -------
    w       : (N,) array  final weights after inertia
    w_star  : (N,) array  unconstrained CPT-optimal weights
    """
    N = len(r_implied_vec)

    # Pre-compute CPT values for each asset
    cpt_vals = np.array([
        cpt_value(r_implied_vec[j], r_0_vec[j], sigma_tilde_vec[j],
                  alpha_v, beta_v, lam, gamma_plus, delta_minus)
        for j in range(N)
    ])

    def neg_total_cpt(w):
        return -np.dot(w, cpt_vals)

    def neg_total_cpt_grad(w):
        return -cpt_vals

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * N
    w0          = w_prev.copy()  # warm start from previous weights

    result = minimize(
        neg_total_cpt,
        w0,
        jac=neg_total_cpt_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500}
    )

    w_star = result.x

    # Portfolio inertia (captures institutional sluggishness)
    w = rho * w_star + (1 - rho) * w_prev

    # Re-normalise to ensure weights sum to 1 after inertia
    w = w / w.sum()

    return w, w_star


# ====================================
# Household wealth update
# ====================================
def update_wealth(w_prev, D_vec, K_vec, K_vec_prev, V_vec, W_prev):
    """
    Update household wealth using portfolio rate of return.

    r_j     = (D_j + K_j - K_{j-1}) / V_j
    W_new_i = W_i * (1 + sum_j w_ij * r_j)

    Parameters
    ----------
    w_prev    : (N,) array  portfolio weights previous period
    D_vec     : (N,) array  realised dividends
    K_vec     : (N,) array  capital stocks this period
    K_vec_prev: (N,) array  capital stocks previous period
    V_vec     : (N,) array  firm valuations this period
    W_prev    : float       household wealth previous period

    Returns
    -------
    float  W_t
    """
    r_vec = (D_vec + K_vec - K_vec_prev) / V_vec
    return W_prev * (1 + float(np.dot(w_prev, r_vec)))


# ====================================
# Bilateral CPIS / FDI positions
# ====================================
def bilateral_positions(w, W):
    """
    Compute bilateral portfolio positions (CPIS equivalent).

    CPIS_{i,j,t} = w_{i,j,t} * W_{i,t}

    Parameters
    ----------
    w : (N,) array  portfolio weights of household i across N countries
    W : float       household i wealth

    Returns
    -------
    (N,) array  bilateral positions
    """
    return w * W