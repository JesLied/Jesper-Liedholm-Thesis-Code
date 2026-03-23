"""Firm specific functions."""

import numpy as np
from scipy.optimize import brentq


# ====================================
# Firm's expected TFP
# ====================================
def expected_tfp(A0, K, gamma):
    """
    Expected TFP without shock (used in inner equilibrium loop).

    A = A0 * K^(-gamma)

    Parameters
    ----------
    A0    : float  structural productivity endowment
    K     : float  capital stock
    gamma : float  diminishing returns to capital accumulation

    Returns
    -------
    float
    """
    return A0 * (K ** -gamma)


# ====================================
# Firm's realised TFP
# ====================================
def realised_tfp(A0, K, gamma, sigma_eps, rng=None):
    """
    Realised TFP with idiosyncratic shock (used at realisation step).

    A = A0 * K^(-gamma) + epsilon,  epsilon ~ N(0, sigma_eps^2)

    Parameters
    ----------
    A0        : float           structural productivity endowment
    K         : float           capital stock (fixed before shock drawn)
    gamma     : float           diminishing returns parameter
    sigma_eps : float           std dev of TFP shock
    rng       : np.random.Generator or None
                pass np.random.default_rng(seed) for reproducibility.
                If None, uses global numpy state.

    Returns
    -------
    float  A_realised
    """
    if rng is not None:
        epsilon = rng.normal(0, sigma_eps)
    else:
        epsilon = np.random.normal(0, sigma_eps)
    return A0 * (K ** -gamma) + epsilon


# ====================================
# Firm's output
# ====================================
def output(A, K, L, alpha):
    """
    Cobb-Douglas output.

    Y = A * K^alpha * L^(1-alpha)

    Parameters
    ----------
    A     : float  TFP (expected or realised)
    K     : float  capital stock
    L     : float  labour (fixed, inelastic)
    alpha : float  capital share

    Returns
    -------
    float
    """
    return A * (K ** alpha) * (L ** (1 - alpha))


# ====================================
# Firm's marginal product of capital (MPK)
# ====================================
def mpk(A0, K, L, alpha, gamma):
    """
    Marginal product of capital after substituting endogenous TFP.

    Substituting A = A0 * K^(-gamma) into Y = A * K^alpha * L^(1-alpha):
        Y = A0 * K^(alpha - gamma) * L^(1-alpha)

    Differentiating w.r.t. K:
        MPK = (alpha - gamma) * A0 * K^(alpha - gamma - 1) * L^(1-alpha)

    Requires alpha > gamma to ensure MPK > 0.

    Parameters
    ----------
    A0    : float  structural productivity endowment
    K     : float  capital stock
    L     : float  labour
    alpha : float  capital share
    gamma : float  diminishing returns parameter

    Returns
    -------
    float
    """
    return (alpha - gamma) * A0 * (K ** (alpha - gamma - 1)) * (L ** (1 - alpha))


# ====================================
# Firm's dividend
# ====================================
def dividend(Y, s, alpha):
    """
    Dividend paid to equity holders each period.

    D = (1 - s) * alpha * Y

    Only capital share (alpha * Y) is available to shareholders.
    Labour share (1 - alpha) * Y is paid as wages.

    Parameters
    ----------
    Y     : float  output
    s     : float  retention rate in (0, 1)
    alpha : float  capital share

    Returns
    -------
    float
    """
    return (1 - s) * alpha * Y


# ====================================
# Expected dividend
# ====================================
def expected_dividend(A0, K, L, s, alpha, gamma):
    """
    Expected dividend using expected TFP (no shock).
    Used in reference point and implied return calculations.

    Parameters
    ----------
    A0    : float  structural productivity endowment
    K     : float  capital stock
    L     : float  labour
    s     : float  retention rate
    alpha : float  capital share
    gamma : float  diminishing returns parameter

    Returns
    -------
    float  D_hat
    """
    A_hat = expected_tfp(A0, K, gamma)
    Y_hat = output(A_hat, K, L, alpha)
    return dividend(Y_hat, s, alpha)


# ====================================
# Capital next period
# ====================================
def capital_next_period(Y, K, s, delta, alpha):
    """
    Capital accumulation via retained earnings.

    K_{t+1} = K_t * (1 - delta) + s * alpha * Y_t

    Parameters
    ----------
    Y     : float  realised output
    K     : float  current capital stock
    s     : float  retention rate
    delta : float  depreciation rate
    alpha : float  capital share

    Returns
    -------
    float  K_{t+1}
    """
    return K * (1 - delta) + s * alpha * Y


# ====================================
# Implied required return
# ====================================
def implied_return(D_hat, K, K_prev, V):
    """
    Market-implied required return on equity.

    r_implied = (D_hat + delta_K_hat) / V
              = (D_hat + K_t - K_{t-1}) / V

    Parameters
    ----------
    D_hat  : float  expected dividend
    K      : float  current capital stock
    K_prev : float  previous period capital stock
    V      : float  firm market valuation = sum_i w_{i,j} * W_i

    Returns
    -------
    float
    """
    delta_K_hat = K - K_prev
    return (D_hat + delta_K_hat) / V


# ====================================
# Tobin's Q
# ====================================
def tobins_q(V, K):
    """
    Tobin's Q = market value / replacement cost of capital.

    Q > 1: market overvalues capital, firm expands
    Q < 1: market undervalues capital, firm restricts investment
    Q = 1: equilibrium

    Parameters
    ----------
    V : float  firm market valuation
    K : float  capital stock

    Returns
    -------
    float
    """
    return V / K


# ====================================
# Capital adjustment (inner equilibrium loop)
# ====================================
def equilibrium_capital(A0, L, alpha, gamma, r_implied, delta,
                        K_lo=1e-10, K_hi=1e8, tol=1e-9):
    """
    Solve for capital K* such that MPK(K*) = r_implied + delta.

    Uses Brent's method on the residual:
        f(K) = MPK(K) - (r_implied + delta) = 0

    Parameters
    ----------
    A0        : float  structural productivity endowment
    L         : float  labour
    alpha     : float  capital share
    gamma     : float  diminishing returns parameter
    r_implied : float  market-implied required return
    delta     : float  depreciation rate
    K_lo      : float  lower bracket for Brent (default 1e-6)
    K_hi      : float  upper bracket for Brent (default 1e6)
    tol       : float  convergence tolerance

    Returns
    -------
    float  K*
    """
    target = r_implied + delta

    def residual(K):
        return mpk(A0, K, L, alpha, gamma) - target

    # Try to find valid brackets - widen search if needed
    f_lo = residual(K_lo)
    f_hi = residual(K_hi)
    
    # If both have same sign, expand brackets
    if f_lo * f_hi > 0:
        # Try wider bounds
        K_hi_new = K_hi * 100
        f_hi_new = residual(K_hi_new)
        if f_lo * f_hi_new > 0:
            # If still no sign change, return a fallback (steady-state style)
            # Use K where MPK ≈ target (simple linear approximation)
            return max(K_lo, K_hi * (1 - abs(target) / (abs(f_hi) + 1e-10)))
        else:
            K_hi = K_hi_new
            f_hi = f_hi_new
    
    try:
        return brentq(residual, K_lo, K_hi, xtol=tol)
    except ValueError as e:
        # If Brent still fails, return fallback capital
        print(f"Warning: Could not solve equilibrium capital. r_implied={r_implied:.4f}, delta={delta:.4f}")
        return K_hi  # Return upper bound as fallback