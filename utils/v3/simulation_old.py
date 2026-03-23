"""Simulation and equilibrium functions."""

import numpy as np

from firm import (expected_dividend, implied_return, tobins_q,
                  mpk, equilibrium_capital,
                  realised_tfp, output, dividend, capital_next_period)
from household import (reference_point, sigma_tilde,
                       cpt_portfolio, update_wealth, bilateral_positions)


# ====================================
# Market clearing
# ====================================
def market_clearing(W_vec: np.ndarray, w_mat: np.ndarray) -> np.ndarray:
    """
    Compute firm valuations from household wealth and portfolio weights.

    V_j = sum_i w_{i,j} * W_i

    Each household i allocates fraction w_{i,j} of its wealth W_i
    to country j. Summing across all households gives the total
    market value of firm j.

    Parameters
    ----------
    W_vec : (N,)   household wealth per country
    w_mat : (N, N) portfolio weights, w_mat[i, j] = weight of household i in firm j

    Returns
    -------
    (N,) firm valuations V_j
    """
    return w_mat.T @ W_vec


# ====================================
# Inner equilibrium loop
# ====================================
def inner_equilibrium(K_vec: np.ndarray,
                      K_vec_prev: np.ndarray,
                      W_vec: np.ndarray,
                      w_mat: np.ndarray,
                      w_mat_prev: np.ndarray,
                      omega_mat: np.ndarray,
                      A0_vec: np.ndarray,
                      L_vec: np.ndarray,
                      s_vec: np.ndarray,
                      sigma_vec: np.ndarray,
                      rho: float,
                      phi_0: float,
                      alpha: float,
                      gamma: float,
                      delta: float,
                      alpha_v: float = 0.88,
                      beta_v: float = 0.88,
                      lam: float = 2.25,
                      gamma_plus: float = 0.61,
                      delta_minus: float = 0.69,
                      max_iter: int = 100,
                      tol: float = 1e-6) -> tuple:
    """
    Iterate between CPT portfolio choice and firm capital adjustment
    until MPK = r_implied + delta for all countries.

    Per iteration:
        1. Market clearing      -> V_j = sum_i w_{i,j} * W_i
        2. Implied returns      -> r_j = (D_hat_j + delta_K_j) / V_j
        3. Reference points     -> r_0_{i,j} (same as r_j; all households
                                   see same firm fundamentals)
        4. Friction volatilities-> sigma_tilde_{i,j} per household-firm pair
        5. CPT optimisation     -> w*_i per household, then apply inertia
        6. Capital adjustment   -> K*_j via Brent on MPK(K) = r_j + delta
        7. Convergence check    -> loss = sum_j (MPK_j - r_j - delta)^2

    Parameters
    ----------
    K_vec      : (N,)   capital stocks entering this period
    K_vec_prev : (N,)   capital stocks from previous period (for delta_K)
    W_vec      : (N,)   household wealth
    w_mat      : (N,N)  portfolio weights warm start
    w_mat_prev : (N,N)  portfolio weights previous period (for inertia)
    omega_mat  : (N,N)  information levels Omega_{i,j}
    A0_vec     : (N,)   structural TFP endowments
    L_vec      : (N,)   labour endowments (fixed)
    s_vec      : (N,)   retention rates
    sigma_vec  : (N,)   true return volatilities per country j
    rho        : float  portfolio inertia in (0,1)
    phi_0      : float  ambiguity premium scaling
    alpha      : float  capital share
    gamma      : float  TFP diminishing returns
    delta      : float  depreciation rate
    max_iter   : int    maximum inner iterations
    tol        : float  convergence tolerance

    Returns
    -------
    K_vec  : (N,)  equilibrium capital
    w_mat  : (N,N) equilibrium portfolio weights
    V_vec  : (N,)  equilibrium firm valuations
    r_vec  : (N,)  equilibrium implied returns
    Q_vec  : (N,)  Tobin's Q per country
    n_iter : int   iterations taken
    """
    N     = len(K_vec)
    K_vec = K_vec.copy()
    w_mat = w_mat.copy()

    for iteration in range(max_iter):

        # 1. Market clearing
        V_vec = market_clearing(W_vec, w_mat)

        # 2. Implied returns
        r_vec = np.array([
            implied_return(
                expected_dividend(A0_vec[j], K_vec[j], L_vec[j], s_vec[j], alpha, gamma),
                K_vec[j], K_vec_prev[j], V_vec[j]
            )
            for j in range(N)
        ])

        # 3. Reference points — one per (i,j) but same across i
        r_0_vec = r_vec.copy()   # r_0_{i,j} = r_j for all i

        # 4. Friction-adjusted volatilities
        sigma_tilde_mat = np.array([
            [sigma_tilde(sigma_vec[j], phi_0, omega_mat[i, j]) for j in range(N)]
            for i in range(N)
        ])

        # 5. CPT portfolio optimisation per household
        w_mat_new = np.zeros((N, N))
        for i in range(N):
            w_mat_new[i], _ = cpt_portfolio(
                r_vec, r_0_vec, sigma_tilde_mat[i],
                w_mat_prev[i], rho,
                alpha_v, beta_v, lam, gamma_plus, delta_minus
            )

        # 6. Firm capital adjustment
        K_vec_new = np.array([
            equilibrium_capital(A0_vec[j], L_vec[j], alpha, gamma, r_vec[j], delta)
            for j in range(N)
        ])

        # 7. Convergence
        mpk_vec = np.array([mpk(A0_vec[j], K_vec_new[j], L_vec[j], alpha, gamma)
                            for j in range(N)])
        loss = np.sum((mpk_vec - (r_vec + delta)) ** 2)

        w_mat = w_mat_new
        K_vec = K_vec_new

        if loss < tol:
            break

    V_vec = market_clearing(W_vec, w_mat)
    Q_vec = V_vec / K_vec

    return K_vec, w_mat, V_vec, r_vec, Q_vec, iteration + 1


# ====================================
# Realisation step
# ====================================
def realise_period(K_vec: np.ndarray,
                   K_vec_prev: np.ndarray,
                   V_vec: np.ndarray,
                   w_mat: np.ndarray,
                   W_vec: np.ndarray,
                   A0_vec: np.ndarray,
                   L_vec: np.ndarray,
                   s_vec: np.ndarray,
                   alpha: float,
                   gamma: float,
                   delta: float,
                   sigma_eps: float,
                   rng: np.random.Generator) -> tuple:
    """
    Draw TFP shocks, realise output, update wealth and capital.

    Called after inner_equilibrium has fixed K_vec and w_mat.
    Firms cannot react to the shock — K is already determined.

    Wealth update uses portfolio rate-of-return:
        r_j     = (D_j + K_j - K_{j-1}) / V_j
        W_new_i = W_i * (1 + sum_j w_{i,j} * r_j)

    Parameters
    ----------
    K_vec      : (N,)  equilibrium capital from inner loop
    K_vec_prev : (N,)  capital previous period
    V_vec      : (N,)  firm valuations from inner loop
    w_mat      : (N,N) equilibrium portfolio weights
    W_vec      : (N,)  household wealth entering period
    A0_vec     : (N,)  structural TFP
    L_vec      : (N,)  labour
    s_vec      : (N,)  retention rates
    alpha      : float capital share
    gamma      : float TFP diminishing returns
    delta      : float depreciation
    sigma_eps  : float TFP shock std dev
    rng        : np.random.Generator

    Returns
    -------
    Y_vec     : (N,)   realised output
    D_vec     : (N,)   realised dividends
    W_vec_new : (N,)   updated household wealth
    K_vec_new : (N,)   updated capital stocks
    cpis_mat  : (N,N)  bilateral portfolio positions
    """
    N = len(K_vec)

    A_vec = np.array([realised_tfp(A0_vec[j], K_vec[j], gamma, sigma_eps, rng)
                      for j in range(N)])
    Y_vec = np.array([output(A_vec[j], K_vec[j], L_vec[j], alpha) for j in range(N)])
    D_vec = np.array([dividend(Y_vec[j], s_vec[j], alpha) for j in range(N)])

    W_vec_new = np.array([
        update_wealth(w_mat[i], D_vec, K_vec, K_vec_prev, V_vec, W_vec[i])
        for i in range(N)
    ])
    K_vec_new = np.array([
        capital_next_period(Y_vec[j], K_vec[j], s_vec[j], delta, alpha)
        for j in range(N)
    ])
    cpis_mat = np.array([
        bilateral_positions(w_mat[i], W_vec_new[i]) for i in range(N)
    ])

    return Y_vec, D_vec, W_vec_new, K_vec_new, cpis_mat


# ====================================
# Full simulation
# ====================================
def run_simulation(initial_state: dict,
                   params: dict,
                   omega_panel: dict,
                   T: int,
                   seed: int = 42) -> list:
    """
    Run the full simulation over T periods.

    Each period:
        1. inner_equilibrium  -> K*, w*, V*, r*, Q*
        2. realise_period     -> Y, D, W_new, K_new, CPIS

    State carried forward: K_vec, K_vec_prev, W_vec, w_mat, cpis_mat.

    Parameters
    ----------
    initial_state : dict with keys:
        K_vec    : (N,)   initial capital stocks
        W_vec    : (N,)   initial household wealth.
                          Initialise as W_i = PE_ratio * mean(K) * N
                          so that V_j ~ K_j and Q ~ 1 at t=0.
        w_mat    : (N,N)  initial portfolio weights
        cpis_mat : (N,N)  initial bilateral positions

    params : dict with keys:
        A0_vec, L_vec, s_vec, sigma_vec : arrays
        alpha, gamma, delta, rho, phi_0, sigma_eps : floats

    omega_panel : dict  {t: (N,N) omega matrix} for t in 0..T-1

    T    : int  number of periods
    seed : int  random seed

    Returns
    -------
    list of dicts, one per period, each containing:
        K_vec, W_vec, w_mat, cpis_mat  (end-of-period state)
        Y_vec, D_vec, V_vec, r_vec, Q_vec  (period diagnostics)
        n_iter  (inner loop iterations)
    """
    rng       = np.random.default_rng(seed)
    K_vec     = initial_state["K_vec"].copy()
    K_prev    = initial_state["K_vec"].copy()   # tracks previous period K
    W_vec     = initial_state["W_vec"].copy()
    w_mat     = initial_state["w_mat"].copy()
    w_mat_prev= initial_state["w_mat"].copy()
    history   = []

    for t in range(T):

        K_vec, w_mat, V_vec, r_vec, Q_vec, n_iter = inner_equilibrium(
            K_vec      = K_vec,
            K_vec_prev = K_prev,
            W_vec      = W_vec,
            w_mat      = w_mat,
            w_mat_prev = w_mat_prev,
            omega_mat  = omega_panel[t],
            A0_vec     = params["A0_vec"],
            L_vec      = params["L_vec"],
            s_vec      = params["s_vec"],
            sigma_vec  = params["sigma_vec"],
            rho        = params["rho"],
            phi_0      = params["phi_0"],
            alpha      = params["alpha"],
            gamma      = params["gamma"],
            delta      = params["delta"],
        )

        Y_vec, D_vec, W_vec_new, K_vec_new, cpis_mat = realise_period(
            K_vec      = K_vec,
            K_vec_prev = K_prev,
            V_vec      = V_vec,
            w_mat      = w_mat,
            W_vec      = W_vec,
            A0_vec     = params["A0_vec"],
            L_vec      = params["L_vec"],
            s_vec      = params["s_vec"],
            alpha      = params["alpha"],
            gamma      = params["gamma"],
            delta      = params["delta"],
            sigma_eps  = params["sigma_eps"],
            rng        = rng,
        )

        history.append({
            "K_vec"   : K_vec_new,
            "W_vec"   : W_vec_new,
            "w_mat"   : w_mat,
            "cpis_mat": cpis_mat,
            "Y_vec"   : Y_vec,
            "D_vec"   : D_vec,
            "V_vec"   : V_vec,
            "r_vec"   : r_vec,
            "Q_vec"   : Q_vec,
            "n_iter"  : n_iter,
        })

        # Roll state forward
        K_prev     = K_vec.copy()
        K_vec      = K_vec_new
        W_vec      = W_vec_new
        w_mat_prev = w_mat.copy()

    return history