"""
simulation.py  —  EU Capital Markets Union Simulation  v5
==========================================================

Changes relative to v4
-----------------------
1.  CPT removed entirely — replaced with mean-variance portfolio optimisation
        - Closed-form solution: w*_j ∝ r_j / (gamma * sigma_tilde_j²)
        - No short-selling (floor at zero), renormalised to sum to 1
        - Single risk-aversion parameter gamma_ra replaces 6 CPT params
2.  Implied return derived from MPK, not Tobin's Q
        - r_impl_j = MPK(K_j) - delta_j  (pure fundamentals)
        - Decouples returns from market valuation V_vec
        - Tobin's Q retained as post-equilibrium diagnostic only
3.  Effective labour: L_eff = hc * L throughout production function
        - compute_A0, mpk_scalar, realise_period all use hc-adjusted L
4.  Retention rate s_vec from PWT inv_share_j (per country, per year)
        - Replaces fixed scalar s_vec = 0.40
        - Clipped to [0.05, 0.60] for numerical stability
5.  K_prev_vec fixed — now correctly pulled from results[-1]["K_star"]
        - Previously always equalled K_vec (zero delta_K every period)
6.  inner_equilibrium returns 5 values, not 6 (Q_vec removed from signature)
        - Q_vec computed in run_simulation after equilibrium as diagnostic
7.  mpk_vec stored in results dict for convergence diagnostics
8.  print_summary updated: Q relabelled diagnostic, MPK spread added

Unchanged from v4
-----------------
- sigma_tilde / Omega information friction structure
- compute_omega_mat and update_omega_dynamic
- equilibrium_capital_brent (Brent solver for K*)
- market_clearing
- realise_period structure (only L → L_eff substitution)
- Forward-fill of state variables to t+1
- Dynamic Omega via FDI feedback

Calibrated parameters
---------------------
theta* = { phi_0, rho, gamma_ra, beta_1 ... beta_4 }
            ↑       ↑      ↑           ↑
      ambiguity  inertia  risk    gravity elasticities
      premium            aversion

Units
-----
All monetary values in millions of 2021 USD (PWT units).
Rates (alpha, delta, r) are dimensionless fractions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from tqdm import tqdm


# ===========================================================================
# 1. PORTFOLIO OPTIMISATION — MEAN-VARIANCE
# ===========================================================================

def mv_portfolio(r_vec, sigma_tilde_vec, w_prev_i, rho, gamma):
    """
    Mean-variance optimal portfolio for household i.

    Assumes diagonal covariance matrix (country returns uncorrelated).
    Closed-form solution under no-short-selling constraint:

        w*_j  ∝  max(r_j, 0) / (gamma * sigma_tilde_j²)

    Then apply inertia:

        w = rho * w_prev + (1 - rho) * w*,  renormalised

    The information friction enters through sigma_tilde_vec: a country
    that is culturally/geographically distant has higher perceived
    volatility → lower weight, generating home bias without CPT.

    Parameters
    ----------
    r_vec           : (N,) risk-adjusted expected returns per destination
    sigma_tilde_vec : (N,) friction-adjusted perceived vol for investor i
    w_prev_i        : (N,) previous period weights for household i
    rho             : float  portfolio inertia  [0 = none, 1 = fully sticky]
    gamma           : float  coefficient of relative risk aversion

    Returns
    -------
    w : (N,) portfolio weights summing to 1
    """
    var_vec = np.maximum(sigma_tilde_vec ** 2, 1e-12)

    # Score: higher return, lower perceived variance → higher allocation
    score = r_vec / (gamma * var_vec)

    # No short-selling: floor negative scores at zero
    w_star = np.maximum(score, 0.0)

    total = w_star.sum()
    if total < 1e-12:
        # Degenerate case (all returns negative): equal-weight fallback
        w_star = np.ones(len(r_vec)) / len(r_vec)
    else:
        w_star = w_star / total

    # Apply inertia and renormalise
    w = rho * w_prev_i + (1.0 - rho) * w_star
    w = np.maximum(w, 0.0)
    return w / w.sum()


# ===========================================================================
# 2. INFORMATION FRICTIONS
# ===========================================================================

def sigma_tilde(sigma, phi_0, omega):
    """
    Friction-adjusted perceived volatility (Van Nieuwerburgh & Veldkamp 2009).

        sigma_tilde = sigma + phi_0 / (1 + ln(1 + Omega))

    Omega → ∞ : sigma_tilde → sigma          (frictionless, full information)
    Omega → 0 : sigma_tilde → sigma + phi_0  (full ambiguity)

    Parameters
    ----------
    sigma : float  fundamental return volatility for country j
    phi_0 : float  ambiguity premium scale (calibrated via SMM)
    omega : float  information level for this (investor i, asset j) pair
    """
    return sigma + phi_0 / (1.0 + np.log(1.0 + max(omega, 0.0)))


def compute_omega_mat(year_df, countries, coef_map, domestic_omega=1e6):
    """
    Build N×N Omega matrix from bilateral dataframe for one year.

        ln(Omega_{i,j}) = Σ_k  beta_k * x_k    (gravity equation)

    Domestic pairs (i == j) get Omega = domestic_omega (perfect information).

    Parameters
    ----------
    year_df       : pd.DataFrame  bilateral panel for one year
    countries     : list          ordered list of ISO3 codes
    coef_map      : dict          {column_name: coefficient}
    domestic_omega: float         Omega assigned to domestic pairs
    """
    N         = len(countries)
    idx_map   = {c: i for i, c in enumerate(countries)}
    omega_mat = np.ones((N, N)) * 0.1   # default: low information

    for _, row in year_df.iterrows():
        ci = row.get("iso3_i")
        cj = row.get("iso3_j")
        if ci not in idx_map or cj not in idx_map:
            continue
        i, j = idx_map[ci], idx_map[cj]
        if ci == cj:
            omega_mat[i, j] = domestic_omega
        else:
            ln_omega = sum(
                coef * row[col]
                for col, coef in coef_map.items()
                if col in row.index and pd.notna(row[col])
            )
            omega_mat[i, j] = np.exp(np.clip(ln_omega, -10.0, 10.0))

    return omega_mat


def update_omega_dynamic(omega_dyn_mat, w_mat, W_vec,
                         legal_mat, legal_star,
                         kappa, omega_max, omega_bar):
    """
    Update endogenous Omega component from FDI feedback.

        Omega_dyn_{t+1} = Omega_dyn_t + kappa * FDI_{i,j,t} * 1[Legal_{i,j} > Legal*]

    Capped by a logistic ceiling to prevent unbounded growth.
    The CMU shock (reducing Legal barriers) directly gates this accumulation:
    pairs that become more open accumulate information faster.

    Parameters
    ----------
    omega_dyn_mat : (N,N) current dynamic Omega component
    w_mat         : (N,N) portfolio weights  (rows = investor, cols = asset)
    W_vec         : (N,)  household wealth
    legal_mat     : (N,N) legal openness index per pair
    legal_star    : float threshold — pairs above this accumulate info
    kappa         : float accumulation speed
    omega_max     : float logistic ceiling
    omega_bar     : float inflection point of logistic ceiling
    """
    fdi_mat   = w_mat * W_vec[:, None]
    indicator = (legal_mat > legal_star).astype(float)
    raw       = omega_dyn_mat + kappa * fdi_mat * indicator
    ceiling   = omega_max / (1.0 + np.exp(-kappa * (raw - omega_bar)))
    return np.minimum(raw, ceiling)


# ===========================================================================
# 3. FIRM / PRODUCTION FUNCTIONS
# ===========================================================================

def compute_A0(Y_vec, K_vec, L_vec, alpha_vec, hc_vec=None):
    """
    Back-calculate structural TFP level from PWT data.

    Production function:  Y = A0 * K^alpha * L_eff^(1-alpha)
    Inversion:            A0 = Y / (K^alpha * L_eff^(1-alpha))

    If hc_vec is provided, effective labour L_eff = hc * L.
    This adjusts for cross-country differences in human capital
    (Mincerian quality adjustment) and is the preferred specification.

    Parameters
    ----------
    Y_vec     : (N,) real GDP  (PWT rgdpna, millions 2021 USD)
    K_vec     : (N,) capital stock  (PWT rnna)
    L_vec     : (N,) employed persons  (PWT emp, millions)
    alpha_vec : (N,) capital income shares  (= 1 - labsh)
    hc_vec    : (N,) human capital index  (PWT hc); None → raw L
    """
    if hc_vec is not None:
        L_eff = np.maximum(L_vec * hc_vec, 1e-12)
    else:
        L_eff = np.maximum(L_vec, 1e-12)

    K_term = np.maximum(K_vec, 1e-12) ** alpha_vec
    L_term = L_eff ** (1.0 - alpha_vec)
    return np.maximum(Y_vec, 1e-12) / (K_term * L_term)


def mpk_scalar(A0_j, K_j, L_eff_j, alpha_j):
    """
    Marginal product of capital for country j.

        MPK_j = alpha_j * A0_j * K_j^(alpha_j - 1) * L_eff_j^(1 - alpha_j)
              = alpha_j * Y_j / K_j  (equivalent form)

    L_eff_j should be hc_j * L_j (effective labour), not raw L_j.

    Parameters
    ----------
    A0_j    : float  structural TFP
    K_j     : float  capital stock
    L_eff_j : float  effective labour (hc * emp)
    alpha_j : float  capital income share
    """
    return (alpha_j
            * A0_j
            * max(K_j, 1e-12) ** (alpha_j - 1.0)
            * max(L_eff_j, 1e-12) ** (1.0 - alpha_j))


def equilibrium_capital_brent(A0_j, L_eff_j, alpha_j, r_impl_j, delta_j,
                              K_lo=1e-3, K_hi=1e15):
    """
    Solve MPK(K*) = r_impl_j + delta_j for K* via Brent's method.

    MPK is strictly decreasing in K under Cobb-Douglas, so the residual
        f(K) = MPK(K) - (r_impl_j + delta_j)
    has exactly one zero on (0, ∞).

    Edge cases:
        - target ≤ 0        : return None  (caller uses K_prev as fallback)
        - f(K_lo) ≤ 0       : target too high → K* very small → return K_lo
        - f(K_hi) ≥ 0       : target too low  → K* very large → return K_hi

    Parameters
    ----------
    A0_j      : float  TFP
    L_eff_j   : float  effective labour
    alpha_j   : float  capital share
    r_impl_j  : float  MPK-implied expected return (net of depreciation)
    delta_j   : float  depreciation rate
    """
    target = r_impl_j + delta_j
    if target <= 0.0:
        return None

    f = lambda K: mpk_scalar(A0_j, K, L_eff_j, alpha_j) - target

    if f(K_lo) <= 0.0:
        return K_lo
    if f(K_hi) >= 0.0:
        return K_hi

    return brentq(f, K_lo, K_hi, xtol=1e-9)


# ===========================================================================
# 4. EQUILIBRIUM
# ===========================================================================

def market_clearing(W_vec, w_mat):
    """
    Compute capital supplied to each destination country.

        V_j = Σ_i  w_{i,j} * W_i

    Parameters
    ----------
    W_vec : (N,)   household wealth by investor country
    w_mat : (N,N)  portfolio weights  (rows = investor i, cols = asset j)

    Returns
    -------
    V_vec : (N,)  total capital invested in each country j
    """
    return w_mat.T @ W_vec


def inner_equilibrium(K_vec, K_prev_vec, W_vec, w_mat, w_mat_prev,
                      omega_mat, A0_vec, L_eff_vec, s_vec,
                      alpha_vec, delta_vec, sigma_vec,
                      params, max_iter=50, tol=1e-6):
    """
    Iterate between mean-variance portfolio choice and firm capital adjustment
    until MPK_j ≈ r_impl_j + delta_j for all j.

    Iteration sequence per step
    ---------------------------
    1. Compute MPK-based implied returns:
           r_impl_j = MPK(K_j) - delta_j
       (Pure fundamentals — decoupled from market valuation V_vec)

    2. Build friction-adjusted volatility matrix sigma_tilde_{i,j}
       from Omega and phi_0.

    3. Mean-variance portfolio per household i → w_new[i]

    4. Update capital stocks via Brent solver:
           K*_j  s.t.  MPK(K*_j) = r_impl_j + delta_j

    5. Check convergence: Σ_j (MPK_j - r_impl_j - delta_j)² < tol

    Note: V_vec (market clearing) is computed once at the end for
    diagnostics and wealth updates. It is NOT used to derive r_impl,
    which is the key departure from v4.

    Parameters
    ----------
    K_vec      : (N,)   capital stocks entering this period
    K_prev_vec : (N,)   capital stocks from previous period
    W_vec      : (N,)   household wealth
    w_mat      : (N,N)  warm-start portfolio weights
    w_mat_prev : (N,N)  previous period weights (for inertia)
    omega_mat  : (N,N)  information levels (structural + dynamic)
    A0_vec     : (N,)   structural TFP levels
    L_eff_vec  : (N,)   effective labour (hc * emp)
    s_vec      : (N,)   retention/investment rates (from PWT inv_share)
    alpha_vec  : (N,)   capital income shares
    delta_vec  : (N,)   depreciation rates
    sigma_vec  : (N,)   fundamental return volatilities per country j
    params     : dict   phi_0, rho, gamma_ra
    max_iter   : int    maximum inner iterations
    tol        : float  convergence threshold on MPK residual

    Returns
    -------
    K_vec    : (N,)   equilibrium capital stocks
    w_mat    : (N,N)  equilibrium portfolio weights
    V_vec    : (N,)   market-clearing capital supplied per country
    r_impl   : (N,)   MPK-implied expected returns (net of depreciation)
    mpk_vec  : (N,)   final MPK values (convergence diagnostic)
    n_iter   : int    iterations taken
    """
    N     = len(K_vec)
    phi_0 = params["phi_0"]
    rho   = params["rho"]
    gamma = params["gamma_ra"]

    K_vec = K_vec.copy()
    w_mat = w_mat.copy()

    r_impl  = np.zeros(N)
    mpk_vec = np.zeros(N)

    for iteration in range(max_iter):

        # ── 1. MPK-based implied returns ───────────────────────────────
        # r_impl_j = MPK(K_j) - delta_j
        # This is the expected return a firm offers on a marginal unit of
        # capital, derived purely from production fundamentals.
        r_impl = np.array([
            mpk_scalar(A0_vec[j], K_vec[j], L_eff_vec[j], alpha_vec[j])
            - delta_vec[j]
            for j in range(N)
        ])

        # ── 2. Friction-adjusted volatility matrix ─────────────────────
        sigma_tilde_mat = np.array([
            [sigma_tilde(sigma_vec[j], phi_0, omega_mat[i, j])
             for j in range(N)]
            for i in range(N)
        ])

        # ── 3. Mean-variance portfolio per household ───────────────────
        w_new = np.zeros((N, N))
        for i in range(N):
            w_new[i] = mv_portfolio(
                r_vec           = r_impl,
                sigma_tilde_vec = sigma_tilde_mat[i],
                w_prev_i        = w_mat_prev[i],
                rho             = rho,
                gamma           = gamma,
            )

        # ── 4. Firm capital via Brent solver ───────────────────────────
        K_new = np.zeros(N)
        for j in range(N):
            K_j = equilibrium_capital_brent(
                A0_j     = A0_vec[j],
                L_eff_j  = L_eff_vec[j],
                alpha_j  = alpha_vec[j],
                r_impl_j = r_impl[j],
                delta_j  = delta_vec[j],
            )
            K_new[j] = K_j if K_j is not None else K_vec[j]

        # ── 5. Convergence: MPK residual ───────────────────────────────
        mpk_vec = np.array([
            mpk_scalar(A0_vec[j], K_new[j], L_eff_vec[j], alpha_vec[j])
            for j in range(N)
        ])
        loss = np.sum((mpk_vec - (r_impl + delta_vec)) ** 2)

        w_mat = w_new
        K_vec = K_new

        if loss < tol:
            break

    # Market clearing computed once at equilibrium (diagnostic / wealth update)
    V_vec = market_clearing(W_vec, w_mat)
    V_vec = np.maximum(V_vec, 1e-12)

    return K_vec, w_mat, V_vec, r_impl, mpk_vec, iteration + 1


# ===========================================================================
# 5. REALISATION
# ===========================================================================

def realise_period(K_vec, V_vec, w_mat, W_vec,
                   A0_vec, L_eff_vec, s_vec, alpha_vec, delta_vec,
                   sigma_A_frac, rng):
    """
    Draw TFP shocks, compute realised output and dividends, update state.

    TFP shock:
        A_real = A0 + eps,  eps ~ N(0, (sigma_A_frac * A0)²)

    Capital accumulation (standard perpetual inventory):
        K_new = (1 - delta) * K + s * alpha * Y_real

    Wealth update (portfolio rate-of-return):
        r_real_j   = (D_j + K_new_j - K_j) / V_j
        W_new_i    = W_i * (1 + Σ_j w_{i,j} * r_real_j)

    Parameters
    ----------
    K_vec        : (N,)   equilibrium capital stocks (from inner loop)
    V_vec        : (N,)   market-clearing capital supplied per country
    w_mat        : (N,N)  equilibrium portfolio weights
    W_vec        : (N,)   household wealth entering period
    A0_vec       : (N,)   structural TFP
    L_eff_vec    : (N,)   effective labour (hc * emp)
    s_vec        : (N,)   retention rates (from PWT inv_share)
    alpha_vec    : (N,)   capital income shares
    delta_vec    : (N,)   depreciation rates
    sigma_A_frac : float  TFP shock std as fraction of A0
    rng          : np.random.Generator

    Returns
    -------
    Y_vec    : (N,)   realised output
    D_vec    : (N,)   realised dividends
    W_new    : (N,)   updated household wealth
    K_new    : (N,)   updated capital stocks
    cpis_mat : (N,N)  bilateral investment positions
    A_real   : (N,)   realised TFP (A0 + shock)
    r_real   : (N,)   realised returns per country
    """
    N = len(K_vec)

    # TFP shock
    sigma_A_vec = sigma_A_frac * A0_vec
    eps         = rng.normal(0.0, 1.0, size=N)
    A_real      = np.maximum(A0_vec + sigma_A_vec * eps, 1e-12)

    # Realised output and dividends (using effective labour)
    Y_vec = A_real * K_vec ** alpha_vec * L_eff_vec ** (1.0 - alpha_vec)
    D_vec = (1.0 - s_vec) * alpha_vec * Y_vec

    # Capital accumulation
    K_new = (1.0 - delta_vec) * K_vec + s_vec * alpha_vec * Y_vec

    # Realised rate of return per country
    r_real = (D_vec + K_new - K_vec) / np.maximum(V_vec, 1e-12)

    # Wealth update via portfolio return
    W_new = np.array([
        W_vec[i] * (1.0 + np.dot(w_mat[i], r_real))
        for i in range(N)
    ])
    W_new = np.maximum(W_new, 1e-12)

    # Bilateral CPIS investment positions
    cpis_mat = w_mat * W_new[:, None]

    return Y_vec, D_vec, W_new, K_new, cpis_mat, A_real, r_real


# ===========================================================================
# 6. MAIN SIMULATION
# ===========================================================================

def run_simulation(
    df: pd.DataFrame,
    countries: list,
    params: dict,
    coef_map: dict,
    T: int = 10,
    seed: int = 42,
    cmu_shock: float = 0.0,
    cmu_pairs: list = None,
    tqdm_disable: bool = False,
):
    """
    Run the full EU CMU simulation over T periods.

    The simulation has two nested loops:
        Outer: period loop t = 0 … T
        Inner: equilibrium loop (mean-variance ↔ Brent) per period

    CMU shock
    ---------
    If cmu_shock > 0, the legal distance for pairs in cmu_pairs is scaled:
        Legal(i,j) → Legal(i,j) * (1 - cmu_shock)
    This reduces barriers and accelerates Omega accumulation for those pairs.
    cmu_pairs defaults to all (i,j) pairs within `countries` if not supplied.

    Parameters
    ----------
    df           : pd.DataFrame  bilateral panel (columns per COLUMN_MAP)
    countries    : list          ordered ISO3 country codes
    params       : dict          model parameters (see below)
    coef_map     : dict          gravity coefficients {column: beta}
    T            : int           number of periods to simulate
    seed         : int           random seed for TFP shocks
    cmu_shock    : float         CMU barrier reduction  [0 = baseline, 1 = full]
    cmu_pairs    : list of tuple (iso3_i, iso3_j) pairs affected by CMU shock;
                                 None → all within-country pairs in `countries`
    tqdm_disable : bool          suppress progress bar

    Required params keys
    --------------------
    phi_0        : float  ambiguity premium scaling           (SMM calibrated)
    rho          : float  portfolio inertia  [0, 1]           (SMM calibrated)
    gamma_ra     : float  risk aversion coefficient           (SMM calibrated)
    sigma_A_frac : float  TFP shock std as fraction of A0
    kappa        : float  dynamic Omega accumulation speed
    omega_max    : float  logistic ceiling on dynamic Omega
    omega_bar    : float  inflection of logistic ceiling
    legal_star   : float  legal index threshold for Omega accumulation

    Wealth initialisation
    ---------------------
    Autarky start: w_ii = 1  →  V_j = W_j = K_j  →  Q = 1 at t = 0.
    W_i initialised from diagonal of panel (iso3_i == iso3_j).
    """
    rng        = np.random.default_rng(seed)
    N          = len(countries)
    sim_df     = df.copy()
    start_year = int(sim_df["year"].min())
    end_year   = start_year + T - 1

    # Default CMU pairs: all within-sample pairs
    if cmu_pairs is None:
        cmu_pairs = [
            (ci, cj)
            for ci in countries
            for cj in countries
            if ci != cj
        ]
    cmu_set = set(cmu_pairs)

    # ── Per-country fundamental return volatility from MSCI returns ──────
    # r_j = MSCI_Return_d (equity returns); rf_j = Interest Rate_d (IRR).
    # We use r_j (MSCI) as the empirical return series for sigma estimation.
    sigma_by_country = (
        sim_df[sim_df["iso3_i"] == sim_df["iso3_j"]]
        .groupby("iso3_i")["r_j"]
        .std()
        .reindex(countries)
        .fillna(0.02)
    )
    sigma_vec = sigma_by_country.values

    # ── Initialise simulation state columns ───────────────────────────────
    sim_df["K_sim"]   = sim_df["K_j"]
    sim_df["W_sim"]   = sim_df["K_j"]    # autarky: W = K → Q = 1
    sim_df["FDI_sim"] = 0.0

    # Dynamic Omega component (N×N), initialised to zero
    omega_dyn_mat = np.zeros((N, N))

    # Initial portfolio: autarky (100% domestic)
    w_mat      = np.eye(N)
    w_mat_prev = np.eye(N)

    results = []

    for year in tqdm(range(start_year, end_year + 1),
                     desc="Simulating", disable=tqdm_disable):

        year_df = sim_df[sim_df["year"] == year].copy()

        # ── Extract diagonal (domestic) state vectors ──────────────────────
        diag = (
            year_df[year_df["iso3_i"] == year_df["iso3_j"]]
            .set_index("iso3_i")
            .reindex(countries)
        )

        K_vec     = diag["K_sim"].values.astype(float)
        W_vec     = diag["W_sim"].values.astype(float)
        Y_vec_obs = diag["Y_j"].fillna(1.0).values.astype(float)
        L_vec     = diag["L_j"].fillna(1.0).values.astype(float)
        alpha_vec = np.clip(
            1.0 - diag["lab_sh_j"].fillna(0.33).values, 0.1, 0.9
        )
        delta_vec = diag["delta_j"].fillna(0.05).values.astype(float)

        # Human capital index (PWT hc → hc_j) — quality-adjusts labour input
        hc_vec    = diag["hc_j"].fillna(1.0).values.astype(float)
        L_eff_vec = np.maximum(L_vec * hc_vec, 1e-12)

        # Retention / investment rate from PWT investment share (inv_share_j)
        # Clipped to [0.05, 0.60] for numerical stability
        s_vec = np.clip(
            diag["inv_share_j"].fillna(0.25).values.astype(float),
            0.05, 0.60
        )

        # Back-calculate TFP level consistent with observed Y, K, L_eff
        A0_vec = compute_A0(Y_vec_obs, K_vec, L_vec, alpha_vec, hc_vec)

        # K_prev: use previous period's equilibrium K, or current at t=0
        if results:
            K_prev_vec = results[-1]["K_star"].copy()
        else:
            K_prev_vec = K_vec.copy()

        # ── Omega matrix: structural (gravity) + dynamic (FDI feedback) ───
        omega_struct = compute_omega_mat(year_df, countries, coef_map)
        omega_mat    = omega_struct + omega_dyn_mat

        # ── Legal matrix — apply CMU shock to designated pairs ─────────────
        # legal_i in the CSV = destination Heritage legal score
        # (originally "Overall Score_d", renamed to "legal_i" via COLUMN_MAP).
        # Note: the source-country legal score is also "legal_i" — both origin
        # and destination map to the same name. We use the destination value,
        # which for domestic rows (iso3_i == iso3_j) is the country's own score.
        legal_star = params.get("legal_star", 50.0)
        legal_mat  = np.ones((N, N)) * legal_star

        for _, row in year_df.iterrows():
            ci = row.get("iso3_i")
            cj = row.get("iso3_j")
            if ci in countries and cj in countries:
                i = countries.index(ci)
                j = countries.index(cj)
                raw_val = row.get("legal_i")
                if pd.notna(raw_val):
                    base_legal = float(raw_val)
                    if (ci, cj) in cmu_set and cmu_shock > 0.0:
                        legal_mat[i, j] = base_legal * (1.0 - cmu_shock)
                    else:
                        legal_mat[i, j] = base_legal

        # ── Inner equilibrium ──────────────────────────────────────────────
        K_star, w_mat, V_vec, r_impl, mpk_vec, n_iter = inner_equilibrium(
            K_vec      = K_vec,
            K_prev_vec = K_prev_vec,
            W_vec      = W_vec,
            w_mat      = w_mat,
            w_mat_prev = w_mat_prev,
            omega_mat  = omega_mat,
            A0_vec     = A0_vec,
            L_eff_vec  = L_eff_vec,
            s_vec      = s_vec,
            alpha_vec  = alpha_vec,
            delta_vec  = delta_vec,
            sigma_vec  = sigma_vec,
            params     = params,
        )

        # ── Tobin's Q — post-equilibrium diagnostic only ───────────────────
        Q_vec = V_vec / np.maximum(K_star, 1e-12)

        # ── Realisation ────────────────────────────────────────────────────
        Y_vec, D_vec, W_new, K_new, cpis_mat, A_real, r_real = realise_period(
            K_vec        = K_star,
            V_vec        = V_vec,
            w_mat        = w_mat,
            W_vec        = W_vec,
            A0_vec       = A0_vec,
            L_eff_vec    = L_eff_vec,
            s_vec        = s_vec,
            alpha_vec    = alpha_vec,
            delta_vec    = delta_vec,
            sigma_A_frac = params.get("sigma_A_frac", 0.02),
            rng          = rng,
        )

        # ── Update dynamic Omega via FDI feedback ──────────────────────────
        omega_dyn_mat = update_omega_dynamic(
            omega_dyn_mat = omega_dyn_mat,
            w_mat         = w_mat,
            W_vec         = W_new,
            legal_mat     = legal_mat,
            legal_star    = legal_star,
            kappa         = params.get("kappa", 0.001),
            omega_max     = params.get("omega_max", 10.0),
            omega_bar     = params.get("omega_bar", 5.0),
        )

        # ── Store results ──────────────────────────────────────────────────
        home_share = np.array([w_mat[i, i] for i in range(N)])

        results.append({
            "year"       : year,
            "countries"  : countries,
            "K_star"     : K_star.copy(),
            "W_vec"      : W_vec.copy(),
            "W_new"      : W_new.copy(),
            "r_implied"  : r_impl.copy(),
            "r_real"     : r_real.copy(),
            "mpk_vec"    : mpk_vec.copy(),   # convergence diagnostic
            "Y"          : Y_vec.copy(),
            "D"          : D_vec.copy(),
            "A_real"     : A_real.copy(),
            "V_vec"      : V_vec.copy(),
            "Q_vec"      : Q_vec.copy(),     # diagnostic only (not equilibrium criterion)
            "Wmat"       : w_mat.copy(),
            "cpis_mat"   : cpis_mat.copy(),
            "home_share" : home_share.copy(),
            "n_iter"     : n_iter,
            "s_vec"      : s_vec.copy(),
            "alpha_vec"  : alpha_vec.copy(),
            "delta_vec"  : delta_vec.copy(),
        })

        # ── Forward-fill state to t+1 ──────────────────────────────────────
        if year < end_year:
            nxt = sim_df["year"] == (year + 1)
            for idx, c in enumerate(countries):
                mask_j  = nxt & (sim_df["iso3_j"] == c)
                mask_i  = nxt & (sim_df["iso3_i"] == c)
                sim_df.loc[mask_j, "K_sim"] = K_new[idx]
                sim_df.loc[mask_i, "W_sim"] = W_new[idx]
                for jdx, cj in enumerate(countries):
                    mask_ij = (nxt
                               & (sim_df["iso3_i"] == c)
                               & (sim_df["iso3_j"] == cj))
                    sim_df.loc[mask_ij, "FDI_sim"] = cpis_mat[idx, jdx]

        # Roll portfolio state
        w_mat_prev = w_mat.copy()

    return sim_df, results


# ===========================================================================
# 7. SUMMARY OUTPUT
# ===========================================================================

def print_summary(results, scale=1e6, scale_label="(tn)"):
    """
    Print simulation summary table.

    Columns per country:
        W, K, Y, D : levels at t0 and t-1, plus % change
        r_impl     : MPK-implied return (net of depreciation)
        MPK        : marginal product of capital
        Q (diag)   : Tobin's Q as post-equilibrium diagnostic
        Home%      : domestic portfolio share
        A_real     : realised TFP in final period

    Also prints:
        - Portfolio weight matrix (final year)
        - MPK convergence table across years (replaces Q as the key metric)
        - Capital misallocation: σ²(r_implied) per year
    """
    first, last = results[0], results[-1]

    def pct(new, old):
        return (new - old) / abs(old) * 100 if abs(old) > 1e-12 else float("nan")

    rows = []
    for idx, c in enumerate(first["countries"]):
        rows.append({
            "Country"                  : c,
            f"W t0 {scale_label}"      : first["W_vec"][idx]  / scale,
            f"W t-1 {scale_label}"     : last["W_new"][idx]   / scale,
            "ΔW (%)"                   : pct(last["W_new"][idx],  first["W_vec"][idx]),
            f"K t0 {scale_label}"      : first["K_star"][idx] / scale,
            f"K t-1 {scale_label}"     : last["K_star"][idx]  / scale,
            "ΔK (%)"                   : pct(last["K_star"][idx], first["K_star"][idx]),
            f"Y t0 {scale_label}"      : first["Y"][idx]      / scale,
            f"Y t-1 {scale_label}"     : last["Y"][idx]       / scale,
            "ΔY (%)"                   : pct(last["Y"][idx],      first["Y"][idx]),
            f"D t0 {scale_label}"      : first["D"][idx]      / scale,
            f"D t-1 {scale_label}"     : last["D"][idx]       / scale,
            "r_impl t0"                : round(first["r_implied"][idx], 4),
            "r_impl t-1"               : round(last["r_implied"][idx],  4),
            "Δr_impl"                  : round(
                last["r_implied"][idx] - first["r_implied"][idx], 4),
            "MPK t0"                   : round(first["mpk_vec"][idx], 4),
            "MPK t-1"                  : round(last["mpk_vec"][idx],  4),
            "Q (diag) t0"              : round(first["Q_vec"][idx], 3),
            "Q (diag) t-1"             : round(last["Q_vec"][idx],  3),
            "Home% t0"                 : round(first["home_share"][idx] * 100, 1),
            "Home% t-1"                : round(last["home_share"][idx]  * 100, 1),
            "A_real t-1"               : round(last["A_real"][idx], 3),
        })

    summary = pd.DataFrame(rows).set_index("Country")
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print(f"\n{'='*80}")
    print(f"  SIMULATION SUMMARY  |  years {first['year']} → {last['year']}  "
          f"({len(results)} periods)")
    print(f"{'='*80}")
    print(summary.T.to_string())

    print(f"\n{'─'*60}")
    print("  PORTFOLIO WEIGHTS  (final year %, rows=investor, cols=asset)")
    print(f"{'─'*60}")
    Wmat_df = pd.DataFrame(
        np.round(last["Wmat"] * 100, 1),
        index=last["countries"],
        columns=last["countries"],
    )
    print(Wmat_df.to_string())

    print(f"\n{'─'*60}")
    print("  MPK CONVERGENCE  |  MPK spread and σ²(MPK) per year")
    print("  (convergence means MPK equalising across countries)")
    print(f"{'─'*60}")
    for r in results:
        mpk    = r["mpk_vec"]
        spread = mpk.max() - mpk.min()
        vals   = ", ".join(
            f"{c}={v:.3f}" for c, v in zip(r["countries"], mpk)
        )
        print(f"  {r['year']}:  σ²={np.var(mpk):.6f}  "
              f"spread={spread:.4f}  [{vals}]")

    print(f"\n{'─'*60}")
    print("  CAPITAL MISALLOCATION  |  σ²(r_implied) per year")
    print(f"{'─'*60}")
    for r in results:
        rv     = r["r_implied"]
        spread = rv.max() - rv.min()
        vals   = ", ".join(
            f"{c}={v:.3f}" for c, v in zip(r["countries"], rv)
        )
        print(f"  {r['year']}:  σ²={np.var(rv):.6f}  "
              f"spread={spread:.4f}  [{vals}]")


# ===========================================================================
# 8. EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    from utils.v3.countries import aggregate_row

    # Final.csv already has renamed columns from COLUMN_MAP applied upstream.
    # Column reference:
    #   iso3_i, iso3_j, year
    #   Y_j, K_j, L_j, lab_sh_j, delta_j, hc_j, inv_share_j, A_j, rf_j
    #   r_j (MSCI equity returns), r_i (MSCI source)
    #   legal_i (Heritage Overall Score — both source and destination)
    #   ln_dist, lang, ln_cpis_lag1, border, lang_share, lang_official
    #   cpis, cpis_lag1, ln_cpis, ln_cpis_lag1
    df = pd.read_csv(
        "/Users/jesper/Desktop/CBS/Thesis 1/"
        "Jesper-Liedholm-Thesis-Code/Data/Clean/Final.csv"
    )

    countries     = ["DEU", "FRA", "ITA", "ESP", "SWE"]
    row_countries = set(df["iso3_i"].unique()) - set(countries)
    df = aggregate_row(df, row_countries=row_countries)

    print(df.info())
    
    # ── Gravity coef_map ───────────────────────────────────────────────────
    # Keys are exact column names in Final.csv.
    # legal_i  : Heritage legal openness score (destination)
    # ln_dist  : log bilateral distance (pre-computed in CSV)
    # lang     : language proximity
    # ln_cpis_lag1 : lagged log CPIS (pre-computed in CSV)
    # Replace placeholder values with your OLS estimates from CPIS panel.
    coef_map = {
        "legal_i"      :  0.04,
        "ln_dist"      : -0.30,
        "lang"         :  0.50,
        "ln_cpis_lag1" :  0.20,
    }

    # ── Model parameters ───────────────────────────────────────────────────
    # SMM-calibrated: phi_0, rho, gamma_ra
    # Fixed / pre-estimated: sigma_A_frac, kappa, omega_max, omega_bar, legal_star
    params = {
        "phi_0"        : 0.05,    # ambiguity premium (SMM calibrated)
        "rho"          : 0.70,    # portfolio inertia  (SMM calibrated)
        "gamma_ra"     : 3.00,    # risk aversion      (SMM calibrated; 2–5 typical)
        "sigma_A_frac" : 0.02,    # TFP shock scale (from std of Δ A_j in data)
        "kappa"        : 0.001,   # dynamic Omega accumulation speed
        "omega_max"    : 10.0,    # logistic ceiling on Omega
        "omega_bar"    : 5.0,     # inflection of logistic ceiling
        "legal_star"   : 55.0,    # legal threshold for Omega accumulation
    }

    # ── Baseline simulation (no CMU) ───────────────────────────────────────
    print("Running baseline simulation (no CMU)...")
    sim_df_base, results_base = run_simulation(
        df        = df,
        countries = countries,
        params    = params,
        coef_map  = coef_map,
        T         = 10,
        seed      = 42,
        cmu_shock = 0.0,
    )
    print_summary(results_base)

    # ── CMU counterfactual (moderate: Delta = 0.50) ────────────────────────
    coef_map = {
        "legal_i"      :  0.0,
        "ln_dist"      : -0.30,
        "lang"         :  0.50,
        "ln_cpis_lag1" :  0.20,
    }
    
    print("\nRunning CMU counterfactual (Delta = 0.50)...")
    sim_df_cmu, results_cmu = run_simulation(
        df        = df,
        countries = countries,
        params    = params,
        coef_map  = coef_map,
        T         = 10,
        seed      = 42,
    )
    print_summary(results_cmu)

    # ── Quick comparison: MPK spread baseline vs CMU ───────────────────────
    print(f"\n{'─'*60}")
    print("  MPK SPREAD COMPARISON  |  baseline vs CMU (final year)")
    print(f"{'─'*60}")
    mpk_base = results_base[-1]["mpk_vec"]
    mpk_cmu  = results_cmu[-1]["mpk_vec"]
    print(f"  Baseline  σ²(MPK) = {np.var(mpk_base):.6f}  "
          f"spread = {mpk_base.max() - mpk_base.min():.4f}")
    print(f"  CMU       σ²(MPK) = {np.var(mpk_cmu):.6f}  "
          f"spread = {mpk_cmu.max() - mpk_cmu.min():.4f}")
    print(f"  Reduction in spread: "
          f"{(1 - (mpk_cmu.max()-mpk_cmu.min())/(mpk_base.max()-mpk_base.min()))*100:.1f}%")