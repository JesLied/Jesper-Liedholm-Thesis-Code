"""
simulation.py  —  EU Capital Markets Union Simulation  v4
==========================================================

Fixes relative to v3
---------------------
1.  Gravity-based information frictions → per-(i,j) sigma_tilde
2.  Proper CPT (Tversky & Kahneman 1992) with probability weighting
3.  Brent per country for equilibrium K (replaces joint Nelder-Mead)
4.  Household-specific sigma_tilde_{i,j} drives portfolio choice
5.  countries passed explicitly into run_simulation
6.  Wealth updated via portfolio rate-of-return (units consistent)
7.  r_implied = (D_hat + delta_K) / V — always finite
8.  A0 back-calculated from PWT: A0 = Y / (K^alpha * L^(1-alpha))
9.  Portfolio inertia rho on w_mat weights (not on habit)
10. Dynamic Omega update via lagged FDI feedback

Units
-----
All monetary values in millions of 2021 USD (PWT units).
Rates (alpha, delta, r) are dimensionless fractions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.integrate import quad
from scipy.stats import norm as scipy_norm
from tqdm import tqdm


# ===========================================================================
# 1. CPT FUNCTIONS
# ===========================================================================

def probability_weight(p, delta):
    """
    Tversky-Kahneman probability weighting function.
    delta=0.61 for gains (gamma_plus), delta=0.69 for losses (delta_minus).
    Overweights small p, underweights large p.
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (p ** delta) / (p ** delta + (1 - p) ** delta) ** (1 / delta)


def cpt_value(r_implied, r_0, sigma_tilde_val,
              alpha_v=0.88, beta_v=0.88, lam=2.25,
              gamma_plus=0.61, delta_minus=0.69):
    """
    CPT value for a single asset perceived by a single household.

    Household perceives returns as N(r_implied, sigma_tilde^2).
    Evaluated relative to reference point r_0 = r_implied
    (rational expectations: Koszegi & Rabin 2006).

    x = r - r_0 ~ N(0, sigma_tilde^2)

    Parameters
    ----------
    r_implied      : float  market-implied expected return
    r_0            : float  reference point (set to r_implied)
    sigma_tilde_val: float  friction-adjusted perceived std dev
    """
    s    = max(sigma_tilde_val, 1e-8)
    mu_x = r_implied - r_0   # = 0 under rational expectations reference

    def gain_integrand(x):
        p = np.clip(1 - scipy_norm.cdf(x, mu_x, s), 1e-10, 1 - 1e-10)
        w = probability_weight(p, gamma_plus)
        v = x ** alpha_v
        return w * v * scipy_norm.pdf(x, mu_x, s)

    def loss_integrand(x):
        p = np.clip(scipy_norm.cdf(x, mu_x, s), 1e-10, 1 - 1e-10)
        w = probability_weight(p, delta_minus)
        v = -lam * ((-x) ** beta_v)
        return w * v * scipy_norm.pdf(x, mu_x, s)

    upper = mu_x + 8 * s
    lower = mu_x - 8 * s

    gain, _ = quad(gain_integrand, 0,     upper, limit=80)
    loss, _ = quad(loss_integrand, lower, 0,     limit=80)
    return gain + loss


def cpt_portfolio(r_implied_vec, sigma_tilde_vec, w_prev_i, rho,
                  tau=0.005,
                  alpha_v=0.88, beta_v=0.88, lam=2.25,
                  gamma_plus=0.61, delta_minus=0.69):
    """
    Compute optimal portfolio weights for household i.

    Steps:
        1. Compute CPT value V^CPT_j for each asset j
        2. Softmax allocation: w*_j = exp(V^CPT_j / tau) / sum_k exp(...)
           tau is the temperature — lower = more concentrated on best asset
        3. Apply portfolio inertia: w = rho*w* + (1-rho)*w_prev

    Softmax (entropy regularization) gives interior solutions that reflect
    diversification, unlike a linear CPT objective which always corners.

    Parameters
    ----------
    r_implied_vec  : (N,) expected returns per country
    sigma_tilde_vec: (N,) friction-adjusted perceived vols for this household
    w_prev_i       : (N,) previous period weights for household i
    rho            : float portfolio inertia  (0=no inertia, 1=no rebalancing)
    tau            : float temperature for softmax (lower = sharper home bias)
    """
    N        = len(r_implied_vec)
    cpt_vals = np.array([
        cpt_value(r_implied_vec[j], r_implied_vec[j], sigma_tilde_vec[j],
                  alpha_v, beta_v, lam, gamma_plus, delta_minus)
        for j in range(N)
    ])

    # Softmax with numerical stability shift
    shifted   = cpt_vals - cpt_vals.max()
    exp_vals  = np.exp(shifted / max(tau, 1e-10))
    w_star    = exp_vals / exp_vals.sum()

    # Portfolio inertia
    w = rho * w_star + (1 - rho) * w_prev_i
    return w / w.sum()


# ===========================================================================
# 2. INFORMATION FRICTIONS
# ===========================================================================

def sigma_tilde(sigma, phi_0, omega):
    """
    Friction-adjusted perceived volatility (Van Nieuwerburgh & Veldkamp 2009).

    sigma_tilde = sigma + phi_0 / (1 + ln(1 + Omega))

    Omega -> inf : sigma_tilde -> sigma (no friction)
    Omega -> 0   : sigma_tilde -> sigma + phi_0 (full ambiguity)
    """
    return sigma + phi_0 / (1 + np.log(1 + max(omega, 0)))


def compute_omega_mat(year_df, countries, coef_map, domestic_omega=1e6):
    """
    Build NxN Omega matrix from bilateral dataframe for one year.

    ln(Omega_{i,j}) = sum_k beta_k * x_k  (gravity equation)
    Domestic pairs (i==j) get Omega = domestic_omega (perfect info).

    Parameters
    ----------
    year_df      : pd.DataFrame  bilateral panel for one year
    countries    : list          ordered list of ISO3 codes
    coef_map     : dict          {column_name: coefficient}
    domestic_omega: float        Omega for domestic pairs
    """
    N       = len(countries)
    idx_map = {c: i for i, c in enumerate(countries)}
    omega_mat = np.ones((N, N)) * 0.1   # default: low info

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
            omega_mat[i, j] = np.exp(np.clip(ln_omega, -10, 10))

    return omega_mat


def update_omega_dynamic(omega_dyn_mat, w_mat, W_vec,
                         legal_mat, legal_star,
                         kappa, omega_max, omega_bar):
    """
    Update endogenous Omega component from FDI feedback.

    Omega_dynamic_{t+1} = Omega_t + kappa * FDI_{i,j,t} * 1[Legal > Legal*]
    Capped by logistic ceiling to prevent unbounded growth.

    Parameters
    ----------
    omega_dyn_mat : (N,N) current dynamic component
    w_mat         : (N,N) portfolio weights
    W_vec         : (N,)  household wealth
    legal_mat     : (N,N) legal index per pair
    legal_star    : float threshold for legal openness
    """
    fdi_mat   = w_mat * W_vec[:, None]
    indicator = (legal_mat > legal_star).astype(float)
    raw       = omega_dyn_mat + kappa * fdi_mat * indicator
    ceiling   = omega_max / (1 + np.exp(-kappa * (raw - omega_bar)))
    return np.minimum(raw, ceiling)


# ===========================================================================
# 3. FIRM FUNCTIONS
# ===========================================================================

def compute_A0(Y_vec, K_vec, L_vec, alpha_vec):
    """
    Back-calculate structural TFP from PWT data at initialisation year.

    From Y = A0 * K^alpha * L^(1-alpha):
        A0 = Y / (K^alpha * L^(1-alpha))

    A_i from PWT is a relative index (2021=1), not a level.
    This converts it to the A0 level consistent with the production function.

    Parameters
    ----------
    Y_vec     : (N,) real GDP (PWT rgdpna, millions 2021 USD)
    K_vec     : (N,) capital stock (PWT rnna)
    L_vec     : (N,) employed persons (PWT emp, millions)
    alpha_vec : (N,) capital shares
    """
    K_term = np.maximum(K_vec, 1e-12) ** alpha_vec
    L_term = np.maximum(L_vec, 1e-12) ** (1 - alpha_vec)
    return np.maximum(Y_vec, 1e-12) / (K_term * L_term)


def mpk_scalar(A0_j, K_j, L_j, alpha_j):
    """
    Marginal product of capital for country j.
    MPK = alpha * Y / K = alpha * A0 * K^(alpha-1) * L^(1-alpha)
    """
    return alpha_j * A0_j * max(K_j, 1e-12) ** (alpha_j - 1) * L_j ** (1 - alpha_j)


def equilibrium_capital_brent(A0_j, L_j, alpha_j, r_implied_j, delta_j,
                              K_lo=1e-3, K_hi=1e9):
    """
    Solve MPK(K*) = r_implied_j + delta_j via Brent's method.

    MPK is decreasing in K for Cobb-Douglas, so the residual
    f(K) = MPK(K) - target changes sign once.

    Returns K_prev if no bracket found (r_implied too large or negative).
    """
    target = r_implied_j + delta_j
    if target <= 0:
        return None   # caller uses K_prev as fallback

    f = lambda K: mpk_scalar(A0_j, K, L_j, alpha_j) - target

    if f(K_lo) <= 0:   # MPK too small even at K_lo → target too high
        return K_lo
    if f(K_hi) >= 0:   # MPK too large even at K_hi → target too low
        return K_hi

    return brentq(f, K_lo, K_hi, xtol=1e-9)


# ===========================================================================
# 4. EQUILIBRIUM
# ===========================================================================

def market_clearing(W_vec, w_mat):
    """
    V_j = sum_i w_{i,j} * W_i

    Rows of w_mat = investor, cols = asset.
    """
    return w_mat.T @ W_vec


def inner_equilibrium(K_vec, K_prev_vec, W_vec, w_mat, w_mat_prev,
                      omega_mat, A0_vec, L_vec, s_vec,
                      alpha_vec, delta_vec, sigma_vec,
                      params, max_iter=50, tol=1e-6):
    """
    Iterate between CPT portfolio choice and firm capital adjustment
    until MPK_j = r_implied_j + delta_j for all j.

    Per iteration:
        1. Market clearing        -> V_j
        2. Implied returns        -> r_implied_j = (D_hat_j + delta_K_j) / V_j
        3. Friction volatilities  -> sigma_tilde_{i,j} from Omega
        4. CPT portfolios         -> w* with inertia rho
        5. Firm capital (Brent)   -> K*_j
        6. Convergence check

    Parameters
    ----------
    K_vec      : (N,)   capital stocks entering period
    K_prev_vec : (N,)   capital stocks previous period (for delta_K)
    W_vec      : (N,)   household wealth
    w_mat      : (N,N)  portfolio weights warm start
    w_mat_prev : (N,N)  previous period weights (for inertia)
    omega_mat  : (N,N)  information levels this period
    A0_vec     : (N,)   structural TFP
    L_vec      : (N,)   labour
    s_vec      : (N,)   retention rates
    alpha_vec  : (N,)   capital shares
    delta_vec  : (N,)   depreciation rates
    sigma_vec  : (N,)   true return volatilities (per country j)
    params     : dict   phi_0, rho, cpt_tau, and CPT shape params
    """
    N     = len(K_vec)
    phi_0 = params["phi_0"]
    rho   = params["rho"]
    tau   = params.get("cpt_tau", 0.005)
    K_vec = K_vec.copy()
    w_mat = w_mat.copy()

    for iteration in range(max_iter):

        # ── Market clearing ────────────────────────────────────────────
        V_vec = market_clearing(W_vec, w_mat)
        V_vec = np.maximum(V_vec, 1e-12)

        # ── Expected dividends and implied returns ─────────────────────
        Y_hat    = A0_vec * K_vec ** alpha_vec * L_vec ** (1 - alpha_vec)
        D_hat    = (1 - s_vec) * alpha_vec * Y_hat
        delta_K  = K_vec - K_prev_vec
        r_impl   = (D_hat + delta_K) / V_vec

        # ── Friction-adjusted volatilities ─────────────────────────────
        sigma_tilde_mat = np.array([
            [sigma_tilde(sigma_vec[j], phi_0, omega_mat[i, j]) for j in range(N)]
            for i in range(N)
        ])

        # ── CPT portfolio per household ────────────────────────────────
        w_new = np.zeros((N, N))
        for i in range(N):
            w_new[i] = cpt_portfolio(
                r_impl, sigma_tilde_mat[i], w_mat_prev[i], rho, tau,
                alpha_v=params.get("alpha_gain", 0.88),
                beta_v=params.get("beta_loss", 0.88),
                lam=params.get("lambda_loss", 2.25),
                gamma_plus=params.get("gamma_plus", 0.61),
                delta_minus=params.get("delta_minus", 0.69),
            )

        # ── Firm capital via Brent ─────────────────────────────────────
        K_new = np.zeros(N)
        for j in range(N):
            K_j = equilibrium_capital_brent(
                A0_vec[j], L_vec[j], alpha_vec[j], r_impl[j], delta_vec[j]
            )
            K_new[j] = K_j if K_j is not None else K_vec[j]

        # ── Convergence ────────────────────────────────────────────────
        mpk_vec = np.array([mpk_scalar(A0_vec[j], K_new[j], L_vec[j], alpha_vec[j])
                            for j in range(N)])
        loss = np.sum((mpk_vec - (r_impl + delta_vec)) ** 2)

        w_mat = w_new
        K_vec = K_new

        if loss < tol:
            break

    V_vec = market_clearing(W_vec, w_mat)
    Q_vec = V_vec / np.maximum(K_vec, 1e-12)
    return K_vec, w_mat, V_vec, r_impl, Q_vec, iteration + 1


# ===========================================================================
# 5. REALISATION
# ===========================================================================

def realise_period(K_vec, K_prev_vec, V_vec, w_mat, W_vec,
                   A0_vec, L_vec, s_vec, alpha_vec, delta_vec,
                   sigma_A_frac, rng):
    """
    Draw TFP shocks, compute realised Y and D, update W and K.

    TFP shock: A_real = A0 + eps, eps ~ N(0, (sigma_A_frac * A0)^2)
    Wealth update (rate-of-return form):
        r_real_j = (D_j + K_j_new - K_j) / V_j
        W_new_i  = W_i * (1 + sum_j w_ij * r_real_j)
    """
    N = len(K_vec)

    sigma_A_vec = sigma_A_frac * A0_vec
    eps         = rng.normal(0, 1, size=N)
    A_real      = np.maximum(A0_vec + sigma_A_vec * eps, 1e-12)

    Y_vec = A_real * K_vec ** alpha_vec * L_vec ** (1 - alpha_vec)
    D_vec = (1 - s_vec) * alpha_vec * Y_vec

    K_new = (1 - delta_vec) * K_vec + s_vec * alpha_vec * Y_vec

    # Rate-of-return per country
    r_real = (D_vec + K_new - K_vec) / np.maximum(V_vec, 1e-12)

    # Wealth update
    W_new = np.array([
        W_vec[i] * (1 + np.dot(w_mat[i], r_real))
        for i in range(N)
    ])
    W_new = np.maximum(W_new, 1e-12)

    # Bilateral CPIS positions
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
    tqdm_disable: bool = False,
):
    """
    Run the full simulation over T periods.

    Parameters
    ----------
    df         : pd.DataFrame  bilateral panel with columns as per COLUMN_MAP
    countries  : list          ordered list of ISO3 country codes to simulate
    params     : dict          model parameters (see below)
    coef_map   : dict          gravity coefficients {column: beta}
    T          : int           number of periods
    seed       : int           random seed

    Required params keys
    --------------------
    phi_0          : ambiguity premium scaling
    rho            : portfolio inertia in [0,1]
    s_vec          : (N,) retention rates (or scalar applied to all)
    sigma_A_frac   : TFP shock std as fraction of A0
    cpt_tau        : softmax temperature (lower = sharper home bias)
    kappa          : dynamic Omega accumulation speed
    omega_max      : ceiling on dynamic Omega
    omega_bar      : inflection of logistic ceiling
    legal_star     : legal index threshold for Omega accumulation
    alpha_gain     : CPT gain curvature (default 0.88)
    beta_loss      : CPT loss curvature (default 0.88)
    lambda_loss    : loss aversion (default 2.25)
    gamma_plus     : probability weighting gains (default 0.61)
    delta_minus    : probability weighting losses (default 0.69)

    Wealth initialisation
    ---------------------
    Start with autarky (w_ii=1) so V_j = W_j = K_j → Q=1 at t=0.
    W_i is initialised from the diagonal of the panel (iso3_i == iso3_j).
    """
    rng        = np.random.default_rng(seed)
    N          = len(countries)
    sim_df     = df.copy()
    start_year = sim_df["year"].min()
    end_year   = start_year + T - 1

    # ── Compute per-country true return volatility from historical r_j ─────
    sigma_by_country = (
        sim_df[sim_df["iso3_i"] == sim_df["iso3_j"]]
        .groupby("iso3_i")["r_j"]
        .std()
        .reindex(countries)
        .fillna(0.02)
    )
    sigma_vec = sigma_by_country.values

    # ── Initialise simulation state columns ───────────────────────────────
    sim_df["K_sim"] = sim_df["K_j"]
    sim_df["W_sim"] = sim_df["K_j"]         # W = K → Q=1 at autarky start
    sim_df["FDI_sim"] = 0.0

    # Omega dynamic component (NxN), starts at zero
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
        alpha_vec = np.clip(1.0 - diag["lab_sh_j"].fillna(0.33).values, 0.1, 0.9)
        delta_vec = diag["delta_j"].fillna(0.05).values.astype(float)

        # Retention rates — use param if scalar, else use per-country
        s_param = params.get("s_vec", 0.40)
        s_vec   = np.full(N, s_param) if np.isscalar(s_param) else np.asarray(s_param)

        # Back-calculate A0 from PWT data
        A0_vec = compute_A0(Y_vec_obs, K_vec, L_vec, alpha_vec)

        # Previous period K (use K_sim for t>start, K_j for t=start)
        K_prev_vec = K_vec.copy()   # will be overwritten from results in t+1

        # ── Compute Omega matrix (structural + dynamic) ────────────────────
        omega_struct = compute_omega_mat(year_df, countries, coef_map)
        omega_mat    = omega_struct + omega_dyn_mat

        # ── Legal matrix for dynamic Omega update ─────────────────────────
        legal_star = params.get("legal_star", 50.0)
        legal_mat  = np.ones((N, N)) * legal_star   # default: at threshold
        for _, row in year_df.iterrows():
            ci = row.get("iso3_i")
            cj = row.get("iso3_j")
            if ci in countries and cj in countries:
                i = countries.index(ci)
                j = countries.index(cj)
                if pd.notna(row.get("Overall Score_d")):
                    legal_mat[i, j] = row["Overall Score_d"]

        # ── Inner equilibrium ──────────────────────────────────────────────
        K_star, w_mat, V_vec, r_impl, Q_vec, n_iter = inner_equilibrium(
            K_vec      = K_vec,
            K_prev_vec = K_prev_vec,
            W_vec      = W_vec,
            w_mat      = w_mat,
            w_mat_prev = w_mat_prev,
            omega_mat  = omega_mat,
            A0_vec     = A0_vec,
            L_vec      = L_vec,
            s_vec      = s_vec,
            alpha_vec  = alpha_vec,
            delta_vec  = delta_vec,
            sigma_vec  = sigma_vec,
            params     = params,
        )

        # ── Realisation ────────────────────────────────────────────────────
        Y_vec, D_vec, W_new, K_new, cpis_mat, A_real, r_real = realise_period(
            K_vec       = K_star,
            K_prev_vec  = K_prev_vec,
            V_vec       = V_vec,
            w_mat       = w_mat,
            W_vec       = W_vec,
            A0_vec      = A0_vec,
            L_vec       = L_vec,
            s_vec       = s_vec,
            alpha_vec   = alpha_vec,
            delta_vec   = delta_vec,
            sigma_A_frac= params.get("sigma_A_frac", 0.02),
            rng         = rng,
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
            "year"      : year,
            "countries" : countries,
            "K_star"    : K_star.copy(),
            "W_vec"     : W_vec.copy(),
            "W_new"     : W_new.copy(),
            "r_implied" : r_impl.copy(),
            "r_real"    : r_real.copy(),
            "Y"         : Y_vec.copy(),
            "D"         : D_vec.copy(),
            "A_real"    : A_real.copy(),
            "V_vec"     : V_vec.copy(),
            "Q_vec"     : Q_vec.copy(),
            "Wmat"      : w_mat.copy(),
            "cpis_mat"  : cpis_mat.copy(),
            "home_share": home_share.copy(),
            "n_iter"    : n_iter,
        })

        # ── Forward-fill state to t+1 ──────────────────────────────────────
        if year < end_year:
            nxt = sim_df["year"] == (year + 1)
            for idx, c in enumerate(countries):
                mask_i = nxt & (sim_df["iso3_i"] == c)
                mask_j = nxt & (sim_df["iso3_j"] == c)
                sim_df.loc[mask_i, "W_sim"] = W_new[idx]
                sim_df.loc[mask_j, "K_sim"] = K_new[idx]
                for jdx, cj in enumerate(countries):
                    mask_ij = nxt & (sim_df["iso3_i"] == c) & (sim_df["iso3_j"] == cj)
                    sim_df.loc[mask_ij, "FDI_sim"] = cpis_mat[idx, jdx]

        # Roll state
        w_mat_prev = w_mat.copy()

    return sim_df, results


# ===========================================================================
# 7. SUMMARY OUTPUT
# ===========================================================================

def print_summary(results, scale=1e6, scale_label="(tn)"):
    """Print simulation summary table."""
    first, last = results[0], results[-1]

    def pct(new, old):
        return (new - old) / abs(old) * 100 if abs(old) > 1e-12 else float("nan")

    rows = []
    for idx, c in enumerate(first["countries"]):
        rows.append({
            "Country"           : c,
            f"W t0 {scale_label}": first["W_vec"][idx] / scale,
            f"W t-1 {scale_label}":last["W_new"][idx]  / scale,
            "ΔW (%)"            : pct(last["W_new"][idx], first["W_vec"][idx]),
            f"K t0 {scale_label}": first["K_star"][idx] / scale,
            f"K t-1 {scale_label}":last["K_star"][idx]  / scale,
            "ΔK (%)"            : pct(last["K_star"][idx], first["K_star"][idx]),
            f"Y t0 {scale_label}": first["Y"][idx] / scale,
            f"Y t-1 {scale_label}":last["Y"][idx]  / scale,
            "ΔY (%)"            : pct(last["Y"][idx], first["Y"][idx]),
            f"D t0 {scale_label}": first["D"][idx] / scale,
            f"D t-1 {scale_label}":last["D"][idx]  / scale,
            "r_impl t0"         : round(first["r_implied"][idx], 4),
            "r_impl t-1"        : round(last["r_implied"][idx],  4),
            "Δr_impl"           : round(last["r_implied"][idx] - first["r_implied"][idx], 4),
            "Q t0"              : round(first["Q_vec"][idx], 3),
            "Q t-1"             : round(last["Q_vec"][idx],  3),
            "Home% t0"          : round(first["home_share"][idx] * 100, 1),
            "Home% t-1"         : round(last["home_share"][idx]  * 100, 1),
            "A_real t-1"        : round(last["A_real"][idx], 3),
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
    print("  CAPITAL MISALLOCATION  |  σ²(r_implied) per year")
    print(f"{'─'*60}")
    for r in results:
        rv     = r["r_implied"]
        spread = rv.max() - rv.min()
        vals   = ", ".join(f"{c}={v:.3f}" for c, v in zip(r["countries"], rv))
        print(f"  {r['year']}:  σ²={np.var(rv):.6f}  spread={spread:.4f}  [{vals}]")


# ===========================================================================
# 8. EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    from utils.v3.countries import aggregate_row
    
    df = pd.read_csv(
        "/Users/jesper/Desktop/CBS/Thesis 1/"
        "Jesper-Liedholm-Thesis-Code/Data/Clean/Final.csv"
    )

    countries = ["DEU", "FRA", "ITA", "ESP", "SWE"]
    row_countries = set(df.columns) - set(countries)
    df = aggregate_row(df, row_countries=row_countries)

    # Gravity coef_map — betas estimated from OLS on CPIS panel
    # Signs: legal+ (more open = more info), dist- (further = less info),
    #        lang+ (shared language = more info), cpis_lag+ (past flows = more info)
    # These are placeholder values — replace with OLS estimates
    coef_map = {
        "Overall Score_d": 0.04,   # legal openness (ICRG/Heritage)
        "ln_dist"        : -0.30,  # log bilateral distance
        "lang"           :  0.50,  # linguistic proximity
        "ln_cpis_lag1"   :  0.20,  # lagged log CPIS (information feedback)
    }

    params = {
        # Information frictions
        "phi_0"       : 0.05,     # ambiguity premium (calibrate via SMM)
        # Portfolio
        "rho"         : 0.70,     # inertia (calibrate via SMM)
        "cpt_tau"     : 0.003,    # softmax temperature
        "s_vec"       : 0.40,     # retention rate
        # CPT shape (fixed from Tversky & Kahneman 1992)
        "alpha_gain"  : 0.88,
        "beta_loss"   : 0.88,
        "lambda_loss" : 2.25,
        "gamma_plus"  : 0.61,
        "delta_minus" : 0.69,
        # TFP shock
        "sigma_A_frac": 0.02,
        # Dynamic Omega
        "kappa"       : 0.001,
        "omega_max"   : 10.0,
        "omega_bar"   : 5.0,
        "legal_star"  : 55.0,
    }

    print("Running simulation v4...")
    sim_df, results = run_simulation(
        df=df, countries=countries,
        params=params, coef_map=coef_map,
        T=5, seed=42,
    )
    print_summary(results)