"""
CMU Simulation Model with Endogenous TFP  –  v4_simulation_endo_tfp.py
=======================================================================
Extends v4_simulation.py by adding a productivity spillover channel:
when a country receives more foreign equity capital, knowledge and
technology transfer raise its TFP.

The extension slots in between Step 7 (capital reallocation) and
Step 8 (output effects):

  Step 7b  –  decompose capital into domestic / foreign components
  Step 8   –  output with endogenous TFP A_i(θ) = Ā_i·(1 + θ·f_i)

where f_i = foreign equity capital / PWT physical capital stock, and
θ ∈ {0.00, 0.05, 0.10} is the TFP-spillover elasticity.

References
----------
  Baltabaev (2014, Empirical Economics)      – θ ≈ 0.10 (panel cointegration)
  Borensztein, De Gregorio & Lee (1998, JIE) – conservative θ ≈ 0.05
  Bau & Matray (2023, Econometrica)          – capital liberalisation ↑ TFP 3–16%
  Javorcik (2004, AER)                       – micro evidence on FDI spillovers
  Findlay (1978, JIE)                        – technology-contagion theory

Convention
----------
  i  = origin  country (investor / source of capital)
  j  = destination country (receives capital)

Data source: Data/Clean/Final-v4.csv
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0.  CONFIGURATION
# ============================================================
DATA_PATH  = "Data/Clean/Final-v4.csv"
BASE_YEAR  = 2019
ETA        = 1.0          # return elasticity (baseline)
# ω (omega) = financial (hard) integration intensity  — policy lever on hard barriers
# γ (gamma) = linguistic (soft) integration intensity — policy lever on soft barriers
# Both independently take values from BARRIER_SCENARIOS → n×n combinations
BARRIER_SCENARIOS = [0.00, 0.25, 0.50, 1.00]
THETA_SCENARIOS = [0.00, 0.05, 0.10]   # TFP spillover elasticity

# For visualisation purposes, expand the ranges to smaller granularity
m = 20
BARRIER_SCENARIOS = np.round(np.linspace(0.0, 1.0, m), 3).tolist()
THETA_SCENARIOS = np.round(np.linspace(0.0, 0.10, m), 3).tolist()

EU27 = [
    "AUT","BEL","BGR","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","ITA","LVA","LTU","NLD","POL","PRT","ROU","SVK",
    "SVN","ESP","SWE",
    # "HRV", # missing data
]

EU27 += ["CYP","IRL","LUX"]
EU27 += ["MLT"]

OUTSIDE   = ["USA","GBR","CHE","NOR"]
COUNTRIES = EU27 + OUTSIDE          # n = 31
n         = len(COUNTRIES)
idx       = {c: i for i, c in enumerate(COUNTRIES)}

# Financial centres whose domestic holdings are distorted by
# pass-through flows; excluded from diagonal (Δ_ii) calibration.
FINANCIAL_CENTRES = {"IRL", "LUX", "CHE", "SVK", "CYP", "MLT"}


def nearest(scenarios, target):
    """Return the value in `scenarios` closest to `target`.
    Used so that sense-checks, summary tables and exports work
    regardless of the exact grid spacing of BARRIER_SCENARIOS /
    THETA_SCENARIOS.
    """
    return min(scenarios, key=lambda x: abs(x - target))


# Convenience: canonical scenario anchors resolved against actual grids
_gamma0    = nearest(BARRIER_SCENARIOS, 0.00)   # ω or γ = 0  (no integration)
_gamma_mid = nearest(BARRIER_SCENARIOS, 0.50)   # ω or γ ≈ 0.5
_gamma1    = nearest(BARRIER_SCENARIOS, 1.00)   # ω or γ = 1  (full integration)
_theta0  = nearest(THETA_SCENARIOS,   0.00)
_theta_mid = nearest(THETA_SCENARIOS, 0.05)
_theta1  = nearest(THETA_SCENARIOS,   0.10)


# ============================================================
# 1.  LOAD & CLEAN DATA
# ============================================================
print("=" * 60)
print("STEP 0 — Loading data")
print("=" * 60)

df_full = pd.read_csv(DATA_PATH)
print(f"  Raw rows (world, all pairs): {len(df_full):,}  |  columns: {df_full.shape[1]}")
print(f"  Unique source countries: {df_full['iso3_i'].nunique()}")
print(f"  Unique destination countries: {df_full['iso3_j'].nunique()}")

print(f"  Mean values after scaling:")
print(df_full[["Y_i", "Y_j", "a_ij"]].agg(["mean", "min", "max"]))

# --- Keep full world data for PPML; also filter to COUNTRIES for simulation ---
df = df_full[df_full["iso3_i"].isin(COUNTRIES) & df_full["iso3_j"].isin(COUNTRIES)].copy()
print(f"  After country filter (simulation): {len(df):,} rows")
print(f"  Full world data for PPML: {len(df_full):,} rows total")

# --- Floor negative a_ij to 0 (data artefact) --------------
neg_mask = df["a_ij"] < 0
if neg_mask.sum() > 0:
    print(f"  WARNING: {neg_mask.sum()} negative a_ij values → set to 0")
    df.loc[neg_mask, "a_ij"] = 0.0

# --- Build derived columns ---------------------------------
df["ln_d_geo"]    = np.log(df["d_geo"].replace(0, np.nan))
df["euro_ij"]     = df["euro_i"] * df["euro_j"]
df["common_lang"] = (df["d_ling"] < 0.2).astype(float)

df["d_gci_overall"] = np.abs(df["gci_overall_i"] - df["gci_overall_j"])
df["d_gci_corruption"] = np.abs(df["gci_corruption_i"] - df["gci_corruption_j"])
df["d_gci_trade_openness"] = np.abs(df["gci_trade_openness_i"] - df["gci_trade_openness_j"])
df["d_gci_financial_composite"] = np.abs(df["gci_financial_composite_i"] - df["gci_financial_composite_j"])


# ============================================================
# 2.  BUILD BASE-YEAR SNAPSHOT
# ============================================================
print()
print("=" * 60)
print("STEP 0b — Assembling base-year cross-section")
print("=" * 60)

base = df[(df["year"] == BASE_YEAR)].copy()

# Calculate home holdings: domestic portfolio = total_portfolio - total_pip
# Both are aggregate (non-bilateral) totals per country.
# domestic_share_i = 1 - (total_pip_million_usd_i / total_portfolio_million_usd_i)
country_totals = (
    base.drop_duplicates("iso3_i")
    .set_index("iso3_i")[["total_portfolio_million_usd_i", "total_pip_million_usd_i"]]
)

home_hold = {}
for c in COUNTRIES:
    if c not in country_totals.index:
        home_hold[c] = np.nan
        continue
    tot_port = country_totals.at[c, "total_portfolio_million_usd_i"]
    tot_pip  = country_totals.at[c, "total_pip_million_usd_i"]
    if pd.isna(tot_port) or pd.isna(tot_pip) or tot_port <= 0:
        home_hold[c] = np.nan
    else:
        h = tot_port - tot_pip
        if h < 0.01 * tot_port:
            print(f"  NOTE: {c} home holdings floor applied (port={tot_port:.1f}, pip={tot_pip:.1f})")
            h = 0.01 * tot_port
        home_hold[c] = h

# Set diagonal of a_ij to home holdings
for c in COUNTRIES:
    mask = (df["year"] == BASE_YEAR) & (df["iso3_i"] == c) & (df["iso3_j"] == c)
    if mask.sum() > 0 and not np.isnan(home_hold.get(c, np.nan)):
        df.loc[mask, "a_ij"] = home_hold[c]

# Filter to COUNTRIES only
base = df[(df["year"] == BASE_YEAR)].copy()
base = base[base["iso3_i"].isin(COUNTRIES) & base["iso3_j"].isin(COUNTRIES)].copy()

diag_check = base[base["iso3_i"] == base["iso3_j"]][["iso3_i","a_ij","total_portfolio_million_usd_i","total_pip_million_usd_i"]].set_index("iso3_i")
print(f"  Base year rows (after COUNTRIES filter): {len(base)}")
print("  Diagonal a_ii (domestic holdings = total_portfolio - total_pip) — first 10:")
print(diag_check.head(10).round(1).to_string())
print(f"  Diagonal NaN count: {diag_check['a_ij'].isna().sum()}")

# ============================================================
# 3.  GRAVITY REGRESSION  (Step 1) — ON WORLD DATA
# ============================================================
print()
print("=" * 60)
print("STEP 1 — Gravity regression (PPML) — World sample")
print("=" * 60)

# Use full world data for PPML
ppml_df = df.copy()
ppml_df = ppml_df[ppml_df["iso3_i"] != ppml_df["iso3_j"]].copy()
print(f"  After removing i=j: {len(ppml_df):,} rows")
# Only COUNTRIES
unique_iso_codes = set(ppml_df["iso3_i"]).union(set(ppml_df["iso3_j"]))
assert not any(c not in COUNTRIES for c in unique_iso_codes), "PPML data contains countries not in COUNTRIES list"

cols = ["a_ij", "d_ling", "d_gci_financial_composite", "d_num_articles", "year", "iso3_i", "iso3_j", "Y_i", "Y_j"]

# PPML (Poisson MLE) requires non-negative a_ij — zeros are valid and must be KEPT.
# Dropping zero-flow pairs would truncate the sample and cause systematic overprediction
# (the zero observations pull fitted values down via the -exp(xβ) term in the log-likelihood).
ppml_df = ppml_df.copy()   # ensure we own the DataFrame before modifying (avoid SettingWithCopyWarning)
n_neg = (ppml_df["a_ij"] < 0).sum()
if n_neg > 0:
    print(f"  NOTE: {n_neg} negative a_ij values floored to 0 (data artefact)")
ppml_df["a_ij"] = ppml_df["a_ij"].clip(lower=0)

# Create bilateral pair identifier for clustering
ppml_df['pair'] = ppml_df['iso3_i'] + '_' + ppml_df['iso3_j']

# Drop rows only where covariates are missing; retain zero-flow rows
ppml_df = ppml_df.dropna(subset=cols).copy()
n_zeros = (ppml_df["a_ij"] == 0).sum()
n_pos   = (ppml_df["a_ij"]  > 0).sum()
print(f"  PPML sample: {len(ppml_df):,} obs  |  positive flows: {n_pos:,}  |  zero flows: {n_zeros:,}")

# --- Scale monetary columns to prevent overflow in Y_i*Y_j interaction ------
# a_ij: USD millions → USD trillions  |  Y_i, Y_j: same
# Slope coefficients for friction regressors (d_ling, d_gci_fin) are
# invariant to scaling of a_ij and Y — no rescaling needed for them.
# Y-related coefficients are rescaled back below for interpretability.
PPML_SCALE_A = 1e6   # a_ij divisor
PPML_SCALE_Y = 1e6   # Y_i, Y_j divisor
ppml_df = ppml_df.copy()
ppml_df["a_ij"] = ppml_df["a_ij"] / PPML_SCALE_A
ppml_df["Y_i"]  = ppml_df["Y_i"]  / PPML_SCALE_Y
ppml_df["Y_j"]  = ppml_df["Y_j"]  / PPML_SCALE_Y
print(f"  PPML scaling: a_ij ÷ {PPML_SCALE_A:.0e}  |  Y_i, Y_j ÷ {PPML_SCALE_Y:.0e}")

print(ppml_df[cols].describe())

# Fit PPML with gravity specification
formula = 'a_ij ~ d_ling + d_num_articles + d_gci_financial_composite + Y_i + Y_j + C(year)'

ppml = smf.poisson(
    formula=formula,
    data=ppml_df,
).fit(
    method="newton",
    maxiter=300,
    cov_type="cluster",
    cov_kwds={"groups": ppml_df['pair']},
)

print()
print(ppml.summary())

# --- Rescale Y-related coefficients back to original units ------------------
# β_Yi_orig   = β_Yi_scaled   × SCALE_Y
# β_Yj_orig   = β_Yj_scaled   × SCALE_Y
# β_YiYj_orig = β_YiYj_scaled × SCALE_Y²
# Intercept shifts by log(SCALE_A) — not used downstream.
# Friction coefficients (d_ling, d_gci_fin): UNCHANGED — scale-invariant. ✓
_rescale = {"Y_i": PPML_SCALE_Y, "Y_j": PPML_SCALE_Y, "Y_i:Y_j": PPML_SCALE_Y ** 2}
ppml_params_rescaled = ppml.params.copy()
for _k, _s in _rescale.items():
    if _k in ppml_params_rescaled:
        ppml_params_rescaled[_k] *= _s
print("\n  Y-related coefficients rescaled to original units:")
for _k in _rescale:
    if _k in ppml_params_rescaled:
        print(f"    {_k}: scaled={ppml.params[_k]:+.6e}  →  original={ppml_params_rescaled[_k]:+.6e}")

# Extract gravity coefficients
beta_dict = {}

beta_d_ling    = ppml.params.get("d_ling", 0.0)
beta_d_gci_fin = ppml.params.get("d_gci_financial_composite", 0.0)
beta_numart    = ppml.params.get("d_num_articles", 0.0)

print()
print("  Gravity coefficients (friction regressors — scale-invariant):")
print(f"    β_d_ling                   : {beta_d_ling:+.4f}   p={ppml.pvalues.get('d_ling', np.nan):.3f}")
print(f"    β_d_gci_financial_composite: {beta_d_gci_fin:+.4f}   p={ppml.pvalues.get('d_gci_financial_composite', np.nan):.3f}")
print(f"    β_d_num_articles           : {beta_numart:+.4f}   p={ppml.pvalues.get('d_num_articles', np.nan):.3f}")
print()

# Store friction coefficients (negative sign indicates friction)
if beta_d_ling < 0:
    beta_dict["d_ling"] = beta_d_ling
    print("  OK: β_d_ling < 0 → language distance is friction.")
else:
    print("  WARNING: β_d_ling ≥ 0 — unexpected sign.")

if beta_d_gci_fin < 0:
    beta_dict["d_gci_financial_composite"] = beta_d_gci_fin
    print("  OK: β_d_gci_financial_composite < 0 → financial system distance is friction.")
else:
    print("  WARNING: β_d_gci_financial_composite ≥ 0 — unexpected sign.")

# d_num_articles is a friction (β < 0): higher distance in news coverage → less investment
if beta_numart < 0:
    beta_dict["d_num_articles"] = beta_numart
    print("  OK: β_d_num_articles < 0 → news article distance is a friction (soft).")
else:
    print("  WARNING: β_d_num_articles ≥ 0 — unexpected sign for friction.")

print(f"\n  Active friction regressors: {list(beta_dict.keys())}")

# ============================================================
# 4.  BILATERAL WEDGES  (Step 2)
#     Hard barriers  = financial system distance   (policy lever: ω)
#     Soft barriers γ = news article distance        (policy lever: γ)
#     Fixed soft      = linguistic distance          (no policy lever)
# ============================================================
print()
print("=" * 60)
print("STEP 2 — Building bilateral wedge matrices")
print("=" * 60)

base = df[(df["year"] == BASE_YEAR) & df["iso3_i"].isin(COUNTRIES) & df["iso3_j"].isin(COUNTRIES)].copy()

# GCI data only covers up to ~2019; carry forward the last observed value per country
# so that the base-year (2023) cross-section has non-NaN GCI distances.
gci_last = (
    df[df["gci_financial_composite_i"].notna()]
    .sort_values("year")
    .drop_duplicates("iso3_i", keep="last")
    .set_index("iso3_i")["gci_financial_composite_i"]
)
print(f"  Last observed GCI financial composite (n={len(gci_last)} countries):")
print(f"    range: [{gci_last.min():.2f}, {gci_last.max():.2f}]  mean={gci_last.mean():.2f}")

base["gci_financial_composite_i"] = base["iso3_i"].map(gci_last)
base["gci_financial_composite_j"] = base["iso3_j"].map(gci_last)

# Compute financial composite distance on base cross-section
base["d_gci_financial_composite"] = np.abs(
    base["gci_financial_composite_i"] - base["gci_financial_composite_j"]
)
nan_fin = base["d_gci_financial_composite"].isna().sum()
print(f"  d_gci_financial_composite NaN count after carry-forward: {nan_fin}")

Delta_hard   = np.ones((n, n))   # financial system distance (hard / policy lever: ω)
Delta_ling   = np.ones((n, n))   # linguistic distance       (fixed — no policy lever)
Delta_numart = np.ones((n, n))   # news article distance     (soft / policy lever: γ)

for _, row in base.iterrows():
    i_iso, j_iso = row["iso3_i"], row["iso3_j"]
    if i_iso == j_iso:
        continue
    ii, jj = idx[i_iso], idx[j_iso]

    # Hard barrier: financial system distance (reduced by ω)
    ln_hard = 0.0
    if "d_gci_financial_composite" in beta_dict and pd.notna(row.get("d_gci_financial_composite")):
        ln_hard -= beta_dict["d_gci_financial_composite"] * row["d_gci_financial_composite"]
    Delta_hard[ii, jj] = np.exp(ln_hard)

    # Fixed soft barrier: linguistic distance (NOT reduced by γ)
    ln_ling = 0.0
    if "d_ling" in beta_dict and pd.notna(row.get("d_ling")):
        ln_ling -= beta_dict["d_ling"] * row["d_ling"]
    Delta_ling[ii, jj] = np.exp(ln_ling)

    # Policy soft barrier: news article distance (reduced by γ)
    ln_numart = 0.0
    if "d_num_articles" in beta_dict and pd.notna(row.get("d_num_articles")):
        ln_numart -= beta_dict["d_num_articles"] * row["d_num_articles"]
    Delta_numart[ii, jj] = np.exp(ln_numart)

# Combined soft barrier (for baseline calibration)
Delta_soft = Delta_ling * Delta_numart

# Combined baseline wedge (diagonal still = 1, calibrated in Step 2.2)
Delta_arr = Delta_hard * Delta_soft

_offdiag = ~np.eye(n, dtype=bool)
print(f"  Hard (fin) Δ range (off-diag):            [{Delta_hard[_offdiag].min():.3f}, {Delta_hard[_offdiag].max():.3f}]")
print(f"  Ling (fixed) Δ range (off-diag):          [{Delta_ling[_offdiag].min():.3f}, {Delta_ling[_offdiag].max():.3f}]")
print(f"  NumArt (γ-lever) Δ range (off-diag):      [{Delta_numart[_offdiag].min():.3f}, {Delta_numart[_offdiag].max():.3f}]")
print(f"  Combined soft Δ range (off-diag):         [{Delta_soft[_offdiag].min():.3f}, {Delta_soft[_offdiag].max():.3f}]")
print(f"  Combined Δ range (off-diag):              [{Delta_arr[_offdiag].min():.3f}, {Delta_arr[_offdiag].max():.3f}]")
print(f"  Mean combined off-diagonal Δ:             {Delta_arr[_offdiag].mean():.3f}")


# ============================================================
# 5.  COUNTRY-LEVEL MACRO VARIABLES  (Step 3)
# ============================================================
print()
print("=" * 60)
print("STEP 3 — Country-level returns R_j and GDP weights Y_j")
print("=" * 60)

ctry_base = base.drop_duplicates("iso3_i").set_index("iso3_i").reindex(COUNTRIES)

R_vec = (ctry_base["alpha_i"] * ctry_base["Y_i"] / ctry_base["k_i"]).values.astype(float)
# Use GDP (Y_i) as the portfolio-size weight — available for all countries
# without requiring market-cap predictions.
Y_vec = ctry_base["Y_i"].values.astype(float)

# PWT physical capital stock (used as denominator in foreign-capital intensity)
k_PWT = ctry_base["k_i"].values.astype(float)

missing_R    = [c for c, v in zip(COUNTRIES, R_vec)   if np.isnan(v)]
missing_M    = [c for c, v in zip(COUNTRIES, Y_vec)   if np.isnan(v) or v <= 0]
missing_kPWT = [c for c, v in zip(COUNTRIES, k_PWT)   if np.isnan(v) or v <= 0]

if missing_R:    print(f"  MISSING R    (alpha*Y/k): {missing_R}")
if missing_M:    print(f"  MISSING GDP  (Y_i): {missing_M}")
if missing_kPWT: print(f"  MISSING k_PWT (PWT capital): {missing_kPWT}")

assert not missing_R,    f"Missing returns R for: {missing_R}"
assert not missing_M,    f"Missing GDP Y for: {missing_M}"
assert not missing_kPWT, f"Missing PWT capital k for: {missing_kPWT}"

R_vec = R_vec / R_vec.mean()

print(f"  R range (normalised): [{R_vec.min():.3f}, {R_vec.max():.3f}]  mean={R_vec.mean():.3f}")
print(f"  GDP (Y_vec) range:    [{Y_vec.min():.1f}, {Y_vec.max():.1f}]  median={np.median(Y_vec):.1f}")
print(f"  k_PWT range:          [{k_PWT.min():.1f}, {k_PWT.max():.1f}]  median={np.median(k_PWT):.1f}")

R_series = pd.Series(R_vec, index=COUNTRIES)
print("\n  Top-5 returns (R_j, normalised):")
print(R_series.sort_values(ascending=False).head(5).round(3).to_string())


# ============================================================
# 6.  OBSERVED PORTFOLIO SHARES  (Step 0.3 / Step 4 prep)
# ============================================================
print()
print("=" * 60)
print("STEP 0.3 — Observed portfolio shares")
print("=" * 60)

A_arr = base.pivot(index="iso3_i", columns="iso3_j", values="a_ij").reindex(
    index=COUNTRIES, columns=COUNTRIES
)

_total  = A_arr.size
_nan    = A_arr.isna().sum().sum()
_zero   = (A_arr == 0).sum().sum()
_pos    = (A_arr > 0).sum().sum()
print(f"  A_arr sparsity: {_total} cells — "
      f"{_pos} positive ({_pos/_total:.1%}), "
      f"{_zero} zero ({_zero/_total:.1%}), "
      f"{_nan} NaN ({_nan/_total:.1%})")

s_vec = A_arr.sum(axis=1).values.astype(float)

row_sums = A_arr.sum(axis=1).values
row_sums_safe = np.where(row_sums == 0, np.nan, row_sums)
Pi_data_arr = A_arr.values / row_sums_safe[:, np.newaxis]
Pi_data = pd.DataFrame(Pi_data_arr, index=COUNTRIES, columns=COUNTRIES)

pi_home = np.diag(Pi_data_arr)
print("  Observed home bias π_ii (top 10):")
print(pd.Series(pi_home, index=COUNTRIES).sort_values(ascending=False).head(10).round(4).to_string())
print()
print("  Countries with NaN home bias (s_i = 0 or diagonal missing):")
nan_home = [c for c, v in zip(COUNTRIES, pi_home) if np.isnan(v)]
print(f"    {nan_home}")


# ============================================================
# 7.  CALIBRATE DIAGONAL WEDGE Δ_ii  (Step 2.2)
# ============================================================
print()
print("=" * 60)
print("STEP 2.2 — Calibrating diagonal wedges Δ_ii to match home bias")
print("=" * 60)

def compute_portfolio(Delta, R, M, eta):
    """
    Compute n×n portfolio-share matrix.
    π_ij = (R_j^η * M_j / Δ_ij) / Σ_ι (R_ι^η * M_ι / Δ_iι)
    Rows = origin i, columns = destination j.
    """
    RM  = (R ** eta) * M
    W   = RM[np.newaxis, :] / Delta
    Z   = W.sum(axis=1, keepdims=True)
    Z   = np.where(Z == 0, np.nan, Z)
    Pi  = W / Z
    return Pi


eta = ETA

for i, c in enumerate(COUNTRIES):
    if c in FINANCIAL_CENTRES:
        Delta_arr[i, i] = 1.0
    elif not np.isnan(pi_home[i]) and 0 < pi_home[i] < 0.9999:
        Delta_arr[i, i] = 0.01
    else:
        Delta_arr[i, i] = 1.0

RM = (R_vec ** eta) * Y_vec

MAX_ITER = 20_000
TOL      = 1e-5

for iteration in range(MAX_ITER):
    max_err = 0.0
    for i in range(n):
        c = COUNTRIES[i]
        if c in FINANCIAL_CENTRES:
            continue
        if np.isnan(pi_home[i]) or pi_home[i] <= 0 or pi_home[i] >= 0.9999:
            continue
        w      = RM / Delta_arr[i, :]
        Z_i    = w.sum()
        pi_hat = w[i] / Z_i
        Delta_arr[i, i] *= pi_hat / pi_home[i]
        max_err = max(max_err, abs(pi_hat - pi_home[i]))
    if max_err < TOL:
        print(f"  Converged in {iteration+1} iterations  (max |π̂_ii - π_ii| = {max_err:.2e})")
        break
else:
    print(f"  Did NOT converge in {MAX_ITER} iterations  (max err = {max_err:.6f})")

Delta_baseline = Delta_arr.copy()

calib_idx  = [i for i in range(n)
              if COUNTRIES[i] not in FINANCIAL_CENTRES
              and not np.isnan(pi_home[i])
              and 0 < pi_home[i] < 0.9999]
calib_ctry = [COUNTRIES[i] for i in calib_idx]
delta_diag = pd.Series(np.diag(Delta_baseline), index=COUNTRIES)
print("\n  Calibrated Δ_ii (domestic wedges) — calibratable countries:")
print(delta_diag[calib_ctry].sort_values(ascending=False).round(4).to_string())

skipped = [c for c in COUNTRIES if c not in calib_ctry]
print(f"\n  Skipped / set neutral (Δ_ii=1): {skipped}")

neg_diag = delta_diag[calib_ctry][delta_diag[calib_ctry] > 1]
pos_diag = delta_diag[calib_ctry][delta_diag[calib_ctry] <= 1]
print(f"\n  Countries with Δ_ii < 1 (home-biased, as expected): {len(pos_diag)}")
print(f"  Countries with Δ_ii > 1 (foreign-biased portfolio):  {len(neg_diag)}")
if len(neg_diag):
    print(f"    {neg_diag.round(4).to_string()}")


# ============================================================
# 8.  BASELINE PORTFOLIO  (Step 4)
# ============================================================
print()
print("=" * 60)
print("STEP 4 — Baseline portfolio shares")
print("=" * 60)

Pi_baseline = compute_portfolio(Delta_baseline, R_vec, Y_vec, eta)

pi_base_home = np.diag(Pi_baseline)
comparison = pd.DataFrame({
    "π_data":     pi_home,
    "π_baseline": pi_base_home,
    "diff":       pi_base_home - pi_home,
}, index=COUNTRIES).dropna(subset=["π_data"])
print("  Home bias: data vs. baseline (calibrated countries):")
print(comparison.sort_values("π_data", ascending=False).round(4).to_string())

mask_off  = ~np.eye(n, dtype=bool)
pd_flat   = Pi_data.values[mask_off]
pb_flat   = Pi_baseline[mask_off]
finite    = np.isfinite(pd_flat) & np.isfinite(pb_flat)
corr_off  = np.corrcoef(pd_flat[finite], pb_flat[finite])[0, 1]
print(f"\n  Off-diagonal correlation (data vs baseline): {corr_off:.4f}")
if corr_off < 0.5:
    print("  WARNING: low off-diagonal correlation — model fit is poor.")


# ============================================================
# 9.  CMU SHOCK  (Steps 5–6)
#   ω ∈ BARRIER_SCENARIOS — financial (hard) integration intensity
#   γ ∈ BARRIER_SCENARIOS — language  (soft) integration intensity
#   All combinations computed independently
# ============================================================
print()
print("=" * 60)
print("STEPS 5-6 — CMU shock (all ω×γ combinations) and counterfactual portfolios")
print("="  * 60)

eu_flag  = np.array([1 if c in EU27 else 0 for c in COUNTRIES])
eu_pair  = np.outer(eu_flag, eu_flag) * (1 - np.eye(n))
eu_pair_bool = eu_pair.astype(bool)

results = {}
_barrier_pairs = [(o, g) for o in BARRIER_SCENARIOS for g in BARRIER_SCENARIOS]
for omega, gamma in tqdm(_barrier_pairs, desc="CMU portfolios (ω×γ)", unit="scenario", position=0, leave=True):
    hard_cmu   = np.where(eu_pair_bool, Delta_hard   ** (1 - omega), Delta_hard)
    numart_cmu = np.where(eu_pair_bool, Delta_numart ** (1 - gamma), Delta_numart)
    # Linguistic barrier is fixed — not reduced by γ
    Delta_cmu = hard_cmu * Delta_ling * numart_cmu
    np.fill_diagonal(Delta_cmu, np.diag(Delta_baseline))
    Pi_cmu = compute_portfolio(Delta_cmu, R_vec, Y_vec, eta)
    results[(omega, gamma)] = {"Delta_cmu": Delta_cmu, "Pi_cmu": Pi_cmu}

print(f"  ω (fin / hard) scenarios: {BARRIER_SCENARIOS}")
print(f"  γ (numart / soft) scenarios: {BARRIER_SCENARIOS}")
print(f"  Total: {len(results)} scenarios (all ω×γ combinations)")

eu_idx_list = [idx[c] for c in EU27]
print("\n  EU avg home-bias change — selected (ω, γ) pairs:")
for omega, gamma in [(o, g) for o in BARRIER_SCENARIOS for g in [_gamma0, BARRIER_SCENARIOS[-1]]]:
    pi_cmu = np.diag(results[(omega, gamma)]["Pi_cmu"])
    eu_hb  = np.nanmean(pi_cmu[eu_idx_list]) - np.nanmean(np.diag(Pi_baseline)[eu_idx_list])
    print(f"    ω={omega:.2f}, γ={gamma:.2f}: Δπ_EU = {eu_hb:+.5f}")


# ============================================================
# 10.  CAPITAL REALLOCATION  (Step 7)
# ============================================================
print()
print("=" * 60)
print("STEP 7 — Capital reallocation")
print("=" * 60)

k_baseline = Pi_baseline.T @ s_vec

for key in results:
    Pi_cmu = results[key]["Pi_cmu"]
    k_cmu  = Pi_cmu.T @ s_vec
    results[key]["k_cmu"] = k_cmu

print("  Capital conservation check (all scenarios):")
bad = [(k, results[k]["k_cmu"].sum()/k_baseline.sum()) for k in results
       if abs(results[k]["k_cmu"].sum()/k_baseline.sum()-1) >= 1e-6]
if bad:
    for k, r in bad:
        print(f"    ω={k[0]:.2f}, γ={k[1]:.2f}: {r:.8f}  WARNING")
else:
    print("  All OK ✓")

print("\n  EU avg Δk/k (%) — selected (ω, γ) pairs:")
for omega, gamma in [(o, g) for o in BARRIER_SCENARIOS for g in [_gamma0, BARRIER_SCENARIOS[-1]]]:
    k_cmu = results[(omega, gamma)]["k_cmu"]
    eu_dk = np.mean((k_cmu - k_baseline)[eu_idx_list] / k_baseline[eu_idx_list]) * 100
    print(f"    ω={omega:.2f}, γ={gamma:.2f}: {eu_dk:+.4f}%")


# ============================================================
# 11.  STEP 7b — FOREIGN CAPITAL DECOMPOSITION (NEW)
# ============================================================
print()
print("=" * 60)
print("STEP 7b — Foreign capital decomposition")
print("=" * 60)

# Foreign capital received by country i under baseline:
#   k_i^foreign = k_i^total - π_ii^baseline * s_i
#   (total capital into i minus the part that originates from i itself)
k_foreign_baseline = k_baseline - np.diag(Pi_baseline) * s_vec

# Floor at zero (financial centres can have near-zero domestic holdings)
k_foreign_baseline = np.maximum(k_foreign_baseline, 0.0)

# Foreign-capital intensity: foreign equity / PWT physical capital stock
# k_PWT is the correct denominator because TFP spillovers operate on the
# physical production process, not just the equity portfolio.
f_baseline = k_foreign_baseline / k_PWT

for key in results:
    Pi_cmu = results[key]["Pi_cmu"]
    k_cmu  = results[key]["k_cmu"]
    k_foreign_cmu = k_cmu - np.diag(Pi_cmu) * s_vec
    k_foreign_cmu = np.maximum(k_foreign_cmu, 0.0)
    f_cmu = k_foreign_cmu / k_PWT
    results[key]["k_foreign_cmu"] = k_foreign_cmu
    results[key]["f_cmu"]         = f_cmu

print("  Foreign capital (USD mn) — baseline (first 10 countries):")
fk_series = pd.Series(k_foreign_baseline, index=COUNTRIES)
print(fk_series.head(10).round(1).to_string())

print("\n  Foreign capital intensity f_i = k_foreign / k_PWT — baseline (first 10):")
f_series = pd.Series(f_baseline, index=COUNTRIES)
print(f_series.head(10).round(6).to_string())

print(f"\n  f_baseline range: [{f_baseline.min():.6f}, {f_baseline.max():.6f}]")

print(f"\n  Δf_i (f_CMU - f_baseline) — ω={_gamma1}, γ={_gamma0} (fin only), EU27:")
delta_f = results[(_gamma1, _gamma0)]["f_cmu"] - f_baseline
print(pd.Series(delta_f, index=COUNTRIES)[EU27].round(6).to_string())
print(f"\n  Δf_i (f_CMU - f_baseline) — ω={_gamma1}, γ={_gamma1} (fin+ling), EU27:")
delta_f = results[(_gamma1, _gamma1)]["f_cmu"] - f_baseline
print(pd.Series(delta_f, index=COUNTRIES)[EU27].round(6).to_string())


# ============================================================
# 12.  HELPER FUNCTIONS
# ============================================================

def compute_tfp(A_bar, f, theta):
    """
    Endogenous TFP: A_i = Ā_i * (1 + θ * f_i)
    When θ=0 this reduces to Ā_i exactly (fixed-TFP baseline).
    """
    return A_bar * (1.0 + theta * f)


def cobb_douglas(A, k, L, alpha):
    """y_j = A_j * k_j^α_j * L_j^(1-α_j)"""
    return A * (k ** alpha) * (L ** (1 - alpha))


# ============================================================
# 13.  PRODUCTION PARAMETERS
# ============================================================
A_bar  = ctry_base["A_i"].values.astype(float)
L_prod = ctry_base["L_i"].values.astype(float)
alp    = ctry_base["alpha_i"].values.astype(float)

missing_A   = [c for c, v in zip(COUNTRIES, A_bar)  if np.isnan(v)]
missing_L   = [c for c, v in zip(COUNTRIES, L_prod) if np.isnan(v)]
missing_alp = [c for c, v in zip(COUNTRIES, alp)    if np.isnan(v)]

if missing_A:   print(f"  MISSING A_bar (TFP): {missing_A}")
if missing_L:   print(f"  MISSING L    (labour): {missing_L}")
if missing_alp: print(f"  MISSING alpha (capital share): {missing_alp}")

assert not missing_A,   f"Missing TFP A for: {missing_A}"
assert not missing_L,   f"Missing labour L for: {missing_L}"
assert not missing_alp, f"Missing alpha for: {missing_alp}"


# ============================================================
# 14.  STEP 8 (MODIFIED) — OUTPUT WITH ENDOGENOUS TFP
# ============================================================
print()
print("=" * 60)
print("STEP 8 — Output and productivity effects (endogenous TFP)")
print("=" * 60)
print(f"  θ scenarios: {THETA_SCENARIOS}")
print(f"  ω (fin) scenarios: {BARRIER_SCENARIOS}")
print(f"  γ (ling) scenarios: {BARRIER_SCENARIOS}")
print(f"  Total output scenarios: {len(THETA_SCENARIOS) * len(BARRIER_SCENARIOS)**2}")
print()

eu_idx = [idx[c] for c in EU27 if c in idx]

# Store all (theta, omega, gamma) results
endo_results = {}   # key: (theta, omega, gamma)

for theta in tqdm(THETA_SCENARIOS, desc="Step 8: output scenarios (θ)", unit="θ", position=0, leave=True):
    A_baseline_theta = compute_tfp(A_bar, f_baseline, theta)
    y_baseline_theta = cobb_douglas(A_baseline_theta, k_baseline, L_prod, alp)
    Y_EU_base_theta  = y_baseline_theta[eu_idx].sum()

    for omega in BARRIER_SCENARIOS:
        for gamma in BARRIER_SCENARIOS:
            f_cmu        = results[(omega, gamma)]["f_cmu"]
            k_cmu        = results[(omega, gamma)]["k_cmu"]
            A_cmu_theta  = compute_tfp(A_bar, f_cmu, theta)
            y_cmu_theta  = cobb_douglas(A_cmu_theta, k_cmu, L_prod, alp)
            Y_EU_cmu_theta = y_cmu_theta[eu_idx].sum()

            total_effect   = (y_cmu_theta - y_baseline_theta) / y_baseline_theta
            capital_effect = alp * (k_cmu - k_baseline) / k_baseline
            tfp_effect     = (A_cmu_theta - A_baseline_theta) / A_baseline_theta

            mpk_baseline = alp * y_baseline_theta / k_baseline
            mpk_cmu      = alp * y_cmu_theta / k_cmu

            eu_hb_change = (np.nanmean(np.diag(results[(omega, gamma)]["Pi_cmu"])[eu_idx])
                            - np.nanmean(np.diag(Pi_baseline)[eu_idx]))

            sigma_mpk_base = pd.Series(mpk_baseline, index=COUNTRIES).loc[EU27].std()
            sigma_mpk_cmu  = pd.Series(mpk_cmu, index=COUNTRIES).loc[EU27].std()
            sigma_reduction = (sigma_mpk_base - sigma_mpk_cmu) / sigma_mpk_base * 100

            endo_results[(theta, omega, gamma)] = {
                "theta": theta, "omega": omega, "gamma": gamma,
                "A_baseline":     A_baseline_theta,
                "A_cmu":          A_cmu_theta,
                "y_baseline":     y_baseline_theta,
                "y_cmu":          y_cmu_theta,
                "Y_EU_base":      Y_EU_base_theta,
                "Y_EU_cmu":       Y_EU_cmu_theta,
                "total_effect":   total_effect,
                "capital_effect": capital_effect,
                "tfp_effect":     tfp_effect,
                "mpk_baseline":   mpk_baseline,
                "mpk_cmu":        mpk_cmu,
                "eu_hb_change":   eu_hb_change,
                "dY_EU_pct":      (Y_EU_cmu_theta - Y_EU_base_theta) / Y_EU_base_theta * 100,
                "cap_contribution_EU": (alp[eu_idx] * (k_cmu[eu_idx] - k_baseline[eu_idx]) / k_baseline[eu_idx]).mean() * 100,
                "tfp_contribution_EU": ((A_cmu_theta[eu_idx] - A_baseline_theta[eu_idx]) / A_baseline_theta[eu_idx]).mean() * 100,
                "sigma_mpk_base":  sigma_mpk_base,
                "sigma_mpk_cmu":   sigma_mpk_cmu,
                "sigma_reduction": sigma_reduction,
            }

# ============================================================
# 15.  ECONOMIC SENSE CHECKS
# ============================================================
print()
print("=" * 60)
print("ECONOMIC SENSE CHECKS — Endogenous TFP extension")
print("=" * 60)

# Check 1: Countries that gain more foreign capital should have larger TFP gains
print(f"\nCheck 1 — Corr(Δk_foreign, ΔA) across EU27 at ω={_gamma_mid}, γ={_gamma0}, θ={_theta1}:")
r = endo_results[(_theta1, _gamma_mid, _gamma0)]
delta_k_foreign_eu = results[(_gamma_mid, _gamma0)]["k_foreign_cmu"][eu_idx] - k_foreign_baseline[eu_idx]
delta_A_eu         = r["A_cmu"][eu_idx] - r["A_baseline"][eu_idx]
corr_check1 = np.corrcoef(delta_k_foreign_eu, delta_A_eu)[0, 1]
print(f"  Correlation = {corr_check1:.6f}  {'OK ✓ (positive)' if corr_check1 > 0 else 'FAIL (should be positive)'}")

# Check 2: θ=0 results have zero TFP effect
print(f"\nCheck 2 — θ={_theta0}: max |ΔA/A| = 0 (no TFP spillover):")
for omega, gamma in [(_gamma0, _gamma0), (_gamma_mid, _gamma0), (_gamma1, _gamma0), (_gamma_mid, _gamma_mid), (_gamma1, _gamma1)]:
    r0 = endo_results[(_theta0, omega, gamma)]
    max_tfp_diff = np.max(np.abs(r0["tfp_effect"]))
    print(f"  ω={omega:.2f}, γ={gamma:.2f}: max |ΔA/A| = {max_tfp_diff:.2e}  {'OK ✓' if max_tfp_diff < 1e-10 else 'WARNING'}")

# Check 3: TFP contribution < capital deepening contribution for EU27
print("\nCheck 3 — TFP contribution < capital deepening contribution (EU27 avg):")
for omega, gamma in [(_gamma_mid, _gamma0), (_gamma1, _gamma0), (_gamma1, _gamma1)]:
    r10 = endo_results[(_theta1, omega, gamma)]
    cap_share = abs(r10["cap_contribution_EU"])
    tfp_share = abs(r10["tfp_contribution_EU"])
    ok = tfp_share < cap_share
    print(f"  ω={omega:.2f}, γ={gamma:.2f}: capital={cap_share:.4f}%  TFP={tfp_share:.4f}%  {'OK ✓' if ok else 'WARNING: TFP dominates'}")

# Check 4: Non-EU countries lose foreign capital under full combined CMU
print(f"\nCheck 4 — Non-EU lose foreign capital under ω={_gamma1}, γ={_gamma1}, θ={_theta1}:")
r_full = endo_results[(_theta1, _gamma1, _gamma1)]
outside_idx = [idx[c] for c in OUTSIDE]
for ci in outside_idx:
    c = COUNTRIES[ci]
    dk_f = results[(_gamma1, _gamma1)]["k_foreign_cmu"][ci] - k_foreign_baseline[ci]
    dA   = r_full["A_cmu"][ci] - r_full["A_baseline"][ci]
    ok   = dk_f < 0 and dA < 0
    print(f"  {c}: Δk_foreign={dk_f:+.1f}  ΔA={dA:+.6f}  {'OK ✓' if ok else 'NOTE: unexpected sign'}")

# Check 5: EU GDP gain increasing in θ (hard-only, φ=1.0)
print(f"\nCheck 5 — EU GDP gain increasing in θ (ω={_gamma1}, γ={_gamma0}):")
gains = [endo_results[(theta, _gamma1, _gamma0)]["dY_EU_pct"] for theta in THETA_SCENARIOS]
mono  = all(gains[i] <= gains[i+1] for i in range(len(gains)-1))
print(f"  gains = {[f'{g:.5f}%' for g in gains]}  {'OK ✓' if mono else 'WARNING: not monotone'}")


# ============================================================
# 16.  ROBUSTNESS — η × θ × φ
# ============================================================
print()
print("=" * 60)
print("STEP 9 — Robustness checks (η × θ × φ)")
print("=" * 60)

def run_model_variant_endo(D_hard, D_ling, D_numart, R, M, s, k_pwt, A_bar_in,
                           L_in, alp_in, Pi_data_in, countries,
                           EU27_list, eta_val, barrier_list, theta_val, label):
    """
    Full model run for given (eta, theta).
    Scenarios: all (ω, γ) combinations from barrier_list × barrier_list.
    Re-calibrates Δ_ii for the given eta.
    γ only reduces D_numart (news article distance); D_ling is fixed.
    """
    n_ = len(countries)
    eu_flag_ = np.array([1 if c in EU27_list else 0 for c in countries])
    eu_pair_ = np.outer(eu_flag_, eu_flag_) * (1 - np.eye(n_))
    eu_pair_bool_ = eu_pair_.astype(bool)
    eu_idx_  = [i for i, c in enumerate(countries) if c in EU27_list]

    D = (D_hard * D_ling * D_numart).copy()
    RM_ = (R ** eta_val) * M
    pi_home_ = np.diag(Pi_data_in)

    for i, c in enumerate(countries):
        if c in FINANCIAL_CENTRES:
            D[i, i] = 1.0
        elif not np.isnan(pi_home_[i]) and 0 < pi_home_[i] < 0.9999:
            D[i, i] = 0.01
        else:
            D[i, i] = 1.0

    for _ in range(20_000):
        max_err_ = 0.0
        for i, c in enumerate(countries):
            if c in FINANCIAL_CENTRES:
                continue
            if np.isnan(pi_home_[i]) or pi_home_[i] <= 0 or pi_home_[i] >= 0.9999:
                continue
            w = RM_ / D[i, :]
            Z = w.sum()
            ph = w[i] / Z
            D[i, i] *= ph / pi_home_[i]
            max_err_ = max(max_err_, abs(ph - pi_home_[i]))
        if max_err_ < 1e-5:
            break

    Pi_base_   = compute_portfolio(D, R, M, eta_val)
    k_base_    = Pi_base_.T @ s
    k_f_base_  = np.maximum(k_base_ - np.diag(Pi_base_) * s, 0.0)
    f_base_    = k_f_base_ / k_pwt
    A_base_    = compute_tfp(A_bar_in, f_base_, theta_val)
    y_base_    = cobb_douglas(A_base_, k_base_, L_in, alp_in)
    Y_EU_base_ = y_base_[eu_idx_].sum()

    rows = []
    _pairs = [(o, g) for o in barrier_list for g in barrier_list]
    for omega, gamma in tqdm(_pairs, desc=f"  {label} (ω×γ)", unit="scenario", position=1, leave=False):
        hard_cmu_   = np.where(eu_pair_bool_, D_hard   ** (1 - omega), D_hard)
        numart_cmu_ = np.where(eu_pair_bool_, D_numart ** (1 - gamma), D_numart)
        # Linguistic barrier fixed — not reduced by γ
        D_cmu_      = hard_cmu_ * D_ling * numart_cmu_
        np.fill_diagonal(D_cmu_, np.diag(D))
        Pi_cmu_   = compute_portfolio(D_cmu_, R, M, eta_val)
        k_cmu_    = Pi_cmu_.T @ s
        k_f_cmu_  = np.maximum(k_cmu_ - np.diag(Pi_cmu_) * s, 0.0)
        f_cmu_    = k_f_cmu_ / k_pwt
        A_cmu_    = compute_tfp(A_bar_in, f_cmu_, theta_val)
        y_cmu_    = cobb_douglas(A_cmu_, k_cmu_, L_in, alp_in)
        Y_EU_cmu_ = y_cmu_[eu_idx_].sum()
        eu_hb_ch  = (np.nanmean(np.diag(Pi_cmu_)[eu_idx_])
                     - np.nanmean(np.diag(Pi_base_)[eu_idx_]))
        dY_EU = (Y_EU_cmu_ - Y_EU_base_) / Y_EU_base_ * 100
        rows.append({
            "label": label, "eta": eta_val, "theta": theta_val,
            "omega": omega, "gamma": gamma,
            "ΔY_EU/Y_EU (%)": dY_EU,
            "Avg ΔHomeBias (EU)": eu_hb_ch,
        })
    return pd.DataFrame(rows)


D_hard_offdiag   = Delta_hard.copy();   np.fill_diagonal(D_hard_offdiag, 1.0)
D_ling_offdiag   = Delta_ling.copy();   np.fill_diagonal(D_ling_offdiag, 1.0)
D_numart_offdiag = Delta_numart.copy(); np.fill_diagonal(D_numart_offdiag, 1.0)

rob_rows = []
_rob_combos = [(e, el, t) for e, el in [(0.5, "η=0.5"), (1.0, "η=1.0 (baseline)"), (2.0, "η=2.0")] for t in THETA_SCENARIOS]
for eta_val, eta_lbl, theta_val in tqdm(_rob_combos, desc="Robustness (η×θ)", unit="variant", position=0, leave=True):
    lbl = f"{eta_lbl}, θ={theta_val}"
    r   = run_model_variant_endo(
        D_hard_offdiag, D_ling_offdiag, D_numart_offdiag,
        R_vec, Y_vec, s_vec, k_PWT, A_bar,
        L_prod, alp, Pi_data.values, COUNTRIES, EU27,
        eta_val, BARRIER_SCENARIOS, theta_val, lbl,
    )
    rob_rows.append(r)

rob_df = pd.concat(rob_rows, ignore_index=True)
print("\n  Robustness Summary (η × θ × ω × γ):")
print(rob_df.round(5).to_string(index=False))

if (rob_df["ΔY_EU/Y_EU (%)"] >= 0).all():
    print("\n  OK ✓ — EU GDP gain is non-negative for all scenarios.")
else:
    negative_cases = rob_df[rob_df["ΔY_EU/Y_EU (%)"] < 0]
    print(f"\n  WARNING: {len(negative_cases)} cases with negative EU GDP gain:")
    print(negative_cases.to_string(index=False))


# ============================================================
# 17.  SUMMARY TABLES
# ============================================================
print()
print("=" * 60)
print("FINAL SUMMARY TABLES")
print("=" * 60)

print("\nTable 1 — Gravity coefficients (PPML):")
spec_coefs = {}
for col in ["d_geo", "d_ling", "same_legal_origin", "ln_Y_i", "ln_Y_j"]:
    if col in ppml.params.index:
        spec_coefs[col] = {
            "coeff":    ppml.params.get(col, np.nan),
            "p-value":  ppml.pvalues.get(col, np.nan),
            "in_wedge": col in beta_dict,
        }
grav_summary = pd.DataFrame(spec_coefs).T
print(grav_summary.round(4).to_string())
print(f"  Specification: d_geo + d_ling + same_legal_origin + ln_Y_i + ln_Y_j + C(year)")

# -----------------------------------------------------------
print(f"\nTable 2a — EU GDP gain (%) — γ={_gamma0} (fin only), θ={_theta1}")
print(f"  {'ω':>6}  {'Avg Δπ_EU':>12}  {'ΔY_EU/Y_EU':>12}  {'Capital':>12}  {'TFP':>10}  {'σ_MPK red.':>12}")
print("-" * 72)
for omega in BARRIER_SCENARIOS:
    r = endo_results[(_theta1, omega, _gamma0)]
    print(f"  {omega:>6.2f}  {r['eu_hb_change']:>12.5f}  {r['dY_EU_pct']:>12.5f}%"
          f"  {r['cap_contribution_EU']:>12.5f}%  {r['tfp_contribution_EU']:>10.5f}%"
          f"  {r['sigma_reduction']:>12.2f}%")

print(f"\nTable 2b — EU GDP gain (%) — γ=ω (fin+ling together), θ={_theta1}")
print(f"  {'ω':>6}  {'Avg Δπ_EU':>12}  {'ΔY_EU/Y_EU':>12}  {'Capital':>12}  {'TFP':>10}  {'σ_MPK red.':>12}")
print("-" * 72)
for omega in BARRIER_SCENARIOS:
    r = endo_results[(_theta1, omega, omega)]
    print(f"  {omega:>6.2f}  {r['eu_hb_change']:>12.5f}  {r['dY_EU_pct']:>12.5f}%"
          f"  {r['cap_contribution_EU']:>12.5f}%  {r['tfp_contribution_EU']:>10.5f}%"
          f"  {r['sigma_reduction']:>12.2f}%")

# -----------------------------------------------------------
print(f"\nTable 3 — Country-level results: ω={_gamma1}, γ={_gamma0}, θ={_theta1} (EU27):")
r_main = endo_results[(_theta1, _gamma1, _gamma0)]
ctry_table = pd.DataFrame({
    "Δk_i/k_i (%)":  (results[(_gamma1, _gamma0)]["k_cmu"] - k_baseline) / k_baseline * 100,
    "ΔA_i/A_i (%)":  (r_main["A_cmu"] - r_main["A_baseline"]) / r_main["A_baseline"] * 100,
    "Δy_i/y_i (%)":  r_main["total_effect"] * 100,
    "Capital share": (r_main["capital_effect"] / np.where(r_main["total_effect"] == 0, np.nan, r_main["total_effect"])),
    "TFP share":     (r_main["tfp_effect"]     / np.where(r_main["total_effect"] == 0, np.nan, r_main["total_effect"])),
}, index=COUNTRIES)
print(ctry_table.loc[EU27].round(4).to_string())

# -----------------------------------------------------------
print(f"\nTable 4 — Top 5 EU capital gainers: ω={_gamma1}, γ={_gamma1}, θ={_theta1}:")
dk_full   = (results[(_gamma1, _gamma1)]["k_cmu"] - k_baseline) / k_baseline * 100
dk_series = pd.Series(dk_full, index=COUNTRIES)
print(dk_series[EU27].sort_values(ascending=False).head(5).round(3).to_string())

print("\nTable 5 — Off-diagonal portfolio correlation (data vs model):")
print(f"  {corr_off:.4f}")

print(f"\nTable 6 — MPK dispersion reduction (EU27), θ={_theta1}:")
print(f"  {'ω':>6}  {'γ':>6}  {'σ_baseline':>12}  {'σ_CMU':>10}  {'reduction (%)':>14}")
print("-" * 58)
for omega in BARRIER_SCENARIOS:
    for gamma in BARRIER_SCENARIOS:
        r = endo_results[(_theta1, omega, gamma)]
        print(f"  {omega:>6.2f}  {gamma:>6.2f}  {r['sigma_mpk_base']:>12.6f}  {r['sigma_mpk_cmu']:>10.6f}  {r['sigma_reduction']:>14.2f}%")

print("\nDone.")

# ============================================================
# 18.  EXPORT TO EXCEL
# ============================================================
print()
print("=" * 60)
print("STEP 10 — Exporting results to v4-simulation.xlsx")
print("=" * 60)

EXCEL_PATH = "v4-simulation.xlsx"

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:

    # ----------------------------------------------------------
    # Sheet 1: Config
    # ----------------------------------------------------------
    config_df = pd.DataFrame([
        {"parameter": "BASE_YEAR",      "value": BASE_YEAR},
        {"parameter": "ETA (baseline)", "value": ETA},
        {"parameter": "BARRIER_SCENARIOS", "value": str(BARRIER_SCENARIOS)},
        {"parameter": "THETA_SCENARIOS","value": str(THETA_SCENARIOS)},
        {"parameter": "n_countries",    "value": n},
        {"parameter": "EU27",           "value": ", ".join(EU27)},
        {"parameter": "OUTSIDE",        "value": ", ".join(OUTSIDE)},
        {"parameter": "FINANCIAL_CENTRES", "value": ", ".join(sorted(FINANCIAL_CENTRES))},
        {"parameter": "DATA_PATH",      "value": DATA_PATH},
    ])
    config_df.to_excel(writer, sheet_name="Config", index=False)

    # ----------------------------------------------------------
    # Sheet 2: Gravity_Coefficients
    # ----------------------------------------------------------
    ci = ppml.conf_int()
    ci.columns = ["CI_lower", "CI_upper"]
    grav_df = pd.DataFrame({
        "coeff_scaled":   ppml.params,
        "coeff_original": ppml_params_rescaled,
        "std_error":      ppml.bse,
        "t_stat":         ppml.tvalues,
        "p_value":        ppml.pvalues,
    }).join(ci)
    grav_df.index.name = "regressor"
    grav_df.reset_index().to_excel(writer, sheet_name="Gravity_Coefficients", index=False)

    # ----------------------------------------------------------
    # Sheet 3: Macro_Variables
    # ----------------------------------------------------------
    macro_df = pd.DataFrame({
        "country":             COUNTRIES,
        "R_normalised":        R_vec,
        "GDP_USDmn":           Y_vec,  # GDP used as portfolio-size weight (replaces market cap)
        "k_PWT_USDmn":         k_PWT,
        "A_bar":               A_bar,
        "L_labour":            L_prod,
        "alpha":               alp,
        "pi_home_data":        pi_home,
        "pi_home_model":       np.diag(Pi_baseline),
        "delta_ii_calibrated": np.diag(Delta_baseline),
        "s_total_wealth":      s_vec,
        "k_equity_baseline":   k_baseline,
        "k_foreign_baseline":  k_foreign_baseline,
        "f_foreign_intensity": f_baseline,
        "in_EU27":             [1 if c in EU27 else 0 for c in COUNTRIES],
    })
    macro_df.to_excel(writer, sheet_name="Macro_Variables", index=False)

    # ----------------------------------------------------------
    # Sheet 4: Wedge_Hard  (financial system barriers)
    # ----------------------------------------------------------
    pd.DataFrame(Delta_hard, index=COUNTRIES, columns=COUNTRIES).to_excel(
        writer, sheet_name="Wedge_Hard"
    )

    # ----------------------------------------------------------
    # Sheet 5: Wedge_Soft  (linguistic barriers)
    # ----------------------------------------------------------
    pd.DataFrame(Delta_soft, index=COUNTRIES, columns=COUNTRIES).to_excel(
        writer, sheet_name="Wedge_Soft"
    )

    # ----------------------------------------------------------
    # Sheet 6: Wedge_Baseline  (combined + calibrated diagonal)
    # ----------------------------------------------------------
    pd.DataFrame(Delta_baseline, index=COUNTRIES, columns=COUNTRIES).to_excel(
        writer, sheet_name="Wedge_Baseline"
    )

    # ----------------------------------------------------------
    # Sheet 7: Portfolio_Data  (observed shares)
    # ----------------------------------------------------------
    Pi_data.to_excel(writer, sheet_name="Portfolio_Data")

    # ----------------------------------------------------------
    # Sheet 8: Portfolio_Baseline  (model-implied shares)
    # ----------------------------------------------------------
    pd.DataFrame(Pi_baseline, index=COUNTRIES, columns=COUNTRIES).to_excel(
        writer, sheet_name="Portfolio_Baseline"
    )

    # ----------------------------------------------------------
    # Sheet 9: Capital_Reallocation
    # Columns: country, k_baseline, then k_cmu and Δk/k for each (omega, gamma)
    # ----------------------------------------------------------
    cap_df = pd.DataFrame({"country": COUNTRIES, "k_baseline": k_baseline})
    for (omega, gamma) in [(o, g) for o in BARRIER_SCENARIOS for g in BARRIER_SCENARIOS]:
        k_cmu = results[(omega, gamma)]["k_cmu"]
        tag   = f"om{omega:.3f}_gm{gamma:.3f}"
        cap_df[f"k_cmu_{tag}"]   = k_cmu
        cap_df[f"dk_pct_{tag}"]  = (k_cmu - k_baseline) / k_baseline * 100
    cap_df.to_excel(writer, sheet_name="Capital_Reallocation", index=False)

    # ----------------------------------------------------------
    # Sheet 10: Foreign_Capital
    # ----------------------------------------------------------
    fcap_df = pd.DataFrame({
        "country":             COUNTRIES,
        "k_foreign_baseline":  k_foreign_baseline,
        "f_baseline":          f_baseline,
        "in_EU27":             [1 if c in EU27 else 0 for c in COUNTRIES],
    })
    for (omega, gamma) in [(o, g) for o in BARRIER_SCENARIOS for g in BARRIER_SCENARIOS]:
        tag = f"om{omega:.3f}_gm{gamma:.3f}"
        fcap_df[f"k_foreign_{tag}"] = results[(omega, gamma)]["k_foreign_cmu"]
        fcap_df[f"f_{tag}"]         = results[(omega, gamma)]["f_cmu"]
    fcap_df.to_excel(writer, sheet_name="Foreign_Capital", index=False)

    # ----------------------------------------------------------
    # Sheet 11: EU_Summary
    # One row per (theta, mode, phi) with all EU aggregate stats
    # ----------------------------------------------------------
    eu_summary_rows = []
    for (theta, omega, gamma), r in endo_results.items():
        eu_summary_rows.append({
            "theta":                  theta,
            "omega":                  omega,
            "gamma":                  gamma,
            "Y_EU":                   r["Y_EU_cmu"] - r["Y_EU_base"],
            "dY_EU_pct":              r["dY_EU_pct"],
            "cap_contribution_EU_pct":r["cap_contribution_EU"],
            "tfp_contribution_EU_pct":r["tfp_contribution_EU"],
            "eu_hb_change":           r["eu_hb_change"],
            "sigma_mpk_baseline":     r["sigma_mpk_base"],
            "sigma_mpk_cmu":          r["sigma_mpk_cmu"],
            "sigma_mpk_reduction_pct":r["sigma_reduction"],
        })
    pd.DataFrame(eu_summary_rows).sort_values(
        ["theta", "omega", "gamma"]
    ).to_excel(writer, sheet_name="EU_Summary", index=False)

    # ----------------------------------------------------------
    # Sheet 12: Output_Country
    # Country-level output effects for every (theta, omega, gamma).
    # This is the master sheet for all plotting — includes every
    # country-level quantity that could ever be needed.
    # ----------------------------------------------------------

    # Pre-build lookup dicts to avoid repeated indexing inside the loop
    _pi_baseline_diag = np.diag(Pi_baseline)        # π_ii baseline (home bias)
    _delta_diag       = np.diag(Delta_baseline)     # Δ_ii calibrated
    _delta_hard_diag  = np.diag(Delta_hard)
    _delta_soft_diag  = np.diag(Delta_soft)

    out_rows = []
    for (theta, omega, gamma), r in endo_results.items():
        _res_og   = results[(omega, gamma)]
        _Pi_cmu   = _res_og["Pi_cmu"]
        _k_cmu    = _res_og["k_cmu"]
        _kf_cmu   = _res_og["k_foreign_cmu"]
        _f_cmu    = _res_og["f_cmu"]
        _pi_cmu_diag = np.diag(_Pi_cmu)

        for ci_idx, c in enumerate(COUNTRIES):
            _L   = L_prod[ci_idx]
            _kb  = k_baseline[ci_idx]
            _kc  = _k_cmu[ci_idx]
            _kfb = k_foreign_baseline[ci_idx]
            _kfc = _kf_cmu[ci_idx]
            _fb  = f_baseline[ci_idx]
            _fc  = _f_cmu[ci_idx]
            _yb  = r["y_baseline"][ci_idx]
            _yc  = r["y_cmu"][ci_idx]
            _Ab  = r["A_baseline"][ci_idx]
            _Ac  = r["A_cmu"][ci_idx]
            _mpk_b = r["mpk_baseline"][ci_idx]
            _mpk_c = r["mpk_cmu"][ci_idx]
            _tot   = r["total_effect"][ci_idx]
            _cap   = r["capital_effect"][ci_idx]
            _tfp   = r["tfp_effect"][ci_idx]

            out_rows.append({
                # ── scenario identifiers ──────────────────────────
                "theta":            theta,
                "omega":            omega,
                "gamma":            gamma,

                # ── country identifiers ───────────────────────────
                "country":          c,
                "in_EU27":          1 if c in EU27 else 0,
                "is_financial_ctr": 1 if c in FINANCIAL_CENTRES else 0,

                # ── structural / data parameters ─────────────────
                "alpha":            alp[ci_idx],        # capital share
                "A_bar":            A_bar[ci_idx],      # exogenous TFP (Ā)
                "L":                _L,                 # labour force
                "R_normalised":     R_vec[ci_idx],      # normalised return
                "GDP_USDmn":        Y_vec[ci_idx],      # GDP (portfolio-size weight)
                "k_PWT":            k_PWT[ci_idx],      # PWT physical capital
                "s_total_wealth":   s_vec[ci_idx],      # total equity wealth

                # ── wedges ────────────────────────────────────────
                "delta_ii":         _delta_diag[ci_idx],        # home wedge (calibrated)
                "delta_hard_mean":  Delta_hard[ci_idx, :].mean(),  # avg hard wedge (row i)
                "delta_soft_mean":  Delta_soft[ci_idx, :].mean(),  # avg soft wedge (row i)

                # ── portfolio shares ──────────────────────────────
                "pi_home_data":     pi_home[ci_idx],            # observed home bias
                "pi_home_baseline": _pi_baseline_diag[ci_idx],  # model baseline home bias
                "pi_home_cmu":      _pi_cmu_diag[ci_idx],       # CMU home bias
                "d_pi_home":        _pi_cmu_diag[ci_idx] - _pi_baseline_diag[ci_idx],
                "d_pi_home_pct":    (_pi_cmu_diag[ci_idx] - _pi_baseline_diag[ci_idx])
                                    / _pi_baseline_diag[ci_idx] * 100
                                    if _pi_baseline_diag[ci_idx] > 0 else np.nan,

                # ── capital (equity, total) ───────────────────────
                "k_baseline":       _kb,
                "k_cmu":            _kc,
                "dk":               _kc - _kb,
                "dk_pct":           (_kc - _kb) / _kb * 100 if _kb > 0 else np.nan,
                "k_baseline_pc":    _kb / _L if _L > 0 else np.nan,
                "k_cmu_pc":         _kc / _L if _L > 0 else np.nan,
                "dk_pc_pct":        ((_kc - _kb) / _L) / (_kb / _L) * 100
                                    if _kb > 0 and _L > 0 else np.nan,

                # ── foreign capital ───────────────────────────────
                "k_foreign_baseline":   _kfb,
                "k_foreign_cmu":        _kfc,
                "dk_foreign":           _kfc - _kfb,
                "dk_foreign_pct":       (_kfc - _kfb) / _kfb * 100 if _kfb > 0 else np.nan,
                "f_baseline":           _fb,         # foreign intensity (baseline)
                "f_cmu":                _fc,         # foreign intensity (CMU)
                "df":                   _fc - _fb,
                "df_pct":               (_fc - _fb) / _fb * 100 if _fb > 0 else np.nan,

                # ── TFP ───────────────────────────────────────────
                "A_baseline":       _Ab,
                "A_cmu":            _Ac,
                "dA":               _Ac - _Ab,
                "dA_pct":           (_Ac - _Ab) / _Ab * 100 if _Ab > 0 else np.nan,

                # ── output (total) ────────────────────────────────
                "y_baseline":       _yb,
                "y_cmu":            _yc,
                "dy":               _yc - _yb,
                "dy_pct":           _tot * 100,       # = total_effect * 100

                "y_baseline_pc":    _yb / _L if _L > 0 else np.nan,
                "y_cmu_pc":         _yc / _L if _L > 0 else np.nan,
                "dy_pc_pct":        ((_yc - _yb) / _L) / (_yb / _L) * 100
                                    if _yb > 0 and _L > 0 else np.nan,

                # ── output decomposition ──────────────────────────
                "capital_effect_pct": _cap * 100,    # α * Δk/k
                "tfp_effect_pct":     _tfp * 100,    # ΔA/A (approx)
                "total_effect_pct":   _tot * 100,
                "capital_share_of_dy": _cap / _tot if _tot != 0 else np.nan,
                "tfp_share_of_dy":     _tfp / _tot if _tot != 0 else np.nan,

                # ── marginal product of capital ───────────────────
                "mpk_baseline":     _mpk_b,
                "mpk_cmu":          _mpk_c,
                "dmpk":             _mpk_c - _mpk_b,
                "dmpk_pct":         (_mpk_c - _mpk_b) / _mpk_b * 100 if _mpk_b > 0 else np.nan,
            })

    pd.DataFrame(out_rows).sort_values(
        ["theta", "omega", "gamma", "country"]
    ).to_excel(writer, sheet_name="Output_Country", index=False)

    # ----------------------------------------------------------
    # Sheet 13: Country_Detail_Main
    # Main scenario: theta=0.10, hard, phi=1.0 — country table
    # ----------------------------------------------------------
    ctry_table.reset_index().rename(columns={"index": "country"}).to_excel(
        writer, sheet_name="Country_Detail_Main", index=False
    )

    # ----------------------------------------------------------
    # Sheet 14: MPK_Dispersion
    # MPK vectors for baseline and all (mode, phi) under theta=0.10
    # ----------------------------------------------------------
    mpk_df = pd.DataFrame({"country": COUNTRIES})
    mpk_df[f"mpk_baseline_theta{_theta1:.3f}"] = endo_results[(_theta1, _gamma0, _gamma0)]["mpk_baseline"]
    for (omega, gamma) in [(o, g) for o in BARRIER_SCENARIOS for g in BARRIER_SCENARIOS]:
        tag = f"om{omega:.3f}_gm{gamma:.3f}"
        mpk_df[f"mpk_{tag}"] = endo_results[(_theta1, omega, gamma)]["mpk_cmu"]
    mpk_df.to_excel(writer, sheet_name="MPK_Dispersion", index=False)

    # ----------------------------------------------------------
    # Sheet 15: Robustness
    # ----------------------------------------------------------
    rob_df.to_excel(writer, sheet_name="Robustness", index=False)

    # ----------------------------------------------------------
    # Sheet 16: Portfolio_CMU_Snapshots
    # Pi_cmu matrices for 4 corner scenarios: (ω=0.5,γ=0), (ω=1,γ=0), (ω=0.5,γ=0.5), (ω=1,γ=1)
    # Stacked vertically with a scenario label column
    # ----------------------------------------------------------
    snap_frames = []
    for (omega, gamma) in [(_gamma_mid, _gamma0), (_gamma1, _gamma0), (_gamma_mid, _gamma_mid), (_gamma1, _gamma1)]:
        df_snap = pd.DataFrame(
            results[(omega, gamma)]["Pi_cmu"],
            index=COUNTRIES, columns=COUNTRIES
        ).reset_index().rename(columns={"index": "origin\\dest"})
        df_snap.insert(0, "scenario", f"om{omega:.2f}_gm{gamma:.2f}")
        snap_frames.append(df_snap)
        # blank separator row
        sep = pd.DataFrame([[""] * df_snap.shape[1]], columns=df_snap.columns)
        snap_frames.append(sep)
    pd.concat(snap_frames, ignore_index=True).to_excel(
        writer, sheet_name="Portfolio_CMU_Snapshots", index=False
    )

    # ----------------------------------------------------------
    # Sheet 17: Portfolio_Flows  (long format)
    # Four named scenarios only — keeps the sheet manageable:
    #   baseline      : ω=0,   γ=0,   θ=0,    η=1.0
    #   most_probable : ω≈0.3, γ≈0.3, θ≈0.05, η=1.0
    #   mid           : ω=0.5, γ=0.5, θ=0.05, η=1.0
    #   max           : ω=1,   γ=1,   θ=0.10, η=1.0
    # ----------------------------------------------------------
    _FLOW_SCENARIOS = [
        ("baseline",      ETA, _theta0,   _gamma0,    _gamma0),
        ("most_probable", ETA, _theta_mid, nearest(BARRIER_SCENARIOS, 0.30), nearest(BARRIER_SCENARIOS, 0.30)),
        ("mid",           ETA, _theta_mid, _gamma_mid, _gamma_mid),
        ("max",           ETA, _theta1,   _gamma1,    _gamma1),
    ]

    print("  Building Portfolio_Flows (long format) …")
    flow_rows = []

    for _label, _eta, _theta, _omega, _gamma in tqdm(
        _FLOW_SCENARIOS, desc="Portfolio_Flows", unit="scenario", position=0, leave=True
    ):
        _Pi = results[(_omega, _gamma)]["Pi_cmu"]
        _Pi_df = pd.DataFrame(_Pi, index=COUNTRIES, columns=COUNTRIES)
        for i_iso in COUNTRIES:
            _si = s_vec[idx[i_iso]]
            for j_iso in COUNTRIES:
                _share = _Pi_df.loc[i_iso, j_iso]
                flow_rows.append({
                    "scenario":        _label,
                    "eta":             _eta,
                    "theta":           _theta,
                    "omega":           _omega,
                    "gamma":           _gamma,
                    "iso3_i":          i_iso,
                    "iso3_j":          j_iso,
                    "portfolio_share": _share,
                    "k_flow":          _si * _share,
                })

    (pd.DataFrame(flow_rows)
       .sort_values(["scenario", "iso3_i", "iso3_j"])
       .to_excel(writer, sheet_name="Portfolio_Flows", index=False))
    print(f"  Portfolio_Flows: {len(flow_rows):,} rows  ({len(_FLOW_SCENARIOS)} scenarios × {n}² pairs)")

print(f"  Exported {len(writer.sheets)} sheets to {EXCEL_PATH}")
print("  Sheets:")
for name in writer.sheets:
    print(f"    • {name}")
print("\nAll done ✓")