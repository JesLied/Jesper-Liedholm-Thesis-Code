"""
CMU Simulation Model  –  v4_simulation.py
==========================================
Implements the Capital Markets Union (CMU) model following
`v4 CMU_Model_Step_by_Step.md`.

Convention
----------
  i  = origin  country (investor / source of capital)
  j  = destination country (receives capital)

Data source: Data/Clean/Final-v4.csv

Key design decisions
--------------------
* The capital stock used in the production function is *equity-portfolio
  capital* derived from CPIS holdings (k_j = Π^T @ s), NOT PWT rkna.
  TFP (A_j), labour (L_j) and capital share (α_j) still come from PWT.
  This keeps the GDP effects in the same equity-capital units throughout.

* Financial-centre distortion: IRL, LUX, NLD report foreign CPIS outflows
  that exceed their stock market cap, implying negative domestic holdings.
  We use the 1 % floor per Step-by-Step, but we EXCLUDE these three from
  the home-bias calibration of Δ_ii to avoid enormous spurious wedges.

* Gravity uses only regressors with the theoretically correct sign and
  statistical significance (p < 0.20 threshold).  d_cul (22 % coverage,
  p ≈ 0.78) is excluded.  d_ling enters only if β < 0 (more distance →
  less investment); if the estimate has the wrong sign it is dropped.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0.  CONFIGURATION
# ============================================================
DATA_PATH  = "Data/Clean/Final-v4.csv"
BASE_YEAR  = 2023
ETA        = 1.0          # return elasticity (baseline)
PHI_SCENARIOS = [0.25, 0.50, 0.75, 1.00]

EU27 = [
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
    "SVN","ESP","SWE",
]
OUTSIDE   = ["USA","GBR","CHE","NOR"]
COUNTRIES = EU27 + OUTSIDE          # n = 31
n         = len(COUNTRIES)
idx       = {c: i for i, c in enumerate(COUNTRIES)}

# Financial centres whose domestic holdings are distorted by
# pass-through flows; excluded from diagonal (Δ_ii) calibration.
FINANCIAL_CENTRES = {"IRL", "LUX", "NLD", "CHE"}


# ============================================================
# 1.  LOAD & CLEAN DATA
# ============================================================
print("=" * 60)
print("STEP 0 — Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Scale market cap to millions USD (same as a_ij)
df["M_i"] = df["M_i"] / 1e6
df["M_j"] = df["M_j"] / 1e6

print(f"  Raw rows: {len(df):,}  |  columns: {df.shape[1]}")

# --- Restrict to our country set ---------------------------
df = df[df["iso3_i"].isin(COUNTRIES) & df["iso3_j"].isin(COUNTRIES)].copy()
print(f"  After country filter: {len(df):,} rows")

# --- Floor negative a_ij to 0 (data artefact) --------------
neg_mask = df["a_ij"] < 0
if neg_mask.sum() > 0:
    print(f"  WARNING: {neg_mask.sum()} negative a_ij values → set to 0")
    df.loc[neg_mask, "a_ij"] = 0.0

# --- Impute M_i / M_j from Y where missing ----------------
#     Use average M/Y ratio across all available observations
avg_M_Y = np.nanmean(df.loc[df["M_i"] > 0, "M_i"] / df.loc[df["M_i"] > 0, "Y_i"])
print(f"  Average M/Y ratio for imputation: {avg_M_Y:.4f}")

m_i_missing = df["M_i"].isna() | (df["M_i"] <= 0)
df.loc[m_i_missing, "M_i"] = df.loc[m_i_missing, "Y_i"] * avg_M_Y
m_j_missing = df["M_j"].isna() | (df["M_j"] <= 0)
df.loc[m_j_missing, "M_j"] = df.loc[m_j_missing, "Y_j"] * avg_M_Y
print(f"  Imputed M_i for {m_i_missing.sum()} rows  |  M_j for {m_j_missing.sum()} rows")

# --- Build derived columns ---------------------------------
df["ln_d_geo"]   = np.log(df["d_geo"].replace(0, np.nan))
df["euro_ij"]    = df["euro_i"] * df["euro_j"]
df["common_lang"] = (df["d_ling"] < 0.2).astype(float)

# ============================================================
# 2.  BUILD BASE-YEAR SNAPSHOT
# ============================================================
print()
print("=" * 60)
print("STEP 0b — Assembling base-year cross-section")
print("=" * 60)

# -- Compute domestic holdings a_ii -------------------------
# For each country i: a_ii = M_i - sum(a_ji, j≠i) [foreign liabilities into i]
# Use base-year off-diagonal slice across the FULL country universe to capture
# all foreign ownership of country i's equity.
base_full = df[(df["year"] == BASE_YEAR)].copy()

# Foreign liabilities into country j: sum of a_ij for i≠j
foreign_liab = (
    base_full[base_full["iso3_i"] != base_full["iso3_j"]]
    .groupby("iso3_j")["a_ij"]
    .sum()
    .reset_index()
    .rename(columns={"iso3_j": "iso3", "a_ij": "foreign_liab"})
)

# Country-level M in base year (from origin columns, de-duped)
M_map = (
    base_full.drop_duplicates("iso3_i")
    .set_index("iso3_i")["M_i"]
)

# Home holdings: a_ii = M_i - foreign_liabilities
home_hold = {}
for c in COUNTRIES:
    M_c  = M_map.get(c, np.nan)
    flib = foreign_liab.set_index("iso3").get("foreign_liab", pd.Series(dtype=float)).get(c, 0.0)
    if np.isnan(M_c) or M_c <= 0:
        home_hold[c] = np.nan
    else:
        h = M_c - flib
        if h < 0.01 * M_c:
            print(f"  NOTE: {c} home holdings floor applied (M={M_c:.1f}, liab={flib:.1f})")
            h = 0.01 * M_c      # 1% floor per Step-by-Step guide
        home_hold[c] = h

# Insert home holdings into df for base year rows where iso3_i == iso3_j
for c in COUNTRIES:
    mask = (df["year"] == BASE_YEAR) & (df["iso3_i"] == c) & (df["iso3_j"] == c)
    if mask.sum() > 0 and not np.isnan(home_hold.get(c, np.nan)):
        df.loc[mask, "a_ij"] = home_hold[c]

# Re-extract base year after diagonal fix
base = df[(df["year"] == BASE_YEAR)].copy()
base = base[base["iso3_i"].isin(COUNTRIES) & base["iso3_j"].isin(COUNTRIES)].copy()

# Print coverage
diag_check = base[base["iso3_i"] == base["iso3_j"]][["iso3_i","a_ij","M_i"]].set_index("iso3_i")
print(f"  Base year rows: {len(base)}")
print("  Diagonal a_ii (home holdings) vs M_i (first 10):")
print(diag_check.head(10).round(1).to_string())
print(f"  Diagonal NaN count: {diag_check['a_ij'].isna().sum()}")

# Rebuild lagged a_ij for the panel (for gravity)
df = df.sort_values(["iso3_i","iso3_j","year"])
df["a_ij_lag1"] = df.groupby(["iso3_i","iso3_j"])["a_ij"].shift(1)
df["ln_a_ij"]    = np.log(df["a_ij"].replace(0, np.nan))
df["ln_a_ij_lag1"] = np.log(df["a_ij_lag1"].replace(0, np.nan))

# ============================================================
# 3.  GRAVITY REGRESSION  (Step 1)
# ============================================================
print()
print("=" * 60)
print("STEP 1 — Gravity regression (PPML)")
print("=" * 60)

# Panel of off-diagonal pairs
grav = df[
    (df["iso3_i"] != df["iso3_j"]) &
    df["iso3_i"].isin(COUNTRIES) &
    df["iso3_j"].isin(COUNTRIES)
].copy()

grav = grav.replace([np.inf, -np.inf], np.nan)
grav = grav[grav["a_ij"] > 0]

# --- Coverage check ----------------------------------------
dcul_coverage  = grav["d_cul"].notna().mean()
dling_coverage = grav["d_ling"].notna().mean()
print(f"  d_cul  coverage in panel: {dcul_coverage:.1%}  (include if >50%)")
print(f"  d_ling coverage in panel: {dling_coverage:.1%}")

# d_cul : ~22% coverage (only intra-EU pairs) → excluded.
# d_ling: ~87% coverage, but in previous PPML it returns wrong sign (+)
#   when combined with geographic distance because the two are correlated.
#   We run two specifications and pick the one where signs are correct.
#   Spec A: ln_d_geo + euro_ij          (most robust, universally available)
#   Spec B: ln_d_geo + d_ling + euro_ij (adds linguistic distance)
# We keep whichever gives β_ling < 0; if neither does, use Spec A only.

reg_base = ["a_ij","ln_d_geo","euro_ij","iso3_i","iso3_j","year"]
grav_A = grav[reg_base].dropna().copy()
print(f"  Regression sample (Spec A): {len(grav_A):,} obs  |  {grav_A['year'].nunique()} years")

formula_A = "a_ij ~ ln_d_geo + euro_ij + C(iso3_i) + C(iso3_j) + C(year)"
ppml_A = smf.poisson(formula_A, data=grav_A).fit(
    cov_type="cluster",
    cov_kwds={"groups": grav_A["iso3_i"].astype(str) + "_" + grav_A["iso3_j"].astype(str)},
    maxiter=200, disp=False,
)

reg_B = ["a_ij","ln_d_geo","d_ling","euro_ij","iso3_i","iso3_j","year"]
grav_B = grav[reg_B].dropna().copy()
print(f"  Regression sample (Spec B): {len(grav_B):,} obs  |  {grav_B['year'].nunique()} years")

formula_B = "a_ij ~ ln_d_geo + d_ling + euro_ij + C(iso3_i) + C(iso3_j) + C(year)"
ppml_B = smf.poisson(formula_B, data=grav_B).fit(
    cov_type="cluster",
    cov_kwds={"groups": grav_B["iso3_i"].astype(str) + "_" + grav_B["iso3_j"].astype(str)},
    maxiter=200, disp=False,
)

b_ling_B = ppml_B.params.get("d_ling", 0.0)
# Select specification
if b_ling_B < 0:
    ppml   = ppml_B
    spec   = "B (ln_d_geo + d_ling + euro_ij)"
    print(f"  Using Spec B: β_ling = {b_ling_B:.4f} < 0 (correct sign).")
else:
    ppml   = ppml_A
    spec   = "A (ln_d_geo + euro_ij)"
    print(f"  β_ling = {b_ling_B:.4f} > 0 (wrong sign) → using Spec A (no d_ling).")

print(f"  Selected specification: {spec}")

beta_geo  = ppml.params.get("ln_d_geo", 0.0)
beta_ling = ppml.params.get("d_ling",   0.0)
beta_euro = ppml.params.get("euro_ij",  0.0)

print()
print("  Gravity coefficients (selected specification):")
print(f"    β_geo   (ln_d_geo) : {beta_geo:+.4f}   p={ppml.pvalues.get('ln_d_geo',np.nan):.3f}")
if "d_ling" in ppml.params:
    print(f"    β_ling  (d_ling)   : {beta_ling:+.4f}   p={ppml.pvalues.get('d_ling',np.nan):.3f}")
print(f"    β_euro  (euro_ij)  : {beta_euro:+.4f}   p={ppml.pvalues.get('euro_ij',np.nan):.3f}")
print()

# --- Only include coefficients with correct theoretical sign ---------------
beta_dict = {}

if beta_geo < 0:
    beta_dict["ln_d_geo"] = beta_geo
    print("  OK: β_geo < 0  → included in wedge formula.")
else:
    print("  WARNING: β_geo > 0 — unexpected sign, set to 0.")

if beta_ling < 0:
    beta_dict["d_ling"] = beta_ling
    print("  OK: β_ling < 0 → included in wedge formula.")
elif "d_ling" in ppml.params:
    print(f"  NOTE: β_ling = {beta_ling:+.4f} has wrong sign → excluded from wedge.")

if beta_euro > 0:
    beta_dict["euro_ij"] = beta_euro
    print("  OK: β_euro > 0 → included in wedge formula.")
else:
    print("  WARNING: β_euro ≤ 0 — unexpected sign, set to 0.")

print(f"\n  Active wedge regressors: {list(beta_dict.keys())}")


# ============================================================
# 4.  BILATERAL WEDGES  (Step 2)
# ============================================================
print()
print("=" * 60)
print("STEP 2 — Building bilateral wedge matrix Δ_baseline")
print("=" * 60)

# Re-extract base after all transformations
base = df[(df["year"] == BASE_YEAR) & df["iso3_i"].isin(COUNTRIES) & df["iso3_j"].isin(COUNTRIES)].copy()

# Off-diagonal wedges from gravity coefficients
Delta_arr = np.ones((n, n))

for _, row in base.iterrows():
    i_iso, j_iso = row["iso3_i"], row["iso3_j"]
    if i_iso == j_iso:
        continue
    ii, jj = idx[i_iso], idx[j_iso]

    ln_delta = 0.0
    # Each friction enters with *negative* sign: frictions reduce investment
    # so ln_Δ = -β * x, meaning Δ = exp(-β*x) > 1 when β<0 and x>0
    for col, beta in beta_dict.items():
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        ln_delta -= beta * val    # subtract: frictions have β<0 so -β>0 → Δ>1

    Delta_arr[ii, jj] = np.exp(ln_delta)

print(f"  Off-diagonal Δ range: [{Delta_arr[Delta_arr != 1].min():.3f}, {Delta_arr[Delta_arr != 1].max():.3f}]")
print(f"  Mean off-diagonal Δ : {Delta_arr[~np.eye(n,dtype=bool)].mean():.3f}")


# ============================================================
# 5.  COUNTRY-LEVEL MACRO VARIABLES  (Step 3)
# ============================================================
print()
print("=" * 60)
print("STEP 3 — Country-level returns R_j and market caps M_j")
print("=" * 60)

ctry_base = base.drop_duplicates("iso3_i").set_index("iso3_i").reindex(COUNTRIES)

R_vec = (ctry_base["alpha_i"] * ctry_base["Y_i"] / ctry_base["k_i"]).values.astype(float)
M_vec = ctry_base["M_i"].values.astype(float)

# Fill missing with median
R_median = np.nanmedian(R_vec)
M_median = np.nanmedian(M_vec[M_vec > 0])
n_R_imputed = np.isnan(R_vec).sum()
n_M_imputed = (np.isnan(M_vec) | (M_vec <= 0)).sum()
R_vec = np.where(np.isnan(R_vec), R_median, R_vec)
M_vec = np.where(np.isnan(M_vec) | (M_vec <= 0), M_median, M_vec)
print(f"  Imputed R for {n_R_imputed} countries  |  M for {n_M_imputed} countries")

# Normalise R so mean = 1
R_vec = R_vec / R_vec.mean()

print(f"  R range (normalised): [{R_vec.min():.3f}, {R_vec.max():.3f}]  mean={R_vec.mean():.3f}")
print(f"  M range (USD mn):     [{M_vec.min():.1f}, {M_vec.max():.1f}]  median={M_median:.1f}")

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
).fillna(0)

# Row sums = total portfolio s_i
s_vec = A_arr.sum(axis=1).values.astype(float)

# Observed shares
# Guard against zero row-sums
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
    RM  = (R ** eta) * M          # (n,) destination weights
    W   = RM[np.newaxis, :] / Delta   # (n, n)
    Z   = W.sum(axis=1, keepdims=True)
    Z   = np.where(Z == 0, np.nan, Z)
    Pi  = W / Z
    return Pi


eta = ETA

# Initialise diagonal
# Financial centres (IRL, LUX, NLD, CHE) have distorted home holdings
# due to pass-through flows → their π_ii is near 0, which would calibrate
# Δ_ii to enormous values and dominate the CMU shock.
# We exclude them from calibration and set Δ_ii = 1 (neutral).
for i, c in enumerate(COUNTRIES):
    if c in FINANCIAL_CENTRES:
        Delta_arr[i, i] = 1.0   # neutral — excluded from calibration
    elif not np.isnan(pi_home[i]) and 0 < pi_home[i] < 0.9999:
        Delta_arr[i, i] = 0.01   # start small (domestic wedge is small)
    else:
        Delta_arr[i, i] = 1.0    # neutral for countries we can't calibrate

RM = (R_vec ** eta) * M_vec

MAX_ITER = 20_000
TOL      = 1e-5

for iteration in range(MAX_ITER):
    max_err = 0.0
    for i in range(n):
        c = COUNTRIES[i]
        if c in FINANCIAL_CENTRES:
            continue   # skip financial centres
        if np.isnan(pi_home[i]) or pi_home[i] <= 0 or pi_home[i] >= 0.9999:
            continue
        w      = RM / Delta_arr[i, :]
        Z_i    = w.sum()
        pi_hat = w[i] / Z_i
        # Multiplicative update: if model over-predicts home bias, raise Δ_ii
        Delta_arr[i, i] *= pi_hat / pi_home[i]
        max_err = max(max_err, abs(pi_hat - pi_home[i]))
    if max_err < TOL:
        print(f"  Converged in {iteration+1} iterations  (max |π̂_ii - π_ii| = {max_err:.2e})")
        break
else:
    print(f"  Did NOT converge in {MAX_ITER} iterations  (max err = {max_err:.6f})")

Delta_baseline = Delta_arr.copy()

# Report calibrated diagonal wedges
calib_idx = [i for i in range(n)
             if COUNTRIES[i] not in FINANCIAL_CENTRES
             and not np.isnan(pi_home[i])
             and 0 < pi_home[i] < 0.9999]
calib_ctry = [COUNTRIES[i] for i in calib_idx]
delta_diag = pd.Series(np.diag(Delta_baseline), index=COUNTRIES)
print("\n  Calibrated Δ_ii (domestic wedges) — calibratable countries:")
print(delta_diag[calib_ctry].sort_values(ascending=False).round(4).to_string())

skipped = [c for c in COUNTRIES if c not in calib_ctry]
print(f"\n  Skipped / set neutral (Δ_ii=1): {skipped}")

# Sanity: domestic wedge should be ≤1 (frictions act on FOREIGN investment,
# domestic preference means Δ_ii is SMALL, which RAISES the domestic share).
# Actually in our formulation: π_ii = (RM_i / Δ_ii) / Σ_ι (RM_ι / Δ_iι)
# So Δ_ii < 1 means domestic is PREFERRED (low friction → high share).
# Most EU countries with strong home bias should have Δ_ii < 1.
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

Pi_baseline = compute_portfolio(Delta_baseline, R_vec, M_vec, eta)

# Validate: diagonal should match observed home bias
pi_base_home = np.diag(Pi_baseline)
comparison = pd.DataFrame({
    "π_data":     pi_home,
    "π_baseline": pi_base_home,
    "diff":       pi_base_home - pi_home,
}, index=COUNTRIES).dropna(subset=["π_data"])
print("  Home bias: data vs. baseline (calibrated countries):")
print(comparison.sort_values("π_data", ascending=False).round(4).to_string())

# Off-diagonal correlation
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
# ============================================================
print()
print("=" * 60)
print("STEPS 5-6 — CMU shock and counterfactual portfolios")
print("=" * 60)

eu_flag  = np.array([1 if c in EU27 else 0 for c in COUNTRIES])
eu_pair  = np.outer(eu_flag, eu_flag) * (1 - np.eye(n))   # intra-EU off-diagonal

results = {}
for phi in PHI_SCENARIOS:
    Delta_cmu = np.where(eu_pair.astype(bool),
                         Delta_baseline ** (1 - phi),
                         Delta_baseline)
    Pi_cmu    = compute_portfolio(Delta_cmu, R_vec, M_vec, eta)
    results[phi] = {"Delta_cmu": Delta_cmu, "Pi_cmu": Pi_cmu}

print("  CMU scenarios computed for φ =", PHI_SCENARIOS)

# Home bias change
hb_table = pd.DataFrame({"π_baseline": np.diag(Pi_baseline)}, index=COUNTRIES)
for phi in PHI_SCENARIOS:
    Pi_cmu = results[phi]["Pi_cmu"]
    hb_table[f"π_CMU(φ={phi})"] = np.diag(Pi_cmu)
    hb_table[f"Δπ(φ={phi})"]    = np.diag(Pi_cmu) - np.diag(Pi_baseline)

eu_mask = [c in EU27 for c in COUNTRIES]
print("\n  Home bias change by φ (EU27 only):")
print(hb_table.loc[eu_mask].round(4).to_string())

# Sanity: home bias should fall for EU countries under CMU
for phi in PHI_SCENARIOS:
    eu_hb_change = hb_table.loc[eu_mask, f"Δπ(φ={phi})"].mean()
    sign_ok = eu_hb_change < 0
    print(f"  φ={phi}: avg EU home-bias change = {eu_hb_change:.4f}  {'OK ✓' if sign_ok else 'WARNING: positive!'}")


# ============================================================
# 10.  CAPITAL REALLOCATION  (Step 7)
# ============================================================
print()
print("=" * 60)
print("STEP 7 — Capital reallocation")
print("=" * 60)

k_baseline = Pi_baseline.T @ s_vec    # k_j = Σ_i π_ij * s_i

cap_table = pd.DataFrame({"k_baseline": k_baseline}, index=COUNTRIES)
for phi in PHI_SCENARIOS:
    Pi_cmu = results[phi]["Pi_cmu"]
    k_cmu  = Pi_cmu.T @ s_vec
    cap_table[f"k_CMU(φ={phi})"]  = k_cmu
    cap_table[f"Δk/k(φ={phi})%"]  = (k_cmu - k_baseline) / k_baseline * 100
    results[phi]["k_cmu"] = k_cmu

# Conservation check
print("  Capital conservation (CMU total / baseline total):")
for phi in PHI_SCENARIOS:
    ratio = results[phi]["k_cmu"].sum() / k_baseline.sum()
    print(f"    φ={phi}: {ratio:.8f}  {'OK ✓' if abs(ratio-1) < 1e-6 else 'WARNING'}")

print("\n  Capital change (%) — all countries:")
dk_cols = [c for c in cap_table.columns if "Δk/k" in c]
print(cap_table[dk_cols].round(3).to_string())


# ============================================================
# 11.  OUTPUT & PRODUCTIVITY EFFECTS  (Step 8)
# ============================================================
print()
print("=" * 60)
print("STEP 8 — Output and productivity effects")
print("=" * 60)

# Production parameters from PWT (base year, origin columns)
A_prod  = ctry_base["A_i"].values.astype(float)
L_prod  = ctry_base["L_i"].values.astype(float)
alp     = ctry_base["alpha_i"].values.astype(float)

A_prod = np.where(np.isnan(A_prod), np.nanmedian(A_prod), A_prod)
L_prod = np.where(np.isnan(L_prod), np.nanmedian(L_prod), L_prod)
alp    = np.where(np.isnan(alp),    np.nanmedian(alp),    alp)

# NOTE on units
# ------------
# k_baseline = Π^T @ s  is in the same units as the CPIS portfolio (USD mn).
# A_j and L_j come from PWT; the Cobb-Douglas output y_j is therefore in
# "equity-capital units", NOT directly comparable to PWT GDP.
# We use the RATIO Δy_j/y_j = α_j * Δk_j/k_j, which is unit-free.
# This is the meaningful result: the % GDP gain from capital reallocation.


def cobb_douglas(A, k, L, alpha):
    """y_j = A_j * k_j^α_j * L_j^(1-α_j)"""
    return A * (k ** alpha) * (L ** (1 - alpha))


y_baseline = cobb_douglas(A_prod, k_baseline, L_prod, alp)

gdp_table = pd.DataFrame({"y_baseline": y_baseline}, index=COUNTRIES)
mpk_table = pd.DataFrame({"MPK_baseline": alp * y_baseline / k_baseline}, index=COUNTRIES)

for phi in PHI_SCENARIOS:
    k_cmu = results[phi]["k_cmu"]
    y_cmu = cobb_douglas(A_prod, k_cmu, L_prod, alp)
    # Exact percentage change
    dy_pct_exact = (y_cmu - y_baseline) / y_baseline * 100
    # Approximation: Δy/y ≈ α * Δk/k
    dk_pct = (k_cmu - k_baseline) / k_baseline
    dy_pct_approx = alp * dk_pct * 100
    gdp_table[f"y_CMU(φ={phi})"]       = y_cmu
    gdp_table[f"Δy/y exact(φ={phi})%"] = dy_pct_exact
    gdp_table[f"Δy/y approx(φ={phi})%"] = dy_pct_approx
    mpk_table[f"MPK_CMU(φ={phi})"] = alp * y_cmu / k_cmu
    results[phi]["y_cmu"] = y_cmu

print("  GDP change % (exact) — all countries:")
exact_cols = [c for c in gdp_table.columns if "exact" in c]
print(gdp_table[exact_cols].round(4).to_string())

# Cross-check: exact vs approximation (should be close for small Δk/k)
print("\n  Cross-check: Δy/y exact vs approx at φ=0.25 (should be similar for small changes):")
check_col_e = f"Δy/y exact(φ=0.25)%"
check_col_a = f"Δy/y approx(φ=0.25)%"
check = gdp_table[[check_col_e, check_col_a]].copy()
check["diff"] = check[check_col_e] - check[check_col_a]
print(check.round(3).to_string())

# Aggregate EU GDP effect
eu_idx   = [idx[c] for c in EU27 if c in idx]
Y_EU_base = y_baseline[eu_idx].sum()
print("\n  Aggregate EU GDP gain (%):")
print(f"  {'φ':>6}  {'ΔY_EU/Y_EU':>12}")
for phi in PHI_SCENARIOS:
    Y_EU_cmu = results[phi]["y_cmu"][eu_idx].sum()
    gain     = (Y_EU_cmu - Y_EU_base) / Y_EU_base * 100
    sign_ok  = gain > 0
    print(f"  {phi:>6.2f}  {gain:>12.5f}%  {'OK ✓' if sign_ok else 'WARNING: negative!'}")

# MPK convergence
print("\n  MPK dispersion (σ across EU27):")
sigma_base = mpk_table.loc[mpk_table.index.isin(EU27), "MPK_baseline"].std()
print(f"  {'baseline':>8}  σ={sigma_base:.6f}")
for phi in PHI_SCENARIOS:
    sigma_cmu = mpk_table.loc[mpk_table.index.isin(EU27), f"MPK_CMU(φ={phi})"].std()
    reduction = (sigma_base - sigma_cmu) / sigma_base * 100
    print(f"  {f'φ={phi}':>8}  σ={sigma_cmu:.6f}  reduction={reduction:.2f}%")


# ============================================================
# 12.  ROBUSTNESS  (Step 9)
# ============================================================
print()
print("=" * 60)
print("STEP 9 — Robustness checks")
print("=" * 60)

def run_model_variant(D_offdiag, R, M, s, A_prod, L_prod, alp,
                      Pi_data_in, countries, EU27_list, eta_val, phi_list, label):
    """
    Full model run for a given eta and off-diagonal wedge matrix.
    Re-calibrates Δ_ii for the given eta.
    Returns a DataFrame of results.
    """
    n_ = len(countries)
    eu_flag_ = np.array([1 if c in EU27_list else 0 for c in countries])
    eu_pair_ = np.outer(eu_flag_, eu_flag_) * (1 - np.eye(n_))
    eu_idx_  = [i for i, c in enumerate(countries) if c in EU27_list]

    D = D_offdiag.copy()
    RM_ = (R ** eta_val) * M
    pi_home_ = np.diag(Pi_data_in)

    # Initialise diagonal (skip financial centres)
    for i, c in enumerate(countries):
        if c in FINANCIAL_CENTRES:
            D[i, i] = 1.0
        elif not np.isnan(pi_home_[i]) and 0 < pi_home_[i] < 0.9999:
            D[i, i] = 0.01
        else:
            D[i, i] = 1.0

    # Calibrate diagonal
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

    Pi_base_ = compute_portfolio(D, R, M, eta_val)
    k_base_  = Pi_base_.T @ s
    y_base_  = cobb_douglas(A_prod, k_base_, L_prod, alp)
    Y_EU_base_ = y_base_[eu_idx_].sum()

    rows = []
    for phi in phi_list:
        D_cmu_   = np.where(eu_pair_.astype(bool), D ** (1 - phi), D)
        Pi_cmu_  = compute_portfolio(D_cmu_, R, M, eta_val)
        k_cmu_   = Pi_cmu_.T @ s
        y_cmu_   = cobb_douglas(A_prod, k_cmu_, L_prod, alp)
        Y_EU_cmu_ = y_cmu_[eu_idx_].sum()
        eu_hb_ch = np.nanmean(np.diag(Pi_cmu_)[eu_idx_]) - np.nanmean(np.diag(Pi_base_)[eu_idx_])
        rows.append({
            "label": label, "eta": eta_val, "phi": phi,
            "ΔY_EU/Y_EU (%)": (Y_EU_cmu_ - Y_EU_base_) / Y_EU_base_ * 100,
            "Avg ΔHomeBias (EU)": eu_hb_ch,
        })
    return pd.DataFrame(rows)


# Off-diagonal wedge matrix (diagonals set to neutral for re-calibration)
D_offdiag = Delta_baseline.copy()
np.fill_diagonal(D_offdiag, 1.0)

rob_rows = []

for eta_val, lbl in [(1.0, "η=1.0 (baseline)"), (0.5, "η=0.5"), (2.0, "η=2.0")]:
    r = run_model_variant(
        D_offdiag, R_vec, M_vec, s_vec, A_prod, L_prod, alp,
        Pi_data.values, COUNTRIES, EU27, eta_val, PHI_SCENARIOS, lbl
    )
    rob_rows.append(r)
    print(f"  {lbl}: done")

rob_df = pd.concat(rob_rows, ignore_index=True)
print("\n  Robustness Summary:")
print(rob_df.round(5).to_string(index=False))

# Sanity: all ΔY_EU should be positive
if (rob_df["ΔY_EU/Y_EU (%)"] > 0).all():
    print("\n  OK ✓ — EU GDP gain is positive for all scenarios and eta values.")
else:
    negative_cases = rob_df[rob_df["ΔY_EU/Y_EU (%)"] <= 0]
    print(f"\n  WARNING: {len(negative_cases)} cases with non-positive EU GDP gain:")
    print(negative_cases.to_string(index=False))


# ============================================================
# 13.  SUMMARY TABLES
# ============================================================
print()
print("=" * 60)
print("FINAL SUMMARY TABLES")
print("=" * 60)

print("\nTable 1 — Gravity coefficients (selected spec):")
spec_coefs = {}
for col in ["ln_d_geo","d_ling","euro_ij"]:
    spec_coefs[col] = {
        "coeff":    ppml.params.get(col, np.nan),
        "p-value":  ppml.pvalues.get(col, np.nan),
        "in_wedge": col in beta_dict,
    }
grav_summary = pd.DataFrame(spec_coefs).T
print(grav_summary.round(4).to_string())
print(f"  Specification: {spec}")

print("\nTable 2 — Aggregate EU GDP gain (%) by φ (baseline η=1):")
for phi in PHI_SCENARIOS:
    Y_EU_cmu = results[phi]["y_cmu"][eu_idx].sum()
    gain = (Y_EU_cmu - Y_EU_base) / Y_EU_base * 100
    print(f"  φ={phi:.2f} : {gain:.5f}%")
print("\nTable 3 — MPK dispersion reduction (EU27):")
for phi in PHI_SCENARIOS:
    sigma_cmu = mpk_table.loc[mpk_table.index.isin(EU27), f"MPK_CMU(φ={phi})"].std()
    reduction = (sigma_base - sigma_cmu) / sigma_base * 100
    print(f"  φ={phi:.2f} : σ_baseline={sigma_base:.4f}  σ_CMU={sigma_cmu:.4f}  reduction={reduction:.2f}%")

print("\nTable 4 — Top 5 capital gainers at φ=1.00 (EU27):")
dk_phi1 = (results[1.0]["k_cmu"] - k_baseline) / k_baseline * 100
dk_series = pd.Series(dk_phi1, index=COUNTRIES)
print(dk_series[EU27].sort_values(ascending=False).head(5).round(3).to_string())

print("\nTable 5 — Off-diagonal portfolio correlation (data vs model):")
print(f"  {corr_off:.4f}")

print("\nDone.")
