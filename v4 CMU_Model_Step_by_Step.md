# CMU Simulation Model: Step-by-Step Execution Guide

---

## Notation

| Symbol | Meaning |
|---|---|
| $i$ | Origin country (where the investor lives) |
| $j$ | Destination country (where capital goes) |
| $n$ | Total number of countries in the model |
| $a_{ij}$ | USD value of equity held by country $i$'s investors in country $j$'s assets |
| $\pi_{ij}$ | Share of country $i$'s total equity portfolio allocated to country $j$ |
| $s_i$ | Country $i$'s total outward portfolio equity holdings (sum across all destinations) |
| $k_j$ | Total equity capital that country $j$ receives from all investors |
| $M_j$ | Stock market capitalisation of country $j$ |
| $R_j$ | Expected return on equity in country $j$ |
| $\Delta_{ij}$ | Portfolio wedge (friction) between origin $i$ and destination $j$ |

---

## Step 0: Assemble the Data

You need five datasets. Organise everything into matrices and vectors indexed by your $n$ countries.****

### 0.1 — Choose your country set

Pick $n$ countries. A practical choice:

- All 27 EU member states
- Plus 4 outside options: United States, United Kingdom, Switzerland, Norway

This gives $n = 31$. Label them $1, 2, \ldots, 31$.

### 0.2 — Download bilateral equity holdings

Source: IMF Coordinated Portfolio Investment Survey (CPIS), available at `data.imf.org`.

Download the bilateral table for **Equity and Investment Fund Shares**, most recent year (e.g. end-2023). This gives you an $n \times n$ matrix where row $i$, column $j$ = equity issued by country $j$ held by investors in country $i$.

Store this as matrix $\mathbf{A}$ where entry $A_{ij} = a_{ij}$ (in millions of USD).

**Set diagonal entries:** The CPIS does not report domestic holdings. You need domestic equity holdings $a_{ii}$ for each country. Compute them as:

$$
a_{ii} = M_i - \sum_{j \neq i} a_{ji}^{\text{liabilities}}
$$

where $M_i$ is country $i$'s stock market capitalisation and $\sum_{j \neq i} a_{ji}^{\text{liabilities}}$ is total foreign equity investment *into* $i$ (sum of column $i$ excluding the diagonal). This says: domestic investors hold whatever part of the market that foreigners do not.

If this gives a negative number (unlikely but possible for very open financial centres), set $a_{ii} = 0.01 \cdot M_i$ as a floor and flag it.

### 0.3 — Compute observed portfolio shares

For each origin country $i$, compute total outward equity holdings:

$$
s_i = \sum_{j=1}^{n} a_{ij}
$$

Then compute observed portfolio shares:

$$
\pi_{ij}^{\text{data}} = \frac{a_{ij}}{s_i}
$$

You now have an $n \times n$ matrix $\boldsymbol{\Pi}^{\text{data}}$ where each row $i$ sums to 1. The diagonal entry $\pi_{ii}^{\text{data}}$ is the **home bias** of country $i$.

### 0.4 — Download bilateral distances

Source: `geopoliticaldistance.org` (the Pellegrino, Spolaore & Wacziarg dataset).

For every pair $(i,j)$ where $i \neq j$, download:

- $d_{ij}^{\text{geo}}$ — geographic distance (km between capital cities)
- $d_{ij}^{\text{cul}}$ — cultural distance (continuous, based on World Values Survey)
- $d_{ij}^{\text{ling}}$ — linguistic distance (continuous, between 0 and 1)

Set $d_{ii}^{\text{geo}} = d_{ii}^{\text{cul}} = d_{ii}^{\text{ling}} = 0$ for all $i$.

Also construct:

- $\text{Euro}_{ij} = 1$ if both $i$ and $j$ are Eurozone members, 0 otherwise
- $\text{EU}_{ij} = 1$ if both $i$ and $j$ are EU members, 0 otherwise

### 0.5 — Download country-level variables

From the Penn World Table (version 10.01 or later, `rug.nl/ggdc/productivity/pwt`):

| Variable | PWT code | What it is |
|---|---|---|
| Real GDP | `rgdpna` | Output $y_j$ in constant national prices |
| Capital stock | `rkna` | Capital $k_j$ in constant national prices |
| Labour share | `labsh` | Labour's share of income |
| Employment | `emp` | Number of persons employed $L_j$ |

Compute:

$$
\alpha_j = 1 - \text{labsh}_j \qquad \text{(capital share)}
$$

$$
\text{MPK}_j^{\text{data}} = \alpha_j \cdot \frac{y_j}{k_j} \qquad \text{(marginal product of capital)}
$$

$$
A_j = \frac{y_j}{k_j^{\alpha_j} \cdot L_j^{1-\alpha_j}} \qquad \text{(total factor productivity, backed out)}
$$

From the World Federation of Exchanges or World Bank:

- $M_j$ — stock market capitalisation (USD) for each country

---

## Step 1: Estimate the Gravity Equation

### 1.1 — Prepare the regression dataset

Create a dataset with one row per bilateral pair $(i,j)$ where $i \neq j$. Each row has:

| Column | Value |
|---|---|
| `ln_a_ij` | $\ln(a_{ij})$ — log bilateral equity holdings |
| `ln_dist_geo` | $\ln(d_{ij}^{\text{geo}})$ — log geographic distance |
| `dist_cul` | $d_{ij}^{\text{cul}}$ — cultural distance (in levels) |
| `dist_ling` | $d_{ij}^{\text{ling}}$ — linguistic distance (in levels) |
| `euro` | $\text{Euro}_{ij}$ |
| `orig_fe` | Origin country fixed effect (factor variable for $i$) |
| `dest_fe` | Destination country fixed effect (factor variable for $j$) |

Drop rows where $a_{ij} = 0$ or is missing (or use PPML, see below).

### 1.2 — Run the regression

**OLS specification:**

$$
\ln a_{ij} = \alpha_i + \alpha_j + \beta_1 \ln d_{ij}^{\text{geo}} + \beta_2 \, d_{ij}^{\text{cul}} + \beta_3 \, d_{ij}^{\text{ling}} + \beta_4 \, \text{Euro}_{ij} + \varepsilon_{ij}
$$

In Stata:
```
reghdfe ln_a_ij ln_dist_geo dist_cul dist_ling euro, absorb(orig_fe dest_fe) cluster(pair_id)
```

In R:
```r
library(fixest)
feols(ln_a_ij ~ ln_dist_geo + dist_cul + dist_ling + euro | orig_fe + dest_fe, 
      data = df, cluster = ~pair_id)
```

In Python:
```python
import statsmodels.formula.api as smf
model = smf.ols('ln_a_ij ~ ln_dist_geo + dist_cul + dist_ling + euro + C(orig_fe) + C(dest_fe)', 
                data=df).fit(cov_type='cluster', cov_kwds={'groups': df['pair_id']})
```

**Alternative (preferred): PPML specification** to handle zeros:

$$
a_{ij} = \exp\!\Big(\alpha_i + \alpha_j + \beta_1 \ln d_{ij}^{\text{geo}} + \beta_2 \, d_{ij}^{\text{cul}} + \beta_3 \, d_{ij}^{\text{ling}} + \beta_4 \, \text{Euro}_{ij}\Big) \cdot \eta_{ij}
$$

In Stata:
```
ppmlhdfe a_ij ln_dist_geo dist_cul dist_ling euro, absorb(orig_fe dest_fe) cluster(pair_id)
```

### 1.3 — Store the estimated coefficients

Record:

$$
\hat{\beta}_1, \quad \hat{\beta}_2, \quad \hat{\beta}_3, \quad \hat{\beta}_4
$$

All four should be reported with standard errors and significance levels.

**Expected signs:**
- $\hat{\beta}_1 < 0$ (more geographic distance → less investment)
- $\hat{\beta}_2 < 0$ (more cultural distance → less investment)
- $\hat{\beta}_3 < 0$ (more linguistic distance → less investment)
- $\hat{\beta}_4 > 0$ (sharing the euro → more investment)

**These coefficients are the inputs to Step 2.** They measure how strongly each soft barrier reduces bilateral investment.

---

## Step 2: Construct the Baseline Wedge Matrix

### 2.1 — Compute bilateral wedges for all $i \neq j$

For every pair $(i,j)$ where $i \neq j$, compute:

$$
\ln \Delta_{ij}^{\text{baseline}} = \underbrace{(-\hat{\beta}_1)}_{\text{positive}} \cdot \ln d_{ij}^{\text{geo}} + \underbrace{(-\hat{\beta}_2)}_{\text{positive}} \cdot d_{ij}^{\text{cul}} + \underbrace{(-\hat{\beta}_3)}_{\text{positive}} \cdot d_{ij}^{\text{ling}} + \underbrace{(-\hat{\beta}_4)}_{\text{negative}} \cdot \text{Euro}_{ij}
$$

Then exponentiate:

$$
\Delta_{ij}^{\text{baseline}} = \exp\!\Big(\ln \Delta_{ij}^{\text{baseline}}\Big)
$$

**Interpretation:** $\Delta_{ij} > 1$ means there is a friction. Larger $\Delta_{ij}$ = larger barrier. For two countries that are geographically close, culturally similar, speak the same language, and share the euro, $\Delta_{ij}$ will be small (close to 1). For distant, dissimilar pairs, $\Delta_{ij}$ will be large.

### 2.2 — Calibrate the domestic wedge $\Delta_{ii}$

The domestic wedge is not estimated from the gravity regression (which excluded $i=j$ observations). Instead, calibrate it to match the observed home bias.

The model says:

$$
\pi_{ii} = \frac{R_i^{\eta} \cdot M_i \,/\, \Delta_{ii}}{\displaystyle \sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}}
$$

You know $\pi_{ii}^{\text{data}}$ from Step 0.3. Rearrange to solve for $\Delta_{ii}$:

$$
\Delta_{ii} = \frac{R_i^{\eta} \cdot M_i}{\pi_{ii}^{\text{data}} \cdot \displaystyle\sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}}
$$

This is circular (the sum in the denominator includes $\Delta_{ii}$ itself). Solve it iteratively:

> **Algorithm to calibrate $\Delta_{ii}$:**
>
> 1. Set initial guess: $\Delta_{ii}^{(0)} = 0.01$ for all $i$ (domestic friction is small).
> 2. For each $i$, compute the denominator $Z_i = \sum_{\iota} R_\iota^{\eta} \cdot M_\iota / \Delta_{i\iota}$, using current $\Delta_{ii}$.
> 3. Compute the model-implied home share: $\hat{\pi}_{ii} = (R_i^{\eta} \cdot M_i / \Delta_{ii}) \,/\, Z_i$.
> 4. Update: $\Delta_{ii}^{(\text{new})} = \Delta_{ii}^{(\text{old})} \cdot \hat{\pi}_{ii} \,/\, \pi_{ii}^{\text{data}}$.
> 5. Repeat steps 2–4 until $|\hat{\pi}_{ii} - \pi_{ii}^{\text{data}}| < 0.001$ for all $i$.
>
> This typically converges in 10–20 iterations.

After convergence, you have the full $n \times n$ wedge matrix $\boldsymbol{\Delta}^{\text{baseline}}$.

---

## Step 3: Choose Return Elasticity and Returns

### 3.1 — Set the return elasticity

$$
\eta = 1 \qquad \text{(baseline)}
$$

You will redo Steps 4–7 with $\eta = 0.5$ and $\eta = 2$ as robustness checks.

### 3.2 — Compute expected returns

Use the marginal product of capital from PWT:

$$
R_j = \text{MPK}_j^{\text{data}} = \alpha_j \cdot \frac{y_j}{k_j}
$$

Normalise so that the average $R_j$ across countries equals 1 (this is just a scaling convenience — it doesn't affect portfolio shares because they are ratios):

$$
R_j \leftarrow \frac{R_j}{\bar{R}} \qquad \text{where} \quad \bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j
$$

---

## Step 4: Compute Baseline Portfolio Shares

For every $(i,j)$ pair:

$$
\pi_{ij}^{\text{baseline}} = \frac{R_j^{\eta} \cdot M_j \,/\, \Delta_{ij}^{\text{baseline}}}{\displaystyle\sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}^{\text{baseline}}}
$$

**Concrete computation for a single origin country $i$:**

1. Compute the "attractiveness" of each destination $\iota$ for investor $i$:

$$
w_{i\iota} = \frac{R_\iota^{\eta} \cdot M_\iota}{\Delta_{i\iota}^{\text{baseline}}}
$$

2. Sum them up:

$$
Z_i = \sum_{\iota=1}^{n} w_{i\iota}
$$

3. The portfolio share is:

$$
\pi_{i\iota}^{\text{baseline}} = \frac{w_{i\iota}}{Z_i}
$$

Do this for every $i = 1, \ldots, n$. This gives you the $n \times n$ matrix $\boldsymbol{\Pi}^{\text{baseline}}$.

**Validation:** Compare $\boldsymbol{\Pi}^{\text{baseline}}$ to $\boldsymbol{\Pi}^{\text{data}}$. The diagonal entries should match exactly (by construction, since you calibrated $\Delta_{ii}$). The off-diagonal entries should be approximately correct. Compute the correlation between $\pi_{ij}^{\text{baseline}}$ and $\pi_{ij}^{\text{data}}$ across all $i \neq j$ pairs. Report this correlation — it measures model fit.

---

## Step 5: Define the CMU Shock

### 5.1 — Choose the integration parameter $\phi$

Run the model for $\phi \in \{0.25,\; 0.50,\; 0.75,\; 1.00\}$.

### 5.2 — Construct the counterfactual wedge matrix

For every pair $(i,j)$:

$$
\Delta_{ij}^{\text{CMU}} = \begin{cases}
\left(\Delta_{ij}^{\text{baseline}}\right)^{1-\phi} & \text{if } i \neq j \text{ and both } i,j \in \text{EU} \\[6pt]
\Delta_{ij}^{\text{baseline}} & \text{otherwise (including } i = j\text{)}
\end{cases}
$$

**What this does:** Raising a wedge to the power $(1-\phi)$ shrinks it toward 1. When $\phi = 1$, the exponent is 0, so $\Delta_{ij}^{\text{CMU}} = 1$ (no friction). When $\phi = 0.5$, the wedge is square-rooted (halved in log terms).

**Example:** Suppose $\Delta_{\text{Germany,Poland}}^{\text{baseline}} = 4.0$ (Germany–Poland friction is 4 times the frictionless level).

| $\phi$ | $\Delta_{\text{Germany,Poland}}^{\text{CMU}}$ | Interpretation |
|---|---|---|
| 0.00 | $4.0^{1.0} = 4.00$ | No change |
| 0.25 | $4.0^{0.75} = 2.83$ | CMU removes 25% of barriers |
| 0.50 | $4.0^{0.50} = 2.00$ | CMU removes 50% of barriers |
| 1.00 | $4.0^{0.00} = 1.00$ | Full integration |

**Domestic wedges $\Delta_{ii}$ stay unchanged.** The CMU does not make people like their own market less — it makes foreign EU markets more accessible.

**Non-EU pairs stay unchanged.** The barrier between, say, France and the US is not affected by the CMU.

---

## Step 6: Compute Counterfactual Portfolios

Repeat the exact same calculation as Step 4, but using $\boldsymbol{\Delta}^{\text{CMU}}$ instead of $\boldsymbol{\Delta}^{\text{baseline}}$.

For every $(i,j)$:

$$
w_{i\iota}^{\text{CMU}} = \frac{R_\iota^{\eta} \cdot M_\iota}{\Delta_{i\iota}^{\text{CMU}}}
$$

$$
Z_i^{\text{CMU}} = \sum_{\iota=1}^{n} w_{i\iota}^{\text{CMU}}
$$

$$
\pi_{i\iota}^{\text{CMU}} = \frac{w_{i\iota}^{\text{CMU}}}{Z_i^{\text{CMU}}}
$$

### 6.1 — Compute portfolio changes

For every pair:

$$
\Delta\pi_{ij} = \pi_{ij}^{\text{CMU}} - \pi_{ij}^{\text{baseline}}
$$

For home bias specifically:

$$
\Delta\text{HomeBias}_i = \pi_{ii}^{\text{CMU}} - \pi_{ii}^{\text{baseline}}
$$

This should be **negative** for all EU countries (home bias falls).

---

## Step 7: Compute Capital Reallocation

### 7.1 — Compute capital received by each country under both scenarios

$$
k_j^{\text{baseline}} = \sum_{i=1}^{n} \pi_{ij}^{\text{baseline}} \cdot s_i
$$

$$
k_j^{\text{CMU}} = \sum_{i=1}^{n} \pi_{ij}^{\text{CMU}} \cdot s_i
$$

where $s_i = \sum_{j} a_{ij}$ is the total equity portfolio of country $i$ (computed in Step 0.3). This is held fixed — total savings do not change, only their allocation across destinations changes.

In matrix form:

$$
\mathbf{k}^{\text{baseline}} = \boldsymbol{\Pi}^{\text{baseline}\top} \cdot \mathbf{s}
\qquad \text{and} \qquad
\mathbf{k}^{\text{CMU}} = \boldsymbol{\Pi}^{\text{CMU}\top} \cdot \mathbf{s}
$$

### 7.2 — Compute percentage change in capital for each country

$$
\frac{\Delta k_j}{k_j} = \frac{k_j^{\text{CMU}} - k_j^{\text{baseline}}}{k_j^{\text{baseline}}}
$$

**Expected pattern:**

- Small, peripheral EU countries (e.g. Baltic states, Croatia, Bulgaria): $\Delta k_j / k_j > 0$ (capital inflows).
- Large EU countries with high existing home bias (e.g. Germany, France, Italy): $\Delta k_j / k_j$ could be slightly negative (some domestic capital now goes abroad) or slightly positive (they also receive more from other EU countries).
- Non-EU countries (US, UK): $\Delta k_j / k_j \leq 0$ (EU investors reallocate toward EU, away from outside).

**Check:** Total capital must be conserved. Verify that $\sum_j k_j^{\text{CMU}} = \sum_j k_j^{\text{baseline}} = \sum_i s_i$.

---

## Step 8: Compute Output and Productivity Effects

### 8.1 — Compute output under both scenarios

$$
y_j^{\text{baseline}} = A_j \cdot \left(k_j^{\text{baseline}}\right)^{\alpha_j} \cdot L_j^{1 - \alpha_j}
$$

$$
y_j^{\text{CMU}} = A_j \cdot \left(k_j^{\text{CMU}}\right)^{\alpha_j} \cdot L_j^{1 - \alpha_j}
$$

$A_j$ and $L_j$ do not change. Only $k_j$ changes.

### 8.2 — Compute percentage change in GDP for each country

$$
\frac{\Delta y_j}{y_j} = \frac{y_j^{\text{CMU}} - y_j^{\text{baseline}}}{y_j^{\text{baseline}}}
$$

Or equivalently, using the approximation:

$$
\frac{\Delta y_j}{y_j} \approx \alpha_j \cdot \frac{\Delta k_j}{k_j}
$$

Both methods should give nearly identical results for small changes. Use the exact formula and report the approximation as a check.

### 8.3 — Compute aggregate EU GDP effect

$$
\frac{\Delta Y^{\text{EU}}}{Y^{\text{EU}}} = \frac{\displaystyle\sum_{j \in \text{EU}} y_j^{\text{CMU}} - \sum_{j \in \text{EU}} y_j^{\text{baseline}}}{\displaystyle\sum_{j \in \text{EU}} y_j^{\text{baseline}}}
$$

### 8.4 — Compute MPK convergence

$$
\text{MPK}_j^{\text{CMU}} = \alpha_j \cdot \frac{y_j^{\text{CMU}}}{k_j^{\text{CMU}}}
$$

Then compute the cross-country dispersion:

$$
\sigma_{\text{MPK}}^{\text{baseline}} = \text{Std.Dev}\!\left(\text{MPK}_j^{\text{baseline}}\right)_{j \in \text{EU}}
$$

$$
\sigma_{\text{MPK}}^{\text{CMU}} = \text{Std.Dev}\!\left(\text{MPK}_j^{\text{CMU}}\right)_{j \in \text{EU}}
$$

$$
\text{Convergence} = \frac{\sigma_{\text{MPK}}^{\text{baseline}} - \sigma_{\text{MPK}}^{\text{CMU}}}{\sigma_{\text{MPK}}^{\text{baseline}}} \times 100\%
$$

A positive number means the CMU reduced the dispersion of returns — capital is more efficiently allocated.

---

## Step 9: Robustness Checks

Redo Steps 3–8 with the following variations:

| Check | What changes | Why |
|---|---|---|
| $\eta = 0.5$ | Return elasticity halved | Tests sensitivity to how much investors chase returns |
| $\eta = 2.0$ | Return elasticity doubled | Same |
| Exclude LU, IE, NL | Remove Luxembourg, Ireland, Netherlands from sample | These are financial centres that distort the data (Beck et al. 2024) |
| PPML gravity | Use PPML instead of OLS in Step 1 | Handles zeros; different $\hat{\beta}$ values flow through the entire model |
| Only soft barriers | Set $\phi$ to reduce only cultural + linguistic distance, not geographic | Tests whether the CMU effect comes from information barriers specifically |

---

## Flow Diagram: How the Steps Connect

```
STEP 0: Data assembly
  │
  ├── Bilateral holdings matrix A (CPIS)
  ├── Distance matrices (geopoliticaldistance.org)
  ├── Country-level data (PWT, market cap)
  │
  ▼
STEP 1: Gravity regression
  │
  │   Input:  A, distances
  │   Output: β̂₁, β̂₂, β̂₃, β̂₄
  │
  ▼
STEP 2: Build baseline wedge matrix
  │
  │   Input:  β̂ coefficients, distances, observed home shares
  │   Output: Δ_baseline (n × n matrix)
  │
  ▼
STEP 3: Set returns R_j and elasticity η
  │
  │   Input:  PWT data
  │   Output: R vector, η scalar
  │
  ▼
STEP 4: Compute baseline portfolios
  │
  │   Input:  R, M, Δ_baseline, η
  │   Output: Π_baseline (n × n matrix)
  │   Check:  Compare to Π_data
  │
  ▼
STEP 5: Define CMU shock
  │
  │   Input:  Δ_baseline, φ, EU membership
  │   Output: Δ_CMU (n × n matrix)
  │
  ▼
STEP 6: Compute counterfactual portfolios
  │
  │   Input:  R, M, Δ_CMU, η
  │   Output: Π_CMU (n × n matrix)
  │   Result: Change in home bias, change in cross-border shares
  │
  ▼
STEP 7: Capital reallocation
  │
  │   Input:  Π_baseline, Π_CMU, s (savings vector)
  │   Output: k_baseline, k_CMU for each country
  │   Result: Which countries gain/lose capital
  │
  ▼
STEP 8: Output and productivity effects
  │
  │   Input:  k_baseline, k_CMU, A, L, α (from PWT)
  │   Output: Δy_i/y_i for each country, aggregate EU GDP effect,
  │           MPK convergence measure
  │
  ▼
STEP 9: Robustness
  │
  │   Vary η, vary sample, vary gravity estimator
  │   Redo Steps 3–8 for each variation
```

---

## What To Report

### Tables

1. **Gravity results** (Step 1): Coefficient table with $\hat{\beta}_1$ through $\hat{\beta}_4$, standard errors, $R^2$.
2. **Baseline home bias** (Step 4): Column 1 = country, Column 2 = observed home share $\pi_{ii}^{\text{data}}$, Column 3 = model home share $\pi_{ii}^{\text{baseline}}$.
3. **CMU portfolio effects** (Step 6): For each country $i$: home share under baseline, home share under CMU ($\phi = 0.25, 0.50, 1.0$), change in home bias.
4. **Capital and GDP effects** (Steps 7–8): For each country $j$: $\Delta k_j / k_j$ and $\Delta y_j / y_j$ under each $\phi$ scenario.
5. **MPK convergence** (Step 8.4): $\sigma_{\text{MPK}}$ baseline vs. CMU, percentage reduction.

### Figures

1. **Map of home bias change**: Colour EU countries by $\Delta\pi_{ii}$ (how much home bias falls).
2. **Map of GDP gains**: Colour EU countries by $\Delta y_j / y_j$.
3. **Scatter: baseline MPK vs. capital gain**: Show that high-MPK (capital-scarce) countries gain the most capital. This is the convergence result.
4. **Bar chart**: Aggregate EU GDP gain as a function of $\phi$.
5. **Heatmap**: The $\Delta\pi_{ij}$ matrix for a selected $\phi$, showing which bilateral flows increase most.
