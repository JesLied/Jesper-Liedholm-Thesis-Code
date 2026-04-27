# CMU Simulation Model with Endogenous TFP: Step-by-Step Execution Guide

This guide documents `v4_simulation_endo_tfp.py`, which extends the baseline CMU simulation by adding a **productivity spillover channel**: when a country receives more foreign equity capital, knowledge and technology transfer raise its TFP.

The extension slots in between Step 7 (capital reallocation) and Step 8 (output effects):

- **Step 7b** — decompose capital into domestic / foreign components
- **Step 8** — output with endogenous TFP $A_i(\theta) = \bar{A}_i \cdot (1 + \theta \cdot f_i)$

where $f_i$ is foreign equity capital relative to the PWT physical capital stock, and $\theta$ is the TFP-spillover elasticity.

**References for θ:**
- Baltabaev (2014, *Empirical Economics*) — θ ≈ 0.10 (panel cointegration)
- Borensztein, De Gregorio & Lee (1998, *JIE*) — conservative θ ≈ 0.05
- Bau & Matray (2023, *Econometrica*) — capital liberalisation ↑ TFP 3–16%
- Javorcik (2004, *AER*) — micro evidence on FDI spillovers
- Findlay (1978, *JIE*) — technology-contagion theory

---

## Notation

| Symbol | Meaning |
|---|---|
| $i$ | Origin country (where the investor lives) |
| $j$ | Destination country (where capital goes) |
| $n$ | Total number of countries in the model (= 31) |
| $a_{ij}$ | USD value of equity held by country $i$'s investors in country $j$'s assets |
| $\pi_{ij}$ | Share of country $i$'s total equity portfolio allocated to country $j$ |
| $s_i$ | Country $i$'s total outward portfolio equity holdings (sum across all destinations) |
| $k_j$ | Total equity capital that country $j$ receives from all investors |
| $k_j^{\text{foreign}}$ | Foreign equity capital received by country $j$ (from all $i \neq j$) |
| $k_j^{\text{PWT}}$ | Country $j$'s physical capital stock from Penn World Tables |
| $f_j$ | Foreign-capital intensity: $k_j^{\text{foreign}} / k_j^{\text{PWT}}$ |
| $M_j$ | Stock market capitalisation of country $j$ |
| $R_j$ | Expected return on equity in country $j$ (normalised MPK) |
| $\eta$ | Return elasticity (baseline = 1.0) |
| $\Delta_{ij}^{\text{hard}}$ | Financial system distance barrier between $i$ and $j$ (hard / policy-reducible) |
| $\Delta_{ij}^{\text{soft}}$ | Linguistic distance barrier between $i$ and $j$ (soft / cultural) |
| $\Delta_{ij}$ | Combined portfolio wedge: $\Delta_{ij}^{\text{hard}} \cdot \Delta_{ij}^{\text{soft}}$ |
| $\omega$ | Financial (hard) integration intensity: $\omega \in \{0.00, 0.25, 0.50, 1.00\}$ |
| $\gamma$ | Language (soft) integration intensity: $\gamma \in \{0.00, 0.25, 0.50, 1.00\}$ |
| $\bar{A}_j$ | Baseline TFP of country $j$ (backed out from PWT data) |
| $A_j(\theta)$ | Endogenous TFP: $\bar{A}_j \cdot (1 + \theta \cdot f_j)$ |
| $\theta$ | TFP-spillover elasticity: $\theta \in \{0.00, 0.05, 0.10\}$ |
| $L_j$ | Employment in country $j$ (from PWT) |
| $\alpha_j$ | Capital share of income in country $j$ (= 1 − labour share) |

---

## Configuration

| Parameter | Value |
|---|---|
| `DATA_PATH` | `Data/Clean/Final-v4.csv` |
| `BASE_YEAR` | 2019 |
| `ETA` | 1.0 (baseline) |
| `PHI_SCENARIOS` (ω and γ) | [0.00, 0.25, 0.50, 1.00] |
| `THETA_SCENARIOS` | [0.00, 0.05, 0.10] |
| EU27 | AUT, BEL, BGR, HRV, CYP, CZE, DNK, EST, FIN, FRA, DEU, GRC, HUN, IRL, ITA, LVA, LTU, LUX, MLT, NLD, POL, PRT, ROU, SVK, SVN, ESP, SWE |
| OUTSIDE | USA, GBR, CHE, NOR |
| `FINANCIAL_CENTRES` | IRL, LUX, NLD, CHE — excluded from diagonal Δ_ii calibration |

---

## Step 0: Assemble the Data

### 0.1 — Choose your country set

The model uses $n = 31$ countries:

- All 27 EU member states (EU27)
- Plus 4 outside options: United States, United Kingdom, Switzerland, Norway

### 0.2 — Download bilateral equity holdings

Source: IMF **Portfolio Investment Positions (PIP)**, series `IMF.STA:PIP`, available at `data.imf.org`.

Download the bilateral table for **Equity and Investment Fund Shares** for the base year (2019). This gives an $n \times n$ matrix where row $i$, column $j$ = equity issued by country $j$ held by investors in country $i$.

Store this as matrix $\mathbf{A}$ where entry $A_{ij} = a_{ij}$ (in millions of USD).

**Set diagonal entries — home holdings:** The PIP does not report domestic holdings. Compute them as:

$$
a_{ii} = M_i - \sum_{j \neq i} a_{ji}^{\text{foreign liabilities}}
$$

where the sum is all foreign equity investment *into* country $i$. If this gives a value less than $0.01 \cdot M_i$ (e.g. for very open financial centres), apply a floor:

$$
a_{ii} = \max\!\left(M_i - \sum_{j \neq i} a_{ji}, \; 0.01 \cdot M_i\right)
$$

> **Note:** Countries in `FINANCIAL_CENTRES` (IRL, LUX, NLD, CHE) have domestic holdings distorted by pass-through flows. Their diagonal entries are set but their home bias is *not* used to calibrate $\Delta_{ii}$.

### 0.3 — Compute observed portfolio shares

For each origin country $i$, compute total outward equity holdings:

$$
s_i = \sum_{j=1}^{n} a_{ij}
$$

Then compute observed portfolio shares:

$$
\pi_{ij}^{\text{data}} = \frac{a_{ij}}{s_i}
$$

This gives an $n \times n$ matrix $\boldsymbol{\Pi}^{\text{data}}$ where each row $i$ sums to 1. The diagonal entry $\pi_{ii}^{\text{data}}$ is the **home bias** of country $i$.

### 0.4 — Download bilateral distances

For every pair $(i,j)$ where $i \neq j$, download:

- $d_{ij}^{\text{ling}}$ — linguistic distance (continuous, between 0 and 1)

Construct:

- $d_{ij}^{\text{gci\_fin}}$ — absolute difference in GCI Financial Composite score between $i$ and $j$:

$$
d_{ij}^{\text{gci\_fin}} = \left| \text{GCI\_fin}_i - \text{GCI\_fin}_j \right|
$$

> **Note on GCI data:** GCI data is only available up to ~2019. The last observed value per country is carried forward to fill any gaps in the base-year cross-section.

### 0.5 — Download country-level variables

From the Penn World Table (version 10.01 or later):

| Variable | PWT code | What it is |
|---|---|---|
| Real GDP | `rgdpna` | Output $y_j$ in constant national prices |
| Capital stock | `rkna` | Physical capital stock $k_j^{\text{PWT}}$ |
| Labour share | `labsh` | Labour's share of income |
| Employment | `emp` | Number of persons employed $L_j$ |
| TFP | `rtfpna` | Total factor productivity $\bar{A}_j$ |

Compute:

$$
\alpha_j = 1 - \text{labsh}_j \qquad \text{(capital share)}
$$

$$
R_j = \alpha_j \cdot \frac{y_j}{k_j} \qquad \text{(marginal product of capital — used as return proxy)}
$$

From the World Federation of Exchanges or World Bank:

- $M_j$ — stock market capitalisation (USD, in millions) for each country

---

## Step 1: Estimate the Gravity Equation (PPML)

### 1.1 — Prepare the regression dataset

Create a panel dataset with one row per bilateral pair $(i,j)$ and year, where $i \neq j$. Each row contains:

| Column | Value |
|---|---|
| `a_ij` | Bilateral equity holdings (USD millions) |
| `d_ling` | Linguistic distance (levels) — **soft barrier** |
| `d_gci_financial_composite` | GCI financial composite distance (levels) — **hard barrier** |
| `Y_i`, `Y_j` | GDP of origin and destination |
| `year` | Year (for time fixed effects) |

Scale monetary variables before estimation to prevent numerical overflow in the $Y_i \times Y_j$ interaction:

$$
\tilde{a}_{ij} = a_{ij} / 10^6, \qquad \tilde{Y}_i = Y_i / 10^6, \qquad \tilde{Y}_j = Y_j / 10^6
$$

The slope coefficients on the friction regressors (`d_ling`, `d_gci_financial_composite`) are invariant to this scaling.

### 1.2 — Run the PPML regression

The PPML specification is:

$$
a_{ij} = \exp\!\Big(\beta_{\text{ling}} \cdot d_{ij}^{\text{ling}} + \beta_{\text{fin}} \cdot d_{ij}^{\text{gci\_fin}} + \beta_Y \cdot Y_i + \beta_Y \cdot Y_j + \beta_{YY} \cdot Y_i Y_j + \text{Year FE}\Big) \cdot \eta_{ij}
$$

In Python using `statsmodels`:

```python
formula = 'a_ij ~ d_ling + d_gci_financial_composite + Y_i*Y_j + C(year)'
ppml = smf.poisson(formula=formula, data=ppml_df).fit(
    method="newton", maxiter=300,
    cov_type="cluster",
    cov_kwds={"groups": ppml_df['pair']},   # cluster by bilateral pair
)
```

Standard errors are clustered at the bilateral pair level.

### 1.3 — Store the estimated coefficients

Record:

$$
\hat{\beta}_{\text{ling}}, \quad \hat{\beta}_{\text{fin}}
$$

**Expected signs (both must be negative to enter the wedge):**
- $\hat{\beta}_{\text{ling}} < 0$ — more linguistic distance → less investment
- $\hat{\beta}_{\text{fin}} < 0$ — more financial system distance → less investment

If either coefficient is non-negative, it is **excluded** from the wedge construction (a warning is printed). Only coefficients that pass the sign check are stored in `beta_dict` and used downstream.

---

## Step 2: Construct the Bilateral Wedge Matrices

The model separates barriers into two independent components:

- **Hard barriers** — financial system distance, reducible by capital market integration policy (lever: $\omega$)
- **Soft barriers** — linguistic distance, reducible by language/cultural integration (lever: $\gamma$)

### 2.1 — Compute off-diagonal wedges for all $i \neq j$

**Hard (financial) wedge:**

$$
\Delta_{ij}^{\text{hard}} = \exp\!\left(-\hat{\beta}_{\text{fin}} \cdot d_{ij}^{\text{gci\_fin}}\right)
$$

**Soft (linguistic) wedge:**

$$
\Delta_{ij}^{\text{soft}} = \exp\!\left(-\hat{\beta}_{\text{ling}} \cdot d_{ij}^{\text{ling}}\right)
$$

**Combined baseline wedge:**

$$
\Delta_{ij}^{\text{baseline}} = \Delta_{ij}^{\text{hard}} \cdot \Delta_{ij}^{\text{soft}}
$$

Since both $\hat{\beta}_{\text{fin}} < 0$ and $\hat{\beta}_{\text{ling}} < 0$, both factors are $\geq 1$: more distance = larger friction.

The diagonal is set to 1 at this stage and calibrated in Step 2.2.

### 2.2 — Calibrate the domestic wedge $\Delta_{ii}$

The domestic wedge is calibrated to match the observed home bias, using the portfolio-share equation:

$$
\pi_{ii} = \frac{R_i^{\eta} \cdot M_i \,/\, \Delta_{ii}}{\displaystyle \sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}}
$$

**Excluded countries:** Financial centres (IRL, LUX, NLD, CHE) and any country with missing or extreme home bias ($\pi_{ii} \leq 0$ or $\pi_{ii} \geq 0.9999$) are excluded; their $\Delta_{ii}$ is set to 1.

**Iterative calibration algorithm:**

> 1. Initialise: $\Delta_{ii}^{(0)} = 0.01$ for calibratable countries; $\Delta_{ii}^{(0)} = 1$ for excluded countries.
> 2. For each calibratable country $i$:
>    - Compute $w_{i\iota} = R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}$ for all $\iota$
>    - Compute $Z_i = \sum_\iota w_{i\iota}$
>    - Compute $\hat{\pi}_{ii} = w_{ii} / Z_i$
>    - Update: $\Delta_{ii}^{(\text{new})} = \Delta_{ii}^{(\text{old})} \cdot \hat{\pi}_{ii} / \pi_{ii}^{\text{data}}$
> 3. Repeat until $\max_i |\hat{\pi}_{ii} - \pi_{ii}^{\text{data}}| < 10^{-5}$ (up to 20,000 iterations).

After convergence, the full $n \times n$ matrix $\boldsymbol{\Delta}^{\text{baseline}}$ (with calibrated diagonal) is stored as `Delta_baseline`.

---

## Step 3: Choose Return Elasticity and Returns

### 3.1 — Set the return elasticity

$$
\eta = 1 \qquad \text{(baseline)}
$$

Robustness checks use $\eta \in \{0.5, 1.0, 2.0\}$.

### 3.2 — Compute expected returns

Use the marginal product of capital from PWT:

$$
R_j = \alpha_j \cdot \frac{y_j}{k_j}
$$

Normalise so that the cross-country average equals 1:

$$
R_j \leftarrow \frac{R_j}{\bar{R}} \qquad \text{where} \quad \bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j
$$

---

## Step 4: Compute Baseline Portfolio Shares

For every $(i,j)$ pair:

$$
\pi_{ij}^{\text{baseline}} = \frac{R_j^{\eta} \cdot M_j \,/\, \Delta_{ij}^{\text{baseline}}}{\displaystyle\sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}^{\text{baseline}}}
$$

**Validation:** Compare $\boldsymbol{\Pi}^{\text{baseline}}$ to $\boldsymbol{\Pi}^{\text{data}}$. Diagonal entries match by construction. Compute the correlation between $\pi_{ij}^{\text{baseline}}$ and $\pi_{ij}^{\text{data}}$ across all $i \neq j$ pairs — this measures model fit.

---

## Step 5: Define the CMU Shock

The CMU shock operates on hard and soft barriers **independently** via two separate parameters.

### 5.1 — Choose the integration parameters

| Parameter | Lever | Scenarios |
|---|---|---|
| $\omega$ | Financial (hard) integration | $\{0.00, 0.25, 0.50, 1.00\}$ |
| $\gamma$ | Language (soft) integration | $\{0.00, 0.25, 0.50, 1.00\}$ |

All $4 \times 4 = 16$ combinations of $(\omega, \gamma)$ are computed independently.

### 5.2 — Construct the counterfactual wedge matrix

For every pair $(i,j)$:

$$
\Delta_{ij,\text{hard}}^{\text{CMU}} = \begin{cases}
\left(\Delta_{ij}^{\text{hard}}\right)^{1-\omega} & \text{if } i \neq j \text{ and both } i,j \in \text{EU27} \\[4pt]
\Delta_{ij}^{\text{hard}} & \text{otherwise}
\end{cases}
$$

$$
\Delta_{ij,\text{soft}}^{\text{CMU}} = \begin{cases}
\left(\Delta_{ij}^{\text{soft}}\right)^{1-\gamma} & \text{if } i \neq j \text{ and both } i,j \in \text{EU27} \\[4pt]
\Delta_{ij}^{\text{soft}} & \text{otherwise}
\end{cases}
$$

$$
\Delta_{ij}^{\text{CMU}} = \Delta_{ij,\text{hard}}^{\text{CMU}} \cdot \Delta_{ij,\text{soft}}^{\text{CMU}}
$$

**Domestic wedges $\Delta_{ii}$ stay unchanged** — copied from `Delta_baseline`.

**What the parameters do:** Raising a wedge to the power $(1 - \phi)$ shrinks it toward 1 in log space. Setting $\omega = 1$ fully eliminates all EU financial system distance barriers; $\omega = 0$ leaves them unchanged. $\gamma$ does the same for linguistic barriers independently.

**Example** — Germany–Poland pair with $\Delta^{\text{hard}} = 3.0$, $\Delta^{\text{soft}} = 2.0$:

| $\omega$ | $\gamma$ | $\Delta^{\text{CMU}}$ | Interpretation |
|---|---|---|---|
| 0.00 | 0.00 | $3.0 \times 2.0 = 6.00$ | No change |
| 1.00 | 0.00 | $1.0 \times 2.0 = 2.00$ | Only financial barriers removed |
| 0.00 | 1.00 | $3.0 \times 1.0 = 3.00$ | Only linguistic barriers removed |
| 1.00 | 1.00 | $1.0 \times 1.0 = 1.00$ | Full integration |

---

## Step 6: Compute Counterfactual Portfolios

Repeat the same calculation as Step 4 using $\boldsymbol{\Delta}^{\text{CMU}}$:

$$
\pi_{ij}^{\text{CMU}} = \frac{R_j^{\eta} \cdot M_j \,/\, \Delta_{ij}^{\text{CMU}}}{\displaystyle\sum_{\iota=1}^{n} R_\iota^{\eta} \cdot M_\iota \,/\, \Delta_{i\iota}^{\text{CMU}}}
$$

**Home bias change:**

$$
\Delta\pi_{ii} = \pi_{ii}^{\text{CMU}} - \pi_{ii}^{\text{baseline}}
$$

This should be **negative** for EU countries (home bias falls as EU markets become more accessible).

---

## Step 7: Compute Capital Reallocation

$$
k_j^{\text{baseline}} = \sum_{i=1}^{n} \pi_{ij}^{\text{baseline}} \cdot s_i = \left(\boldsymbol{\Pi}^{\text{baseline}\top} \mathbf{s}\right)_j
$$

$$
k_j^{\text{CMU}} = \sum_{i=1}^{n} \pi_{ij}^{\text{CMU}} \cdot s_i = \left(\boldsymbol{\Pi}^{\text{CMU}\top} \mathbf{s}\right)_j
$$

Total capital is conserved: $\sum_j k_j^{\text{CMU}} = \sum_j k_j^{\text{baseline}} = \sum_i s_i$.

**Expected pattern:**
- Small/peripheral EU countries (e.g. Baltic states, Bulgaria, Croatia): $\Delta k_j / k_j > 0$
- Outside countries (USA, GBR): $\Delta k_j / k_j \leq 0$

---

## Step 7b: Foreign Capital Decomposition (New)

This step decomposes total capital received into domestic and foreign components, which feeds the endogenous TFP channel in Step 8.

### 7b.1 — Compute foreign capital

For each country $j$, foreign equity capital is total received capital minus the part contributed by domestic investors:

$$
k_j^{\text{foreign}} = k_j^{\text{total}} - \pi_{jj} \cdot s_j
$$

Floor at zero (financial centres can have near-zero domestic holdings):

$$
k_j^{\text{foreign}} \leftarrow \max\!\left(k_j^{\text{foreign}},\; 0\right)
$$

### 7b.2 — Compute foreign-capital intensity

$$
f_j = \frac{k_j^{\text{foreign}}}{k_j^{\text{PWT}}}
$$

where $k_j^{\text{PWT}}$ is the physical capital stock from PWT. This is the correct denominator because TFP spillovers operate on the physical production process, not just the equity portfolio.

Compute under both scenarios:
- **Baseline:** $f_j^{\text{baseline}} = k_j^{\text{foreign, baseline}} / k_j^{\text{PWT}}$
- **CMU:** $f_j^{\text{CMU}}(\omega,\gamma) = k_j^{\text{foreign, CMU}} / k_j^{\text{PWT}}$

---

## Step 8: Output with Endogenous TFP

### 8.1 — Endogenous TFP

TFP is no longer fixed. Foreign equity inflows transmit knowledge and technology:

$$
A_j(\theta) = \bar{A}_j \cdot \left(1 + \theta \cdot f_j\right)
$$

where $\bar{A}_j$ is the baseline TFP backed out from PWT data. When $\theta = 0$, this reduces to fixed TFP exactly.

**Scenarios:** $\theta \in \{0.00, 0.05, 0.10\}$.

### 8.2 — Production function

$$
y_j = A_j(\theta) \cdot \left(k_j\right)^{\alpha_j} \cdot L_j^{1-\alpha_j}
$$

Under each $(\theta, \omega, \gamma)$ triplet, compute:
- **Baseline:** $A_j^{\text{baseline}} = \bar{A}_j (1 + \theta f_j^{\text{baseline}})$, then $y_j^{\text{baseline}}$
- **CMU:** $A_j^{\text{CMU}} = \bar{A}_j (1 + \theta f_j^{\text{CMU}})$, then $y_j^{\text{CMU}}$

Total number of output scenarios: $3 \times 4 \times 4 = 48$.

### 8.3 — Decompose the output change

The percentage change in output is approximated as:

$$
\frac{\Delta y_j}{y_j} \approx \underbrace{\alpha_j \cdot \frac{\Delta k_j}{k_j}}_{\text{capital deepening}} + \underbrace{\frac{\Delta A_j}{A_j}}_{\text{TFP spillover}}
$$

where:
- **Capital effect:** $\alpha_j \cdot (k_j^{\text{CMU}} - k_j^{\text{baseline}}) / k_j^{\text{baseline}}$
- **TFP effect:** $(A_j^{\text{CMU}} - A_j^{\text{baseline}}) / A_j^{\text{baseline}}$
- **Total effect (exact):** $(y_j^{\text{CMU}} - y_j^{\text{baseline}}) / y_j^{\text{baseline}}$

### 8.4 — Aggregate EU GDP effect

$$
\frac{\Delta Y^{\text{EU}}}{Y^{\text{EU}}} = \frac{\displaystyle\sum_{j \in \text{EU}} y_j^{\text{CMU}} - \sum_{j \in \text{EU}} y_j^{\text{baseline}}}{\displaystyle\sum_{j \in \text{EU}} y_j^{\text{baseline}}}
$$

### 8.5 — MPK convergence

$$
\sigma_{\text{MPK}}^{\text{baseline}} = \text{Std}\!\left(\alpha_j \cdot \frac{y_j^{\text{baseline}}}{k_j^{\text{baseline}}}\right)_{j \in \text{EU}}
$$

$$
\sigma_{\text{MPK}}^{\text{CMU}} = \text{Std}\!\left(\alpha_j \cdot \frac{y_j^{\text{CMU}}}{k_j^{\text{CMU}}}\right)_{j \in \text{EU}}
$$

$$
\text{Reduction} = \frac{\sigma_{\text{MPK}}^{\text{baseline}} - \sigma_{\text{MPK}}^{\text{CMU}}}{\sigma_{\text{MPK}}^{\text{baseline}}} \times 100\%
$$

---

## Economic Sense Checks

After running Step 8, five automatic checks are performed:

| Check | What is verified |
|---|---|
| 1 | $\text{Corr}(\Delta k_j^{\text{foreign}}, \Delta A_j) > 0$ across EU27 — countries that gain more foreign capital should have larger TFP gains |
| 2 | $\theta = 0$: $\max_j |\Delta A_j / A_j| < 10^{-10}$ — no TFP effect when spillover is off |
| 3 | TFP contribution $<$ capital deepening contribution (EU27 avg) — capital channel dominates |
| 4 | Non-EU countries lose foreign capital and TFP under $(\omega=1, \gamma=1)$ |
| 5 | EU GDP gain is monotone increasing in $\theta$ at $(\omega=1, \gamma=0)$ |

---

## Step 9: Robustness Checks

The function `run_model_variant_endo` re-runs the full model (Steps 2.2 through 8) for any given $(\eta, \theta)$ pair. The robustness grid is:

| Dimension | Values |
|---|---|
| $\eta$ (return elasticity) | 0.5, **1.0 (baseline)**, 2.0 |
| $\theta$ (TFP elasticity) | 0.00, 0.05, 0.10 |
| $\omega$ (financial integration) | 0.00, 0.25, 0.50, 1.00 |
| $\gamma$ (linguistic integration) | 0.00, 0.25, 0.50, 1.00 |

Total: $3 \times 3 \times 4 \times 4 = 144$ robustness scenarios.

For each variant, $\Delta_{ii}$ is **re-calibrated** to the specified $\eta$ before computing portfolios and output. The robustness table reports $\Delta Y^{\text{EU}} / Y^{\text{EU}}$ (%) and the average EU home-bias change for each combination.

---

## Flow Diagram: How the Steps Connect

```
STEP 0: Data assembly
  │
  ├── Bilateral holdings matrix A (IMF PIP) → a_ij, π_ij^data, s_i
  ├── GCI Financial Composite distance → d_gci_fin_ij
  ├── Linguistic distance → d_ling_ij
  ├── Country-level data (PWT) → y_j, k_j^PWT, α_j, L_j, Ā_j
  └── Market cap → M_j
  │
  ▼
STEP 1: Gravity regression (PPML)
  │
  │   Spec: a_ij ~ d_ling + d_gci_financial_composite + Y_i*Y_j + C(year)
  │   Output: β_ling, β_fin  (friction coefficients, both negative)
  │
  ▼
STEP 2: Build bilateral wedge matrices
  │
  │   Output: Δ_hard (n×n), Δ_soft (n×n), Δ_baseline = Δ_hard × Δ_soft
  │   Step 2.2: Calibrate Δ_ii to match observed home bias
  │   (IRL, LUX, NLD, CHE excluded from calibration)
  │
  ▼
STEP 3: Set returns R_j and elasticity η
  │
  │   R_j = α_j × y_j / k_j, normalised; η = 1.0 (baseline)
  │
  ▼
STEP 4: Compute baseline portfolios
  │
  │   Output: Π_baseline (n×n)
  │   Check: off-diagonal correlation with Π_data
  │
  ▼
STEPS 5–6: CMU shock → counterfactual portfolios
  │
  │   ω ∈ {0,0.25,0.5,1}, γ ∈ {0,0.25,0.5,1}  → 16 (ω,γ) scenarios
  │   Δ_CMU = Δ_hard^(1-ω) × Δ_soft^(1-γ)  [EU off-diagonal pairs only]
  │   Output: Π_CMU for each (ω, γ)
  │
  ▼
STEP 7: Capital reallocation
  │
  │   k_baseline = Π_baseline^T × s
  │   k_CMU(ω,γ) = Π_CMU^T × s
  │   Check: capital conservation
  │
  ▼
STEP 7b: Foreign capital decomposition  ← NEW
  │
  │   k_foreign_j = k_j - π_jj × s_j
  │   f_j = k_foreign_j / k_j^PWT
  │   Computed under baseline and all 16 (ω, γ) scenarios
  │
  ▼
STEP 8: Output with endogenous TFP  ← MODIFIED
  │
  │   A_j(θ) = Ā_j × (1 + θ × f_j)
  │   y_j = A_j(θ) × k_j^α_j × L_j^(1-α_j)
  │   θ ∈ {0.00, 0.05, 0.10}  →  48 total (θ, ω, γ) scenarios
  │   Decompose: total = capital deepening + TFP spillover
  │
  ▼
STEP 9: Robustness
  │
  │   Vary η ∈ {0.5, 1.0, 2.0} × θ ∈ {0,0.05,0.10} × all (ω, γ)
  │   Re-calibrate Δ_ii for each η
  │   144 robustness scenarios total
  │
  ▼
STEP 10: Export to Excel (16 sheets)
```

---

## What To Report

### Tables

1. **Gravity results** (Step 1): Coefficient table with $\hat{\beta}_{\text{ling}}$ and $\hat{\beta}_{\text{fin}}$, standard errors, p-values. Specification: `d_ling + d_gci_financial_composite + Y_i*Y_j + C(year)`, PPML, clustered by pair.

2. **Baseline home bias** (Step 4): Country | $\pi_{ii}^{\text{data}}$ | $\pi_{ii}^{\text{baseline}}$ | off-diagonal correlation.

3. **CMU portfolio effects** (Step 6): For each EU country: home share under baseline and CMU, change in home bias, across selected $(\omega, \gamma)$ pairs.

4. **Table 2a — EU GDP gain, $\gamma=0$ (financial only), $\theta=0.10$:**

   | $\omega$ | Avg $\Delta\pi_{\text{EU}}$ | $\Delta Y^{\text{EU}}/Y^{\text{EU}}$ (%) | Capital contribution (%) | TFP contribution (%) | $\sigma_{\text{MPK}}$ reduction (%) |
   |---|---|---|---|---|---|
   | 0.00 | … | … | … | … | … |
   | 0.25 | … | … | … | … | … |
   | 0.50 | … | … | … | … | … |
   | 1.00 | … | … | … | … | … |

5. **Table 2b — EU GDP gain, $\gamma=\omega$ (financial + linguistic), $\theta=0.10$**: same structure as 2a.

6. **Table 3 — Country-level results** ($\omega=1.0$, $\gamma=0.0$, $\theta=0.10$, EU27): $\Delta k_i/k_i$ (%), $\Delta A_i/A_i$ (%), $\Delta y_i/y_i$ (%), capital share and TFP share of total output change.

7. **Table 4 — Top 5 EU capital gainers** ($\omega=1.0$, $\gamma=1.0$, $\theta=0.10$).

8. **Table 5 — Off-diagonal portfolio correlation** (data vs. model).

9. **Table 6 — MPK dispersion** ($\theta=0.10$): $\sigma_{\text{MPK}}^{\text{baseline}}$, $\sigma_{\text{MPK}}^{\text{CMU}}$, reduction (%) for all $(\omega, \gamma)$.

10. **Robustness table** ($\eta \times \theta \times \omega \times \gamma$): $\Delta Y^{\text{EU}}/Y^{\text{EU}}$ (%) and avg $\Delta$HomeBias for all 144 scenarios.

### Figures

1. **Map of home bias change**: Colour EU countries by $\Delta\pi_{ii}$ (how much home bias falls).
2. **Map of GDP gains**: Colour EU countries by $\Delta y_j / y_j$.
3. **Scatter: baseline MPK vs. capital gain**: High-MPK countries gain the most capital (convergence).
4. **Bar chart: EU GDP gain vs. $\omega$** for $\gamma=0$ and $\gamma=\omega$, by $\theta$.
5. **Heatmap: $\Delta\pi_{ij}$** for a selected $(\omega, \gamma)$ — which bilateral flows increase most.
6. **Decomposition bars**: Capital vs. TFP contribution to $\Delta y_j/y_j$ by country, for $\theta=0.10$.

### Excel Export (`v4-simulation.xlsx`)

The script exports 16 sheets:

| Sheet | Contents |
|---|---|
| Config | All model parameters |
| Gravity_Coefficients | PPML coefficients, std errors, CI |
| Macro_Variables | R, M, k_PWT, Ā, L, α, home bias, f_baseline per country |
| Wedge_Hard | $\boldsymbol{\Delta}^{\text{hard}}$ matrix |
| Wedge_Soft | $\boldsymbol{\Delta}^{\text{soft}}$ matrix |
| Wedge_Baseline | $\boldsymbol{\Delta}^{\text{baseline}}$ matrix (combined + calibrated diagonal) |
| Portfolio_Data | Observed shares $\boldsymbol{\Pi}^{\text{data}}$ |
| Portfolio_Baseline | Model-implied shares $\boldsymbol{\Pi}^{\text{baseline}}$ |
| Capital_Reallocation | $k_j$ baseline and CMU for all 16 $(\omega,\gamma)$ scenarios |
| Foreign_Capital | $k_j^{\text{foreign}}$ and $f_j$ for all 16 scenarios |
| EU_Summary | EU aggregate stats for all 48 $(\theta,\omega,\gamma)$ scenarios |
| Output_Country | Country-level output, TFP, MPK for all 48 scenarios |
| Country_Detail_Main | Country table for main scenario ($\theta=0.10$, $\omega=1.0$, $\gamma=0.0$) |
| MPK_Dispersion | MPK vectors for all scenarios |
| Robustness | Full robustness grid (144 rows) |
| Portfolio_CMU_Snapshots | $\boldsymbol{\Pi}^{\text{CMU}}$ for 4 corner scenarios |
