# Simulating the Capital Markets Union: A Model Manual

## What This Document Is

This is a step-by-step manual for building a simulation model that answers: **if the EU Capital Markets Union removes soft barriers to cross-border investment, how do investor portfolios change, and what happens to capital allocation and productivity across EU countries?**

The model is a simplified, static version of Pellegrino, Spolaore & Wacziarg (2025, *QJE*). It is designed to be feasible for a master thesis while remaining structurally grounded in frontier research.

---

## Model Overview

The model has three blocks that run sequentially:

1. **Estimation Block** — Estimate a gravity equation on bilateral portfolio data to measure how much soft barriers (cultural distance, language, geography) distort investment within the EU.
2. **Portfolio Block** — Use the estimated barrier elasticities to compute counterfactual portfolio shares under a "CMU scenario" where intra-EU soft barriers are reduced.
3. **Production Block** — Feed the new portfolio shares into a capital market clearing condition to get the new distribution of capital across countries, then compute the change in output.

The chain of logic is:

$$
\text{CMU reduces } \Delta_{ij} \;\longrightarrow\; \text{Portfolios shift (less home bias)} \;\longrightarrow\; \text{Capital reallocates} \;\longrightarrow\; \text{Output changes}
$$

---

## Block 1: Estimation — The Gravity Equation

### 1.1 Setup

We have $n$ countries. Index the country where capital is invested (the *destination*) by $i$, and the country whose investors hold the capital (the *origin*) by $j$. We restrict the sample to EU-27 member states, plus a small number of non-EU countries (e.g. the US, UK, Switzerland, Norway) as "outside options" so that investors have the choice of investing outside the EU. This gives roughly $n \approx 31$ countries.

### 1.2 Data Required

| Variable | Source | Notes |
|---|---|---|
| Bilateral portfolio equity holdings $a_{ij}$ | IMF Coordinated Portfolio Investment Survey (CPIS), available at data.imf.org | Use the most recent year available. Equity + investment fund shares. Download the full bilateral matrix. |
| Geographic distance $d^{\text{geo}}_{ij}$ | geopoliticaldistance.org (Pellegrino, Spolaore & Wacziarg dataset) | Great-circle distance between capital cities, in km. |
| Cultural distance $d^{\text{cul}}_{ij}$ | geopoliticaldistance.org | Based on World Values Survey responses. Euclidean distance in values/beliefs. |
| Linguistic distance $d^{\text{ling}}_{ij}$ | geopoliticaldistance.org | Based on language family trees. Ranges from 0 (same language) to 1 (unrelated). |
| Common currency dummy $\text{Euro}_{ij}$ | Construct manually | $= 1$ if both $i$ and $j$ are in the Eurozone. |
| GDP of destination $Y_i$ | Penn World Table (PWT) or World Bank WDI | Controls for market size. |
| GDP of origin $Y_j$ | Same | Controls for investor wealth. |
| Stock market capitalisation $M_i$ | World Bank Financial Structure Database or World Federation of Exchanges | Proxy for investable market size in destination. |

### 1.3 The Gravity Regression

Following Pellegrino et al. (2025) and Portes & Rey (2005), estimate:

$$
\ln a_{ij} = \alpha_i + \alpha_j + \beta_1 \ln d^{\text{geo}}_{ij} + \beta_2 \, d^{\text{cul}}_{ij} + \beta_3 \, d^{\text{ling}}_{ij} + \beta_4 \, \text{Euro}_{ij} + \varepsilon_{ij}
$$

where:

- $a_{ij}$ = equity holdings of country $j$'s investors in country $i$'s assets (in USD millions)
- $\alpha_i$ = destination fixed effect (absorbs everything about destination $i$: market size, returns, risk, institutions)
- $\alpha_j$ = origin fixed effect (absorbs everything about origin $j$: total savings, risk aversion, home market size)
- $d^{\text{geo}}_{ij}$ = geographic distance (in logs because the relationship is log-linear)
- $d^{\text{cul}}_{ij}$ = cultural distance (in levels; it is already a continuous index)
- $d^{\text{ling}}_{ij}$ = linguistic distance (in levels)
- $\text{Euro}_{ij}$ = shared currency dummy

**Why fixed effects matter:** The origin and destination fixed effects absorb all country-level variation. What remains is *bilateral* variation — how much more or less does country $j$ invest in country $i$ compared to a frictionless benchmark, given the distances between them. This is the identification strategy: the $\beta$ coefficients measure the soft barrier elasticities.

**Estimation notes:**

- Drop observations where $i = j$ (domestic holdings). The model will handle home bias through the wedges.
- Drop zero or missing bilateral holdings. Alternatively, use a Poisson Pseudo Maximum Likelihood (PPML) estimator to handle zeros: $a_{ij} = \exp(\alpha_i + \alpha_j + \beta_1 \ln d^{\text{geo}}_{ij} + \ldots) \cdot \eta_{ij}$. PPML is preferable and standard in the gravity literature (Santos Silva & Tenreyro, 2006).
- Cluster standard errors by country-pair.

### 1.4 What You Get

The estimated coefficients $\hat{\beta}_1, \hat{\beta}_2, \hat{\beta}_3$ tell you: a one-unit increase in cultural distance reduces bilateral equity holdings by approximately $\hat{\beta}_2$ percent (in the log-linear spec) or by a factor of $\exp(\hat{\beta}_2)$ (in levels). These are the **soft barrier elasticities** — the key inputs for the simulation.

Pellegrino et al. find that geographic and cultural distances have large, statistically significant negative effects on bilateral investment, even after controlling for fixed effects. You should expect similar results within the EU sample, though likely with smaller magnitudes (EU countries are more similar than the global sample).

---

## Block 2: The Portfolio Model

### 2.1 The Asset Demand System

This is the core of the model. Following Pellegrino et al. (2025, eq. 2.14), the share of country $j$'s total portfolio invested in country $i$ is:

$$
\pi_{ij} = \frac{\displaystyle R_i^{\,\eta} \;\cdot\; M_i \;\big/\; \Delta_{ij}}{\displaystyle \sum_{\iota=1}^{n} R_\iota^{\,\eta} \;\cdot\; M_\iota \;\big/\; \Delta_{\iota j}}
$$

where:

- $\pi_{ij}$ = portfolio weight that investor country $j$ places on destination country $i$
- $R_i$ = risk-adjusted expected return on capital in country $i$
- $\eta$ = return elasticity (how sensitive portfolios are to return differentials)
- $M_i$ = investable market size of country $i$ (stock market capitalisation)
- $\Delta_{ij}$ = **portfolio wedge** — the bilateral friction between $i$ and $j$

### 2.2 What Is the Wedge?

The wedge $\Delta_{ij}$ captures everything that makes it harder for $j$'s investors to invest in $i$ relative to a frictionless world. In a frictionless world, $\Delta_{ij} = 1$ for all pairs, and everyone holds the market-cap-weighted world portfolio (the CAPM prediction).

The wedge has a structural interpretation coming from the gravity regression. Define it as:

$$
\ln \Delta_{ij} = -\hat{\beta}_1 \ln d^{\text{geo}}_{ij} - \hat{\beta}_2 \, d^{\text{cul}}_{ij} - \hat{\beta}_3 \, d^{\text{ling}}_{ij} - \hat{\beta}_4 \, \text{Euro}_{ij} + \mathbb{1}[i = j] \cdot \gamma
$$

where:

- The $\hat{\beta}$ coefficients come from your gravity regression (Block 1).
- The home bias term $\gamma$ captures the additional preference for domestic assets beyond what distances explain. You calibrate $\gamma$ so that the model reproduces observed home bias levels. (In practice, compute $\gamma$ as a residual: set it so that $\pi_{jj}^{\text{model}} = \pi_{jj}^{\text{data}}$ for each country $j$.)

**Important sign convention:** A higher $\Delta_{ij}$ means a *larger* barrier (more friction). Since the $\hat{\beta}$ coefficients are negative in the gravity regression (more distance = less investment), the minus signs in front of them ensure that $\Delta_{ij}$ increases with distance.

For the domestic portfolio share ($i = j$), set $d^{\text{geo}}_{jj} = 0$, $d^{\text{cul}}_{jj} = 0$, $d^{\text{ling}}_{jj} = 0$, so the only term is $\gamma$, which captures pure home bias.

### 2.3 Calibrating the Returns $R_i$ and Elasticity $\eta$

**Returns.** For a static, partial-equilibrium exercise, you can take returns as given. Use the inverse of the price-to-earnings ratio of each country's stock market index as a proxy for expected equity returns, or use the marginal product of capital from the Penn World Table (capital income share $\times$ GDP / capital stock). The precise measure matters less than consistency.

**Return elasticity $\eta$.** Pellegrino et al. estimate $\eta$ structurally. For your thesis, a simpler approach: set $\eta = 1$ as a baseline (log-linear demand) and check robustness at $\eta = 0.5$ and $\eta = 2$. With $\eta = 1$, the model says investors respond proportionally to return differentials. With $\eta = 0$, investors ignore returns entirely and portfolios are driven only by market size and frictions.

### 2.4 Computing Baseline Portfolio Shares

With all inputs in hand, compute the baseline portfolio matrix:

$$
\pi_{ij}^{\text{baseline}} = \frac{\displaystyle R_i^{\,\eta} \cdot M_i \,/\, \Delta_{ij}^{\text{baseline}}}{\displaystyle \sum_{\iota=1}^{n} R_\iota^{\,\eta} \cdot M_\iota \,/\, \Delta_{\iota j}^{\text{baseline}}}
$$

**Validation check:** Compare $\pi_{ij}^{\text{baseline}}$ to observed portfolio shares from the CPIS data. The model should reproduce the broad patterns: high domestic shares, declining foreign shares with distance, higher shares for Eurozone neighbours. If it doesn't match well, adjust $\gamma$ or check data inputs.

### 2.5 Defining the CMU Counterfactual

This is the key step. The CMU scenario is a *reduction in intra-EU soft barriers*. Define the counterfactual wedge as:

$$
\ln \Delta_{ij}^{\text{CMU}} = \begin{cases} (1 - \phi) \cdot \ln \Delta_{ij}^{\text{baseline}} & \text{if both } i \text{ and } j \text{ are EU members and } i \neq j \\ \ln \Delta_{ij}^{\text{baseline}} & \text{otherwise} \end{cases}
$$

where $\phi \in [0, 1]$ is the **CMU integration parameter**:

- $\phi = 0$: no change (status quo)
- $\phi = 0.25$: CMU removes 25% of intra-EU soft barriers
- $\phi = 0.50$: CMU removes 50% of intra-EU soft barriers
- $\phi = 1.0$: CMU removes all intra-EU soft barriers (full integration upper bound)

**Domestic wedges do not change.** The home bias term $\gamma$ for $i = j$ stays the same — the CMU makes foreign EU investment easier, but it doesn't change how people feel about their own market.

**Non-EU pairs do not change.** Barriers between EU countries and the US, UK, etc. remain the same.

You run the simulation for several values of $\phi$ to show a range of outcomes. This is more informative than picking a single number.

### 2.6 Computing Counterfactual Portfolios

Recompute portfolio shares with the new wedges:

$$
\pi_{ij}^{\text{CMU}} = \frac{\displaystyle R_i^{\,\eta} \cdot M_i \,/\, \Delta_{ij}^{\text{CMU}}}{\displaystyle \sum_{\iota=1}^{n} R_\iota^{\,\eta} \cdot M_\iota \,/\, \Delta_{\iota j}^{\text{CMU}}}
$$

The change in portfolio allocation for each origin $j$ is:

$$
\Delta \pi_{ij} = \pi_{ij}^{\text{CMU}} - \pi_{ij}^{\text{baseline}}
$$

**What to expect:**

- $\Delta \pi_{jj} < 0$ for EU countries: domestic share falls (home bias declines).
- $\Delta \pi_{ij} > 0$ for $i \neq j$, both in EU: cross-border EU holdings increase.
- $\Delta \pi_{ij} \approx 0$ for non-EU destinations: unchanged (or slightly negative as investors reallocate toward EU).

**This is the main result for the "portfolio" part of the thesis.** You can report the change in home bias for each EU country, the increase in cross-border EU holdings, and which countries gain the most foreign investment.

---

## Block 3: Capital Allocation and Productivity

### 3.1 From Portfolios to Capital Stocks

Each country's capital stock equals the sum of investment it receives from all origins:

$$
k_i = \sum_{j=1}^{n} \pi_{ij} \cdot s_j
$$

where $s_j$ is the total savings (portfolio wealth) of country $j$. Take $s_j$ from the data (gross national savings from the World Bank, or total portfolio equity assets from CPIS).

Compute capital stocks under both scenarios:

$$
k_i^{\text{baseline}} = \sum_{j=1}^{n} \pi_{ij}^{\text{baseline}} \cdot s_j \qquad \text{and} \qquad k_i^{\text{CMU}} = \sum_{j=1}^{n} \pi_{ij}^{\text{CMU}} \cdot s_j
$$

The **change in capital** received by country $i$ is:

$$
\Delta k_i = k_i^{\text{CMU}} - k_i^{\text{baseline}}
$$

Countries that were previously underweighted (small, peripheral, culturally distant) should see $\Delta k_i > 0$. Countries that were overweighted in home-biased portfolios may see $\Delta k_i < 0$ (capital outflows toward other EU countries).

### 3.2 Production Function

Each country produces output with a Cobb-Douglas technology:

$$
y_i = A_i \cdot k_i^{\alpha_i} \cdot L_i^{1 - \alpha_i}
$$

where:

- $A_i$ = total factor productivity (TFP) of country $i$
- $k_i$ = capital stock
- $L_i$ = labour force (fixed; does not move)
- $\alpha_i$ = capital income share (from PWT: variable `labsh` gives the labour share; $\alpha_i = 1 - \text{labsh}_i$)

Take $A_i$ and $L_i$ as given from the data. They do not change in the counterfactual — only $k_i$ changes.

### 3.3 Computing the Output Effect

The percentage change in output for country $i$ is approximately:

$$
\frac{\Delta y_i}{y_i} \approx \alpha_i \cdot \frac{\Delta k_i}{k_i}
$$

This follows from log-differentiating the production function, holding $A_i$ and $L_i$ fixed. Since $\alpha_i \approx 0.3$–$0.4$ for most EU countries, a 10% increase in capital translates to a 3–4% increase in output.

The aggregate EU output effect is:

$$
\frac{\Delta Y^{\text{EU}}}{Y^{\text{EU}}} = \frac{\sum_{i \in \text{EU}} \Delta y_i}{\sum_{i \in \text{EU}} y_i^{\text{baseline}}}
$$

### 3.4 The Marginal Product of Capital and Convergence

The marginal product of capital (MPK) in country $i$ is:

$$
\text{MPK}_i = \alpha_i \cdot \frac{y_i}{k_i} = \alpha_i \cdot A_i \cdot \left(\frac{L_i}{k_i}\right)^{1 - \alpha_i}
$$

Under the baseline, countries with less capital (peripheral EU members) have higher MPK. Under the CMU scenario, capital flows *toward* high-MPK countries, which pushes their MPK *down* toward the EU average. This is the **convergence** result: the CMU reduces the dispersion of returns across EU countries.

Report:

$$
\text{Dispersion}_{\text{baseline}} = \text{Std.Dev}\left(\text{MPK}_i^{\text{baseline}}\right) \qquad \text{vs.} \qquad \text{Dispersion}_{\text{CMU}} = \text{Std.Dev}\left(\text{MPK}_i^{\text{CMU}}\right)
$$

A decline in MPK dispersion means capital is more efficiently allocated.

### 3.5 Connecting to Firms

You do not need a full firm-level model. The argument is:

1. When $k_i$ increases for a capital-scarce country, the cost of capital falls (MPK falls, which proxies the equilibrium required return).
2. Firms in that country face cheaper financing, which allows them to invest more and expand.
3. Bau & Matray (2023) found that in India, when foreign capital was liberalised, revenues of capital-constrained firms rose by 23% and their capital stock by 53%. You cite this as a benchmark for the magnitude.
4. Gossé & Jehle (2024) show that the diversification gains within the EU are concentrated in Central and Eastern European countries — exactly the countries where you'd expect the CMU to send the most additional capital.

This gives you the narrative: **CMU reduces soft barriers → portfolios diversify → capital flows to underserved EU economies → firms there get cheaper funding → productivity converges across Europe.**

---

## Summary of Model Parameters

| Parameter | Description | How to Obtain |
|---|---|---|
| $\hat{\beta}_1, \hat{\beta}_2, \hat{\beta}_3, \hat{\beta}_4$ | Gravity elasticities for geographic, cultural, linguistic distance, and common currency | Estimated in Block 1 |
| $\gamma_j$ | Home bias residual for each country $j$ | Calibrated to match observed domestic portfolio share |
| $\eta$ | Return elasticity of portfolio demand | Baseline $\eta = 1$; robustness at 0.5 and 2 |
| $\phi$ | CMU integration parameter | Scenarios: 0%, 25%, 50%, 100% barrier removal |
| $R_i$ | Risk-adjusted return in country $i$ | Inverse P/E ratio or MPK from PWT |
| $M_i$ | Market capitalisation of country $i$ | World Federation of Exchanges or World Bank |
| $s_j$ | Total portfolio wealth of country $j$ | CPIS total equity assets or gross national savings |
| $A_i, L_i, \alpha_i$ | TFP, labour, capital share for country $i$ | Penn World Table 10.01 |

---

## What to Report in the Thesis

### Portfolio Results (Block 2)

- Table: Home bias by country, baseline vs. CMU scenarios ($\phi = 0.25, 0.50, 1.0$).
- Chart: Change in domestic portfolio share for each EU country.
- Chart: Heatmap of $\Delta \pi_{ij}$ showing which bilateral flows increase most.
- Highlight: Which countries see the largest increase in foreign capital inflows.

### Capital and Output Results (Block 3)

- Table: Percentage change in capital stock $\Delta k_i / k_i$ for each EU country, by scenario.
- Table: Percentage change in GDP $\Delta y_i / y_i$ for each EU country, by scenario.
- Chart: Scatter plot of baseline MPK vs. change in capital — should show that high-MPK countries gain more capital.
- Aggregate number: Total EU GDP gain from the CMU.
- Chart: Reduction in MPK dispersion across EU countries — the convergence result.

### Robustness

- Vary $\eta$ (return elasticity): does the result depend on how return-sensitive investors are?
- Vary $\phi$ (integration depth): show the results as a function of how much the CMU actually achieves.
- Use PPML instead of OLS for the gravity estimation and check if the $\hat{\beta}$ coefficients change.
- Exclude financial centre countries (Luxembourg, Ireland, Netherlands) following Beck et al. (2024), who show these distort EU integration statistics.

---

## Key References Supporting Each Block

| Block | Paper | What It Provides |
|---|---|---|
| 1 (Gravity) | Pellegrino, Spolaore & Wacziarg (2025) | The structural gravity model for investment; distance data |
| 1 (Gravity) | Portes & Rey (2005) | Earlier gravity evidence; information friction interpretation |
| 2 (Portfolios) | Van Nieuwerburgh & Veldkamp (2009) | Theoretical microfoundation: why soft barriers persist (endogenous information) |
| 2 (Portfolios) | Coeurdacier & Rey (2013) | Survey of home bias explanations; motivates the wedge structure |
| 2 (Portfolios) | Beck et al. (2024) | Shows EU equity home bias is higher than it appears; validates the exercise |
| 3 (Production) | Bau & Matray (2023) | Causal evidence that removing barriers increases firm productivity |
| 3 (Production) | Gossé & Jehle (2024) | Diversification gains within the EU, concentrated in CEECs |
| Context | Darvas & Schoenmaker (2017) | Institutional investors reduce home bias; EU membership helps |
| Context | Lewis (1999) | Historical baseline and welfare costs of home bias |

---

## Limitations to Acknowledge

1. **Partial equilibrium.** Returns $R_i$ are held fixed. In reality, capital inflows would bid up asset prices and push returns down. This means the model overstates capital reallocation for large shocks (high $\phi$). Acknowledging this is fine — you're computing an upper bound.
2. **Static.** No dynamics, no transition path. You compare two steady states. The actual adjustment would take years.
3. **No general equilibrium feedback.** Wages, savings rates, and TFP do not respond to capital flows. A full GE model (like Pellegrino et al.) would capture these, but is beyond master thesis scope.
4. **CPIS data limitations.** Bilateral holdings may be distorted by financial centres (Luxembourg, Ireland). Beck et al. (2024) document this. You can address it by running a robustness check excluding these countries, or by using their corrected data if available.
5. **CMU is not a single shock.** The real CMU involves many heterogeneous reforms (harmonised disclosure, consolidated tape, cross-border fund passporting, insolvency reform). Modelling it as a uniform reduction $\phi$ in all soft barriers is a simplification. You can discuss which specific reforms map to which type of distance reduction.
