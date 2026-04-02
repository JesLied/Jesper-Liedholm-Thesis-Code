# EU Capital Markets Union — Structural Model

## 1. Production

Each country $j$ produces output with a Cobb-Douglas technology:

$$Y_j = A_{0,j} \cdot K_j^{\alpha_j} \cdot L_{\text{eff},j}^{1-\alpha_j}$$

where $L_{\text{eff},j} = \text{hc}_j \cdot L_j$ is human-capital-adjusted labour, $\alpha_j = 1 - \text{labsh}_j$ is the capital income share, and $A_{0,j}$ is structural TFP back-calculated from PWT:

$$A_{0,j} = \frac{Y_j}{K_j^{\alpha_j} \cdot L_{\text{eff},j}^{1-\alpha_j}}$$

The marginal product of capital is:

$$\text{MPK}_j = \alpha_j \cdot A_{0,j} \cdot K_j^{\alpha_j - 1} \cdot L_{\text{eff},j}^{1-\alpha_j} = \alpha_j \frac{Y_j}{K_j}$$

---

## 2. Information Frictions

The bilateral information level $\Omega_{ij}$ governs how well investors in $i$ understand returns in $j$. It is estimated via a gravity equation on CPIS bilateral investment data:

$$\ln \Omega_{ij} = \underbrace{\beta_1 \cdot \text{Legal}_{ij} + \beta_2 \cdot \ln \text{Dist}_{ij} + \beta_3 \cdot \text{Lang}_{ij} + \beta_4 \cdot \ln \text{CPIS}_{ij,t-1}}_{\text{OLS on CPIS panel, 2000–2015}}$$

Perceived return volatility for investor $i$ looking at country $j$ is:

$$\tilde{\sigma}_{ij} = \sigma_j + \frac{\phi_{ij}}{1 + \ln(1 + \Omega_{ij})}$$

where $\sigma_j$ is the fundamental return volatility (from MSCI data) and $\phi_{ij}$ is the bilateral ambiguity premium. As $\Omega_{ij} \to \infty$, frictions vanish and $\tilde{\sigma}_{ij} \to \sigma_j$.

---

## 3. CMU Shock

CMU is modelled as a reduction in legal distance for all intra-EU pairs, parameterised by $\Delta \in [0,1]$:

$$\phi_{ij}^{\text{CMU}} = \phi_0 \cdot (1 - \Delta) \quad \forall\ (i,j) \in \text{EU} \times \text{EU},\ i \neq j$$

$\Delta = 0$ is the baseline. $\Delta = 1$ eliminates all information frictions between EU pairs. The model is solved at three values: $\Delta \in \{0.2,\, 0.5,\, 0.8\}$.

The mechanism is:

$$\Delta \uparrow \;\Rightarrow\; \phi_{ij} \downarrow \;\Rightarrow\; \tilde{\sigma}_{ij} \downarrow \;\Rightarrow\; w_{ij}^* \uparrow \;\Rightarrow\; K_j^* \uparrow \;\Rightarrow\; \text{MPK}_j \downarrow$$

Capital flows toward previously underinvested (high-MPK) countries until returns equalise.

---

## 4. Portfolio Allocation

Investors in country $i$ allocate wealth using mean-variance optimisation with bilateral perceived variances. The optimal weight on country $j$ is:

$$w_{ij}^* = \frac{\max\!\left(\dfrac{r_{\text{impl},j}}{\gamma \cdot \tilde{\sigma}_{ij}^2},\ 0\right)}{\displaystyle\sum_k \max\!\left(\dfrac{r_{\text{impl},k}}{\gamma \cdot \tilde{\sigma}_{ik}^2},\ 0\right)}$$

where $\gamma$ is the risk aversion coefficient and $r_{\text{impl},j} = \text{MPK}_j - \delta_j$ is the net expected return.

To match the observed sluggishness of capital reallocation, weights are updated with inertia:

$$w_{ij}(t) = \rho \cdot w_{ij}(t-1) + (1-\rho) \cdot w_{ij}^*(t)$$

**Home bias** arises endogenously: domestic assets have $\phi_{ii} = 0$, so $\tilde{\sigma}_{ii} = \sigma_i$, while foreign assets have $\tilde{\sigma}_{ij} > \sigma_i$. No explicit preference for home assets is needed.

---

## 5. Equilibrium

Market clearing requires that capital supplied to each country equals total investment directed there:

$$V_j = \sum_i w_{ij} \cdot W_i$$

Firms invest until the marginal product equals the required return. For each country $j$, equilibrium capital $K_j^*$ solves:

$$\text{MPK}_j(K_j^*) = r_{\text{impl},j} + \delta_j$$

Given the strict monotonicity of MPK in $K$ under Cobb-Douglas, this has a unique solution found via Brent's method. The inner loop iterates between equations (4) and (5) until:

$$\sum_j \left(\text{MPK}_j(K_j^*) - r_{\text{impl},j} - \delta_j\right)^2 < 10^{-6}$$

---

## 6. Steady-State Comparison

The CMU effect on country $j$ is the difference between the two steady states:

$$\Delta K_j = \bar{K}_j^*(\Delta) - \bar{K}_j^*(0)$$

**Result 1 — Capital allocation efficiency.** CMU reduces the cross-country variance of MPK:

$$\sigma^2\!\left(\text{MPK}(\Delta)\right) < \sigma^2\!\left(\text{MPK}(0)\right)$$

In the efficient benchmark (no frictions), MPK is equalised across all countries. The output gain from moving toward that benchmark is computed by reallocating total EU capital $\bar{K} = \sum_j K_j^*$ to equalise MPK, holding $\bar{K}$ fixed.

**Result 2 — Home bias reduction.** CMU reduces the domestic portfolio share:

$$\bar{w}_{jj}^*(\Delta) < \bar{w}_{jj}^*(0) \quad \forall\ j \in \text{EU}$$

The reduction is larger for countries where $\phi_{ij}$ was previously highest — typically peripheral member states with weaker legal integration.

---

## 7. Calibrated Parameters

| Parameter | Symbol | Source |
|-----------|--------|--------|
| Gravity elasticities | $\beta_1, \ldots, \beta_4$ | OLS on CPIS panel |
| Ambiguity premium | $\phi_0$ | SMM — targets home bias |
| Portfolio inertia | $\rho$ | SMM — targets CPIS autocorrelation |
| Risk aversion | $\gamma$ | SMM — typical range 2–5 |
| TFP shock scale | $\sigma_A$ | std$(\Delta A_j)$ from PWT |
| CMU intensity | $\Delta$ | Scenario: $\{0.2, 0.5, 0.8\}$ |
