Good question. Let me be direct about what Pellegrino et al. do and don't cover, and then give you the honest assessment of what's feasible.

**Pellegrino et al. (2025) hold $A_i$ fixed.** Their model treats TFP as exogenous. They explicitly acknowledge this as a limitation. Their counterfactuals only capture the capital-deepening channel (more $k$ → more $y$), not the productivity channel (more foreign capital → knowledge spillovers → higher $A$). This means their 7% global GDP gain estimate is a *lower bound* — it misses the TFP effects entirely.

**What actually makes $A$ endogenous in the real world?** There are three credible channels, and they all point in the same direction — foreign capital inflows raise TFP:

**Channel 1: Technology spillovers from FDI.** When foreign firms invest in a country, they bring production techniques, management practices, and supply chain knowledge that spills over to domestic firms. Borensztein, De Gregorio & Lee (1998, *Journal of International Economics*) is the canonical reference — they find that FDI raises growth through TFP, but only when the host country has sufficient human capital to absorb the technology. Javorcik (2004, *American Economic Review*) finds spillovers are strongest through *vertical* linkages (foreign firms demand higher quality from local suppliers) rather than horizontal competition within the same industry.

**Channel 2: Competition and reallocation.** Foreign capital entry forces domestic firms to become more efficient or exit. This raises *aggregate* TFP even if no individual firm improves, because market share shifts from low-productivity to high-productivity firms. Bau & Matray (2023) — which you already have — captures exactly this: when India liberalised foreign capital, the Solow residual of treated industries rose 3–16%, driven by capital flowing to high-MRPK firms.

**Channel 3: Learning-by-investing (the Van Nieuwerburgh & Veldkamp angle).** When investors start allocating capital to a foreign market, they acquire information about it. This reduces the information asymmetry over time, which further reduces the effective barrier. This creates a positive feedback loop: initial investment → learning → lower barriers → more investment. This is the mirror image of the self-reinforcing home bias in Van Nieuwerburgh & Veldkamp (2009). The CMU could trigger this virtuous cycle.

**Now here's my honest assessment of whether you should model this.**

You shouldn't build a full endogenous TFP model. It would require specifying a knowledge production function, calibrating spillover elasticities, and solving a dynamic system — that's a PhD paper, not a master thesis extension. And the empirical estimates of FDI-to-TFP spillovers vary wildly across studies (some find zero, some find large effects), so any calibration you choose would be arbitrary and easily attacked by an examiner.

**What I'd recommend instead: a simple reduced-form TFP adjustment as a sensitivity analysis.**

The idea is: estimate from the literature how much a 1% increase in foreign capital raises TFP, then apply that elasticity to your $\Delta k$ results. Specifically:

$$
A_i^{\text{CMU}} = A_i^{\text{baseline}} \cdot \left(1 + \theta \cdot \frac{\Delta k_i^{\text{foreign}}}{k_i^{\text{PWT}}}\right)
$$

where $\Delta k_i^{\text{foreign}}$ is the change in *foreign* capital received by country $i$ (excluding domestic reinvestment), and $\theta$ is the TFP spillover elasticity. Borensztein et al. (1998) estimate that a 1 percentage point increase in the FDI-to-GDP ratio raises TFP growth by about 0.8%. Baltabaev (2014) finds a long-run elasticity of FDI stock to TFP of about 0.1 for a broad panel.

You run this as a separate scenario: "Model with endogenous TFP" vs. "Model with fixed TFP." The difference between them gives you the magnitude of the TFP channel. You present the fixed-TFP version as conservative and the endogenous version as an upper bound. This is transparent, easy to implement, and gives your thesis an extra dimension without overcomplicating the model.

The key references to cite for this extension are Borensztein, De Gregorio & Lee (1998) for the spillover elasticity, Javorcik (2004) for the micro evidence on the mechanism, and Bau & Matray (2023) for the most recent causal evidence. You can frame it as: "Pellegrino et al. (2025) acknowledge that fixed TFP is a limitation; I extend their framework with a reduced-form TFP channel calibrated to the empirical spillover literature."