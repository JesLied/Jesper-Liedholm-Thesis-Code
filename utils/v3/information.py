"""Information friction functions — gravity-based Omega estimation."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ====================================
# Gravity Omega (vectorised)
# ====================================
def compute_omega() -> pd.Series:
    """Placeholder"""
    
    # give random number between 0 and 100
    return np.random.rand(len(df)) * 100


# ====================================
if __name__ == "__main__":
    # Make random dataframe for testing
    np.random.seed(0)
    test_df = pd.DataFrame({
        "iso3_i": np.random.choice(["USA", "CAN", "MEX"], size=100),
        "iso3_j": np.random.choice(["USA", "CAN", "MEX"], size=100),
        "year": np.random.choice(range(2000, 2020), size=100),
        "dist": np.random.rand(100) * 1000,  # Random distances
        "cpis_lag1": np.random.rand(100) * 10,  # Random CPIS lagged values
    })
    
    # Inject some random nans to see what happens
    for col in ["dist", "cpis_lag1"]:
        test_df.loc[test_df.sample(frac=0.1).index, col] = np.nan
    
    # Define coefficients for our test
    test_coef_map = {
        "dist": -0.1,
        "cpis_lag1": 0.5,
    }
    
    # Compute Omega
    test_df["omega"] = compute_omega(test_df, test_coef_map)
    print(test_df[["dist", "cpis_lag1", "omega"]].head())