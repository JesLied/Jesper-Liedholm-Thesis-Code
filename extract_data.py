#!/usr/bin/env python3
"""Extract input data for CMU model from Final-V4.ipynb processed data."""

import sys
import os
import pandas as pd
import numpy as np
import pycountry
from pathlib import Path

# Ensure we're in project root
os.chdir(Path(__file__).parent)

# Add notebook directory to path
sys.path.insert(0, 'Data/Notebooks')

# Import util functions from notebook
from util import load_and_merge_data

print("Loading and merging data from Final-V4.ipynb sources...")
df_full = load_and_merge_data()

print(f"Loaded data shape: {df_full.shape}")

# Extract for latest year
latest_year = df_full['year'].max()
print(f"Latest year in data: {latest_year}")

# ===== COUNTRIES =====
df_countries = df_full[df_full['year'] == latest_year][
    ['iso3_i', 'Y_i', 'k_i', 'L_i', 'alpha_i', 'A_i', 'M_i', 'euro_i', 'R_i']
].drop_duplicates()

# Add country names
try:
    country_names = {c.alpha_3: c.name for c in pycountry.countries}
    df_countries['country_name'] = df_countries['iso3_i'].map(country_names)
except:
    df_countries['country_name'] = df_countries['iso3_i']

# Mark EU members
eu_members = {
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 
    'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 
    'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
}
df_countries['is_eu'] = df_countries['iso3_i'].isin(eu_members).astype(int)

df_countries = df_countries[
    ['iso3_i', 'country_name', 'is_eu', 'euro_i', 'Y_i', 'k_i', 'L_i', 'alpha_i', 'A_i', 'M_i', 'R_i']
]

df_countries.to_csv('utils/v4/countries.csv', index=False)
print(f"✓ Saved countries.csv ({len(df_countries)} rows)")

# ===== BILATERAL HOLDINGS =====
df_holdings = df_full[df_full['year'] == latest_year][
    ['iso3_j', 'iso3_i', 'a_ij']
].drop_duplicates()

df_holdings = df_holdings.dropna(subset=['a_ij'])
df_holdings.to_csv('utils/v4/bilateral_holdings.csv', index=False)
print(f"✓ Saved bilateral_holdings.csv ({len(df_holdings)} rows)")

# ===== DISTANCES =====
df_distances = df_full.groupby(['iso3_i', 'iso3_j']).agg({
    'd_geo': 'first',
    'd_cul': 'first',
    'd_ling': 'first'
}).reset_index()

df_distances = df_distances.dropna(subset=['d_geo'])
df_distances.to_csv('utils/v4/distances.csv', index=False)
print(f"✓ Saved distances.csv ({len(df_distances)} rows)")

print("\nExtraction complete. Files are ready for CMU model.")
