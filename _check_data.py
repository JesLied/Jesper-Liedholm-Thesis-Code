import pandas as pd
import numpy as np

BASE = '/Users/jesper/Desktop/CBS/Thesis 1/Jesper-Liedholm-Thesis-Code'
df = pd.read_csv(f'{BASE}/Data/Clean/Final-v4.csv')

EU27 = ['AUT','BEL','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC',
        'HUN','IRL','ITA','LVA','LTU','LUX','MLT','NLD','POL','PRT','ROU','SVK',
        'SVN','ESP','SWE']
OUTSIDE = ['USA','GBR','CHE','NOR']
COUNTRIES = EU27 + OUTSIDE
print(f'n = {len(COUNTRIES)} countries')

yr = 2022
sub = df[(df['year']==yr) & (df['iso3_i'].isin(COUNTRIES)) & (df['iso3_j'].isin(COUNTRIES))]
missing_i = [c for c in COUNTRIES if c not in sub['iso3_i'].unique()]
missing_j = [c for c in COUNTRIES if c not in sub['iso3_j'].unique()]
print('missing as destination (i):', missing_i)
print('missing as origin (j):', missing_j)

diag = sub[sub['iso3_i'] == sub['iso3_j']]
print('\ndiagonal rows sample:')
print(diag[['iso3_i','iso3_j','d_geo','d_ling','d_cul']].head(5))

off_diag = sub[sub['iso3_i'] != sub['iso3_j']]
print(f'\noff-diag d_geo nulls: {off_diag["d_geo"].isna().sum()} of {len(off_diag)}')
print(f'off-diag d_ling nulls: {off_diag["d_ling"].isna().sum()} of {len(off_diag)}')
print(f'off-diag d_cul nulls: {off_diag["d_cul"].isna().sum()} of {len(off_diag)}')

sub2 = sub.drop_duplicates('iso3_i').copy()
sub2['MPK'] = sub2['alpha_i'] * sub2['Y_i'] / sub2['k_i']
print('\nMPK sample:')
print(sub2[['iso3_i','MPK','alpha_i','Y_i','k_i']].sort_values('MPK', ascending=False).head(10).to_string())

# Last non-null year per country pair for a_ij and k_i
print('\nLast non-null year for a_ij (per iso3_j):')
aij_last = df[df['iso3_j'].isin(COUNTRIES) & df['a_ij'].notna()].groupby('iso3_j')['year'].max()
print(aij_last.sort_values().to_string())

print('\nLast non-null year for k_i (per iso3_i):')
ki_last = df[df['iso3_i'].isin(COUNTRIES) & df['k_i'].notna()].groupby('iso3_i')['year'].max()
print(ki_last.sort_values().to_string())
