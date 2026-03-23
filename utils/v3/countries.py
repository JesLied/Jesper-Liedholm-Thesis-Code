import pandas as pd
import numpy as np

def aggregate_row(data: pd.DataFrame, row_countries, row_label: str = "ROW") -> pd.DataFrame:
    """
    Collapse `row_countries` into one synthetic country (`row_label`) and return
    bilateral panel with:
      - aggregated bilateral flows (i,j,year),
      - source-country macro vars (_i),
      - destination-country macro vars (_j),
    excluding self-pairs (iso3_i == iso3_j).
    """
    df = data.copy()

    # If rename map created duplicate column names, keep last occurrence.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()

    required = {"iso3_i", "iso3_j", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep original IDs for macro aggregation (avoid bilateral duplication bias)
    df["_iso3_i_orig"] = df["iso3_i"]
    df["_iso3_j_orig"] = df["iso3_j"]

    # Map selected countries to ROW in bilateral panel
    row_set = set(row_countries)
    df["iso3_i"] = np.where(df["iso3_i"].isin(row_set), row_label, df["iso3_i"])
    df["iso3_j"] = np.where(df["iso3_j"].isin(row_set), row_label, df["iso3_j"])

    def _agg_macro(
        frame: pd.DataFrame,
        orig_iso_col: str,
        out_iso_col: str,
        sum_cols: list[str],
        weighted_cols: list[str],
        gdp_col: str,
    ) -> pd.DataFrame:
        keep_cols = [orig_iso_col, "year"] + [c for c in (sum_cols + weighted_cols) if c in frame.columns]
        keep_cols = list(dict.fromkeys(keep_cols))
        tmp = frame[keep_cols].drop_duplicates(subset=[orig_iso_col, "year"], keep="last").copy()
        tmp[out_iso_col] = np.where(tmp[orig_iso_col].isin(row_set), row_label, tmp[orig_iso_col])

        sum_cols_e = [c for c in sum_cols if c in tmp.columns]
        weighted_cols_e = [c for c in weighted_cols if c in tmp.columns]

        if gdp_col in tmp.columns:
            for c in weighted_cols_e:
                tmp[c] = tmp[c] * tmp[gdp_col]

        agg_dict = {c: "sum" for c in (sum_cols_e + weighted_cols_e)}
        out = tmp.groupby([out_iso_col, "year"], as_index=False).agg(agg_dict)

        if gdp_col in out.columns:
            denom = out[gdp_col].replace({0: np.nan})
            for c in weighted_cols_e:
                out[c] = out[c] / denom

        return out

    # Source macro (_i)
    i_sum = ["Y_i", "K_i", "L_i", "pop_i"]
    i_weighted = ["A_i", "delta_i", "lab_sh_i", "r_i", "inv_freedom_i", "fin_freedom_i", "tax_i", "legal_i"]
    macro_i = _agg_macro(
        frame=df,
        orig_iso_col="_iso3_i_orig",
        out_iso_col="iso3_i",
        sum_cols=i_sum,
        weighted_cols=i_weighted,
        gdp_col="Y_i",
    )

    # Destination macro (_j)
    j_sum = ["Y_j", "K_j", "L_j", "pop_j"]
    j_weighted = ["A_j", "delta_j", "lab_sh_j", "r_j"]
    macro_j = _agg_macro(
        frame=df,
        orig_iso_col="_iso3_j_orig",
        out_iso_col="iso3_j",
        sum_cols=j_sum,
        weighted_cols=j_weighted,
        gdp_col="Y_j",
    )

    # Bilateral side: exclude self-pairs (including ROW-ROW)
    bilat = df[df["iso3_i"] != df["iso3_j"]].copy()
    keys = ["iso3_i", "iso3_j", "year"]

    flow_cols = [c for c in ["cpis", "cpis_lag1"] if c in bilat.columns]
    avg_cols = [c for c in ["dist", "border", "lang", "lang_share", "lang_official"] if c in bilat.columns]

    # Start with summed flow columns (or unique keys if none exist)
    if flow_cols:
        out = bilat.groupby(keys, as_index=False)[flow_cols].sum()
    else:
        out = bilat[keys].drop_duplicates().copy()

    # Weighted averages for bilateral characteristics (weights = cpis if present, else mean)
    use_weights = "cpis" in bilat.columns
    for c in avg_cols:
        if use_weights:
            tmp = bilat[keys + [c, "cpis"]].dropna(subset=[c, "cpis"]).copy()
            if len(tmp) == 0:
                out[c] = np.nan
            else:
                tmp["_num"] = tmp[c] * tmp["cpis"]
                num = tmp.groupby(keys)["_num"].sum()
                den = tmp.groupby(keys)["cpis"].sum().replace({0: np.nan})
                wm = (num / den).rename(c).reset_index()
                out = out.merge(wm, on=keys, how="left")
        else:
            mean_c = bilat.groupby(keys, as_index=False)[c].mean()
            out = out.merge(mean_c, on=keys, how="left")

    # Attach macro vars
    out = out.merge(macro_i, on=["iso3_i", "year"], how="left")
    out = out.merge(macro_j, on=["iso3_j", "year"], how="left")

    # Clean helper columns if leaked
    leak_cols = [c for c in out.columns if c.startswith("_iso3_")]
    if leak_cols:
        out = out.drop(columns=leak_cols)

    return out.sort_values(["year", "iso3_i", "iso3_j"]).reset_index(drop=True)


def quick_asserts(df, aggregated_df, row_countries, row_label="ROW", tol=1e-8):
    import numpy as np

    row_set = set(row_countries)
    keys = ["iso3_i", "iso3_j", "year"]

    # 1) No self-pairs + unique bilateral keys
    assert (aggregated_df["iso3_i"] != aggregated_df["iso3_j"]).all(), "Found self-pairs in output"
    assert not aggregated_df.duplicated(keys).any(), "Duplicate (iso3_i, iso3_j, year) in output"

    # 2) Bilateral flow conservation (cpis)
    if "cpis" in df.columns and "cpis" in aggregated_df.columns:
        mapped = df.copy()
        mapped["iso3_i"] = np.where(mapped["iso3_i"].isin(row_set), row_label, mapped["iso3_i"])
        mapped["iso3_j"] = np.where(mapped["iso3_j"].isin(row_set), row_label, mapped["iso3_j"])
        mapped = mapped[mapped["iso3_i"] != mapped["iso3_j"]]

        exp = mapped.groupby(keys, as_index=False)["cpis"].sum()
        got = aggregated_df[keys + ["cpis"]].copy()
        chk = exp.merge(got, on=keys, how="outer", suffixes=("_exp", "_got"))

        assert chk["cpis_exp"].notna().all() and chk["cpis_got"].notna().all(), "Missing cpis rows"
        assert np.isclose(chk["cpis_exp"], chk["cpis_got"], atol=tol, rtol=tol).all(), "cpis mismatch"

    # 3) ROW macro consistency: Y_i value is same across all (ROW, j, year) pairs
    if "Y_i" in aggregated_df.columns:
        row_macro = aggregated_df[aggregated_df["iso3_i"] == row_label][["year", "Y_i"]].drop_duplicates()
        assert len(row_macro) == len(row_macro[["year"]].drop_duplicates()), "Y_i varies within year for ROW"
    
    
    
# ====================================
if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        "iso3_i": ["USA", "CAN", "MEX", "FRA", "DEU"],
        "iso3_j": ["USA", "CAN", "MEX", "FRA", "DEU"],
        "year": [2000, 2000, 2000, 2000, 2000],
        "Y_i": [100, 80, 60, 120, 110],
        "Y_j": [100, 80, 60, 120, 110],
        "cpis": [10, 8, 6, 12, 11],
        "border": [1, 0, 1, 0, 0],
    })
    countries = ["USA", "CAN", "MEX"]
    aggregated_df = aggregate_row(df, countries)
    quick_asserts(df, aggregated_df, countries)
    print("Quick checks passed.")
    
    # # Assert sum of Y_i for ROW equals sum of Y_i for the row_countries
    # row_countries = ["USA", "CAN", "MEX"]
    # row_sum = df[df["iso3_i"].isin(row_countries)]["Y_i"].sum()
    # row_agg_value = aggregated_df[aggregated_df["iso3_i"] == "ROW"]["Y_i"].values[0]
    # assert np.isclose(row_sum, row_agg_value), f"Expected ROW Y_i to be {row_sum}, but got {row_agg_value}"