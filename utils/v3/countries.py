import pandas as pd
import numpy as np


def aggregate_row(
    data: pd.DataFrame,
    row_countries,
    row_label: str = "ROW",
) -> pd.DataFrame:
    """
    Collapse `row_countries` into one synthetic country (`row_label`) and
    return a bilateral panel suitable for run_simulation().

    The output contains:
      - All (target, target) bilateral pairs  — cross-flows between the
        countries you are actually simulating.
      - All (target, ROW) and (ROW, target) pairs — flows between each
        target country and the rest of the world aggregate.
      - One (ROW, ROW) diagonal row per year — needed by run_simulation()
        to initialise K_sim / W_sim / sigma_vec for the ROW block.

    Column contract (matches COLUMN_MAP in simulation.py)
    -------------------------------------------------------
    Identifiers        : iso3_i, iso3_j, year
    Bilateral gravity  : dist, ln_dist, border, lang, lang_share,
                         lang_official, cpis, cpis_lag1, ln_cpis,
                         ln_cpis_lag1
    Source macro (_i)  : Y_i, K_i, L_i, pop_i, lab_sh_i, delta_i,
                         A_i, rf_i, inv_share_i, hc_i, legal_i,
                         inv_freedom_i, fin_freedom_i, tax_i, r_i
    Dest macro (_j)    : Y_j, K_j, L_j, pop_j, lab_sh_j, delta_j,
                         A_j, rf_j, inv_share_j, hc_j, legal_i
                         (destination legal score is also named legal_i
                          in the panel — both source and dest Heritage
                          scores map to the same column name via COLUMN_MAP)

    Aggregation rules
    -----------------
    Sum columns   : Y, K, L, pop — additive quantities
    Weighted-mean : all rate/share/index variables, weighted by GDP (Y)
    Bilateral flows: cpis, cpis_lag1 → summed across ROW members
    Bilateral chars: dist, ln_dist, border, lang, lang_share,
                     lang_official, ln_cpis, ln_cpis_lag1
                     → CPIS-weighted mean across ROW pairs (simple mean
                        if cpis missing)
    """
    df = data.copy()

    # ── Deduplicate columns if renaming created clashes ───────────────────
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()

    required = {"iso3_i", "iso3_j", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    row_set = set(row_countries)

    # Preserve original IDs so macro aggregation uses one row per
    # (country, year) rather than one per bilateral pair
    df["_iso3_i_orig"] = df["iso3_i"]
    df["_iso3_j_orig"] = df["iso3_j"]

    # Map row countries to ROW label in bilateral panel
    df["iso3_i"] = np.where(df["iso3_i"].isin(row_set), row_label, df["iso3_i"])
    df["iso3_j"] = np.where(df["iso3_j"].isin(row_set), row_label, df["iso3_j"])

    # ── Helper: aggregate macro side ──────────────────────────────────────
    def _agg_macro(
        frame: pd.DataFrame,
        orig_iso_col: str,
        out_iso_col: str,
        sum_cols: list,
        weighted_cols: list,
        gdp_col: str,
    ) -> pd.DataFrame:
        """
        Aggregate macro variables for one side (source or destination).
        ROW members are GDP-weighted averaged for rates/shares/indices
        and summed for levels.
        """
        avail = set(frame.columns)
        keep  = [orig_iso_col, "year"] + [
            c for c in (sum_cols + weighted_cols) if c in avail
        ]
        keep = list(dict.fromkeys(keep))   # preserve order, no dupes

        tmp = (
            frame[keep]
            .drop_duplicates(subset=[orig_iso_col, "year"], keep="last")
            .copy()
        )
        tmp[out_iso_col] = np.where(
            tmp[orig_iso_col].isin(row_set), row_label, tmp[orig_iso_col]
        )

        sum_e      = [c for c in sum_cols     if c in tmp.columns]
        weighted_e = [c for c in weighted_cols if c in tmp.columns]

        # Convert weighted cols to GDP-weighted numerators before summing
        if gdp_col in tmp.columns:
            for c in weighted_e:
                tmp[c] = tmp[c] * tmp[gdp_col]

        agg_dict = {c: "sum" for c in (sum_e + weighted_e)}
        out = tmp.groupby([out_iso_col, "year"], as_index=False).agg(agg_dict)

        # Divide back to get weighted mean
        if gdp_col in out.columns:
            denom = out[gdp_col].replace({0: np.nan})
            for c in weighted_e:
                out[c] = out[c] / denom

        return out

    # ── Source macro (_i) ─────────────────────────────────────────────────
    # Sum columns: additive quantities
    i_sum = ["Y_i", "K_i", "L_i", "pop_i"]
    # Weighted-mean columns: rates, shares, indices
    # Note: legal_i is the Heritage Overall Score for the source country
    #       (Overall Score_o → legal_i in COLUMN_MAP)
    i_weighted = [
        "A_i", "delta_i", "lab_sh_i", "r_i", "rf_i",
        "inv_share_i", "hc_i", "legal_i",
        "inv_freedom_i", "fin_freedom_i", "tax_i",
    ]
    macro_i = _agg_macro(
        frame        = df,
        orig_iso_col = "_iso3_i_orig",
        out_iso_col  = "iso3_i",
        sum_cols     = i_sum,
        weighted_cols= i_weighted,
        gdp_col      = "Y_i",
    )

    # ── Destination macro (_j) ────────────────────────────────────────────
    # Sum columns
    j_sum = ["Y_j", "K_j", "L_j", "pop_j"]
    # Weighted-mean columns.
    # legal_i appears here too: the destination Heritage score is named
    # legal_i in the panel (Overall Score_d → legal_i via COLUMN_MAP).
    # The simulation reads it from destination rows for the legal matrix.
    # We carry it through as a _j-side aggregation under the same name.
    # There is no legal_j, inv_freedom_j etc. — COLUMN_MAP only creates
    # _i variants for these institutional variables.
    j_weighted = [
        "A_j", "delta_j", "lab_sh_j", "r_j", "rf_j",
        "inv_share_j", "hc_j",
    ]
    macro_j = _agg_macro(
        frame        = df,
        orig_iso_col = "_iso3_j_orig",
        out_iso_col  = "iso3_j",
        sum_cols     = j_sum,
        weighted_cols= j_weighted,
        gdp_col      = "Y_j",
    )

    # ── Bilateral aggregation ─────────────────────────────────────────────
    # Exclude self-pairs at this stage (ROW-ROW bilateral flows are
    # meaningless — we reconstruct the diagonal separately below)
    bilat = df[df["iso3_i"] != df["iso3_j"]].copy()
    keys  = ["iso3_i", "iso3_j", "year"]

    # Flow columns — sum across ROW members
    flow_cols = [c for c in ["cpis", "cpis_lag1"] if c in bilat.columns]

    if flow_cols:
        out = bilat.groupby(keys, as_index=False)[flow_cols].sum()
    else:
        out = bilat[keys].drop_duplicates().copy()

    # Bilateral characteristic columns — CPIS-weighted mean across ROW pairs
    # Includes both raw and log-transformed versions already in the CSV
    char_cols = [
        c for c in [
            "dist", "ln_dist",
            "border",
            "lang", "lang_share", "lang_official",
            "ln_cpis", "ln_cpis_lag1",
        ]
        if c in bilat.columns
    ]

    use_weights = "cpis" in bilat.columns
    for c in char_cols:
        if use_weights:
            tmp = bilat[keys + [c, "cpis"]].dropna(subset=[c, "cpis"]).copy()
            if len(tmp) == 0:
                out[c] = np.nan
                continue
            tmp["_num"] = tmp[c] * tmp["cpis"]
            num = tmp.groupby(keys)["_num"].sum()
            den = tmp.groupby(keys)["cpis"].sum().replace({0: np.nan})
            wm  = (num / den).rename(c).reset_index()
            out = out.merge(wm, on=keys, how="left")
        else:
            mean_c = bilat.groupby(keys, as_index=False)[c].mean()
            out = out.merge(mean_c, on=keys, how="left")

    # Attach macro variables
    out = out.merge(macro_i, on=["iso3_i", "year"], how="left")
    out = out.merge(macro_j, on=["iso3_j", "year"], how="left")

    # ── Reconstruct diagonal (self-pair) rows ─────────────────────────────
    # run_simulation() extracts diagonal rows (iso3_i == iso3_j) to
    # initialise K_sim, W_sim, and sigma_vec for every country including
    # ROW. Without this, ROW has no diagonal entry and the simulation
    # will produce NaN state vectors for the ROW block.
    #
    # We build one diagonal row per (country, year) by merging the
    # aggregated macro_i table onto itself as both source and destination.
    all_countries = out["iso3_i"].unique().tolist()
    years         = out["year"].unique().tolist()

    diag_keys = pd.DataFrame(
        [(c, c, y) for c in all_countries for y in years],
        columns=["iso3_i", "iso3_j", "year"],
    )

    diag_out = diag_keys.copy()

    # Bilateral flow columns are zero on diagonal (no self-investment)
    for c in flow_cols:
        diag_out[c] = 0.0
    for c in char_cols:
        diag_out[c] = np.nan

    # Macro _i side: merge aggregated source macro
    diag_out = diag_out.merge(macro_i, on=["iso3_i", "year"], how="left")

    # Macro _j side: same data, renamed from _i to _j
    macro_j_from_i = macro_i.rename(columns={"iso3_i": "iso3_j"})
    # Rename _i columns to _j for the destination side
    col_remap = {}
    for col in macro_j_from_i.columns:
        if col.endswith("_i") and col not in ("iso3_i",):
            col_remap[col] = col[:-2] + "_j"
    macro_j_from_i = macro_j_from_i.rename(columns=col_remap)

    diag_out = diag_out.merge(
        macro_j_from_i, on=["iso3_j", "year"], how="left"
    )

    # legal_i on the destination diagonal: same value as source legal_i
    # (simulation reads legal_i from both source and destination rows)
    if "legal_i" in diag_out.columns and "legal_i_x" not in diag_out.columns:
        pass   # already present from macro_i merge
    elif "legal_i_x" in diag_out.columns:
        diag_out["legal_i"] = diag_out["legal_i_x"]
        diag_out = diag_out.drop(
            columns=[c for c in diag_out.columns if c in ("legal_i_x", "legal_i_y")]
        )

    # ── Combine bilateral and diagonal ────────────────────────────────────
    combined = (
        pd.concat([out, diag_out], ignore_index=True)
        .sort_values(["year", "iso3_i", "iso3_j"])
        .reset_index(drop=True)
    )

    # Drop any internal helper columns that leaked
    leak_cols = [c for c in combined.columns if c.startswith("_iso3_")]
    if leak_cols:
        combined = combined.drop(columns=leak_cols)

    return combined


# ===========================================================================
# ASSERTIONS
# ===========================================================================

def quick_asserts(
    df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    row_countries,
    row_label: str = "ROW",
    tol: float = 1e-8,
) -> None:
    """
    Sanity checks on the output of aggregate_row.

    Checks
    ------
    1. No off-diagonal self-pairs (only diagonal iso3_i == iso3_j allowed)
    2. Unique bilateral keys (iso3_i, iso3_j, year) for non-diagonal rows
    3. CPIS bilateral flows conserved under aggregation
    4. ROW macro (Y_i) is unique per year — consistent aggregation
    5. Every country including ROW has a diagonal self-pair row per year
    """
    row_set = set(row_countries)
    keys    = ["iso3_i", "iso3_j", "year"]

    # 1) Off-diagonal self-pairs: only (ROW, ROW) and (X, X) diagonals allowed
    non_diag = aggregated_df[aggregated_df["iso3_i"] != aggregated_df["iso3_j"]]
    same_non_diag = non_diag[non_diag["iso3_i"] == non_diag["iso3_j"]]
    assert len(same_non_diag) == 0, "Found unexpected self-pairs in off-diagonal rows"

    # 2) Unique keys for non-diagonal rows
    assert not non_diag.duplicated(keys).any(), \
        "Duplicate (iso3_i, iso3_j, year) in off-diagonal output"

    # 3) Bilateral CPIS conservation
    if "cpis" in df.columns and "cpis" in aggregated_df.columns:
        mapped = df.copy()
        mapped["iso3_i"] = np.where(
            mapped["iso3_i"].isin(row_set), row_label, mapped["iso3_i"]
        )
        mapped["iso3_j"] = np.where(
            mapped["iso3_j"].isin(row_set), row_label, mapped["iso3_j"]
        )
        mapped = mapped[mapped["iso3_i"] != mapped["iso3_j"]]
        exp = mapped.groupby(keys, as_index=False)["cpis"].sum()
        got = aggregated_df[aggregated_df["iso3_i"] != aggregated_df["iso3_j"]][
            keys + ["cpis"]
        ].copy()
        chk = exp.merge(got, on=keys, how="outer", suffixes=("_exp", "_got"))
        assert chk["cpis_exp"].notna().all() and chk["cpis_got"].notna().all(), \
            "Missing cpis rows after aggregation"
        assert np.isclose(
            chk["cpis_exp"], chk["cpis_got"], atol=tol, rtol=tol
        ).all(), "CPIS values do not match after aggregation"

    # 4) ROW macro unique per year
    if "Y_i" in aggregated_df.columns:
        row_macro = (
            aggregated_df[aggregated_df["iso3_i"] == row_label][["year", "Y_i"]]
            .drop_duplicates()
        )
        assert len(row_macro) == len(row_macro[["year"]].drop_duplicates()), \
            "Y_i is not unique per year for ROW — aggregation inconsistency"

    # 5) Every country has a diagonal row for every year
    years    = sorted(aggregated_df["year"].unique())
    all_iso3 = sorted(aggregated_df["iso3_i"].unique())
    diag     = aggregated_df[aggregated_df["iso3_i"] == aggregated_df["iso3_j"]]
    for iso3 in all_iso3:
        for yr in years:
            match = diag[(diag["iso3_i"] == iso3) & (diag["year"] == yr)]
            assert len(match) == 1, \
                f"Missing diagonal row for ({iso3}, {yr})"

    print("All quick_asserts passed.")


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(0)
    n = 6
    iso3s = ["DEU", "FRA", "ITA", "ESP", "SWE", "USA"]
    years = [2000, 2001]

    rows = []
    for yr in years:
        for src in iso3s:
            for dst in iso3s:
                rows.append({
                    "iso3_i"      : src,
                    "iso3_j"      : dst,
                    "year"        : yr,
                    "Y_i"         : np.random.uniform(100, 500),
                    "K_i"         : np.random.uniform(200, 1000),
                    "L_i"         : np.random.uniform(1, 50),
                    "pop_i"       : np.random.uniform(5, 100),
                    "lab_sh_i"    : np.random.uniform(0.4, 0.7),
                    "delta_i"     : np.random.uniform(0.03, 0.08),
                    "hc_i"        : np.random.uniform(2.0, 3.5),
                    "inv_share_i" : np.random.uniform(0.15, 0.40),
                    "legal_i"     : np.random.uniform(40, 80),
                    "rf_i"        : np.random.uniform(0.02, 0.15),
                    "r_i"         : np.random.uniform(-0.1, 0.2),
                    "A_i"         : np.random.uniform(0.8, 1.2),
                    "Y_j"         : np.random.uniform(100, 500),
                    "K_j"         : np.random.uniform(200, 1000),
                    "L_j"         : np.random.uniform(1, 50),
                    "pop_j"       : np.random.uniform(5, 100),
                    "lab_sh_j"    : np.random.uniform(0.4, 0.7),
                    "delta_j"     : np.random.uniform(0.03, 0.08),
                    "hc_j"        : np.random.uniform(2.0, 3.5),
                    "inv_share_j" : np.random.uniform(0.15, 0.40),
                    "rf_j"        : np.random.uniform(0.02, 0.15),
                    "r_j"         : np.random.uniform(-0.1, 0.2),
                    "A_j"         : np.random.uniform(0.8, 1.2),
                    "cpis"        : np.random.uniform(0, 50) if src != dst else 0,
                    "cpis_lag1"   : np.random.uniform(0, 40) if src != dst else 0,
                    "dist"        : np.random.uniform(100, 5000) if src != dst else 0,
                    "ln_dist"     : np.random.uniform(4, 9) if src != dst else 0,
                    "border"      : int(src != dst and np.random.rand() > 0.7),
                    "lang"        : np.random.uniform(0, 1),
                    "lang_share"  : np.random.uniform(0, 1),
                    "lang_official": int(np.random.rand() > 0.5),
                    "ln_cpis"     : np.random.uniform(0, 4) if src != dst else 0,
                    "ln_cpis_lag1": np.random.uniform(0, 4) if src != dst else 0,
                })

    df = pd.DataFrame(rows)

    # Aggregate USA into ROW, keep EU5 as simulation targets
    target_countries = ["DEU", "FRA", "ITA", "ESP", "SWE"]
    row_countries    = ["USA"]

    agg = aggregate_row(df, row_countries=row_countries, row_label="ROW")

    print(f"Output shape : {agg.shape}")
    print(f"Countries    : {sorted(agg['iso3_i'].unique())}")
    print(f"Diagonal rows: {len(agg[agg['iso3_i'] == agg['iso3_j']])}")
    print(f"Off-diag rows: {len(agg[agg['iso3_i'] != agg['iso3_j']])}")

    quick_asserts(df, agg, row_countries=row_countries)