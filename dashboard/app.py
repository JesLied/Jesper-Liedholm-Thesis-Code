from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
EXCEL_PATH = ROOT_DIR / "v4-simulation.xlsx"
GEOJSON_PATH = ROOT_DIR / "Data" / "europe.geo.json"
LOGO_PATH = ROOT_DIR / "CBSlogo_extended_rgb_blue.png"
ZERO_ABS_TOL = 1e-6
ZERO_PCT_TOL = 5e-4

OUTPUT_COUNTRY_COLUMNS = [
    "theta",
    "omega",
    "gamma",
    "country",
    "in_EU27",
    "is_financial_ctr",
    "k_fin_baseline",
    "k_fin_cmu",
    "dk_fin",
    "dk_fin_pct",
    "y_baseline",
    "y_cmu",
    "dy",
    "dy_pct",
    "capital_effect_pct",
    "tfp_amplification_pct",
    "mpk_baseline",
    "mpk_cmu",
]

PORTFOLIO_FLOW_COLUMNS = [
    "scenario",
    "theta",
    "omega",
    "gamma",
    "iso3_i",
    "iso3_j",
    "k_flow",
]


st.set_page_config(
    page_title="CMU Simulation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .dashboard-logo {
        width: 20%;
        min-width: 180px;
        max-width: 360px;
        margin: 0 0 0.9rem 0;
    }

    .block-container {
        max-width: 170rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.03;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1.505rem;
        font-weight: 700;
        line-height: 1.1;
    }

    [data-testid="stMetricDelta"] svg {
        height: 1.15rem;
        width: 1.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Output_Country from v4-simulation.xlsx")
def load_output_country(path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name="Output_Country",
        usecols=OUTPUT_COUNTRY_COLUMNS,
        engine="openpyxl",
    )
    numeric_cols = [c for c in OUTPUT_COUNTRY_COLUMNS if c != "country"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(show_spinner="Loading Portfolio_Flows from v4-simulation.xlsx")
def load_portfolio_flows(path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name="Portfolio_Flows",
        usecols=PORTFOLIO_FLOW_COLUMNS,
        engine="openpyxl",
    )
    numeric_cols = ["theta", "omega", "gamma", "k_flow"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        geojson = json.load(fh)

    features = []
    for feature in geojson["features"]:
        props = feature.get("properties", {})
        iso3 = props.get("adm0_a3") or props.get("iso_a3_eh") or props.get("iso_a3")
        if not iso3 or iso3 == "-99":
            continue
        if props.get("sovereignt") == "Iceland":
            continue
        props["dashboard_iso3"] = iso3
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def scenario_slice(df: pd.DataFrame, omega: float, gamma: float, theta: float) -> pd.DataFrame:
    mask = (
        np.isclose(df["omega"], omega)
        & np.isclose(df["gamma"], gamma)
        & np.isclose(df["theta"], theta)
    )
    scenario = df.loc[mask].copy()
    scenario["region"] = np.where(scenario["in_EU27"].eq(1), "EU27", "Outside")
    zero_small_changes(scenario)
    return scenario.sort_values("country")


def compare_to_fixed_baseline(scenario: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = baseline[
        [
            "country",
            "y_baseline",
            "k_fin_baseline",
        ]
    ].rename(
        columns={
            "y_baseline": "fixed_y_baseline",
            "k_fin_baseline": "fixed_k_fin_baseline",
        }
    )
    compared = scenario.merge(baseline_cols, on="country", how="left")
    compared["dy"] = compared["y_cmu"] - compared["fixed_y_baseline"]
    compared["dy_pct"] = np.where(
        np.isclose(compared["fixed_y_baseline"], 0.0),
        np.nan,
        compared["dy"] / compared["fixed_y_baseline"] * 100,
    )
    compared["dk_fin"] = compared["k_fin_cmu"] - compared["fixed_k_fin_baseline"]
    compared["dk_fin_pct"] = np.where(
        np.isclose(compared["fixed_k_fin_baseline"], 0.0),
        np.nan,
        compared["dk_fin"] / compared["fixed_k_fin_baseline"] * 100,
    )
    compared["y_baseline"] = compared["fixed_y_baseline"]
    compared["k_fin_baseline"] = compared["fixed_k_fin_baseline"]
    zero_small_changes(compared)
    return compared.drop(columns=["fixed_y_baseline", "fixed_k_fin_baseline"])


def zero_small_changes(df: pd.DataFrame) -> None:
    for col in ["dy", "dk_fin", "return_change"]:
        if col in df:
            df.loc[df[col].abs() < ZERO_ABS_TOL, col] = 0.0
    for col in ["dy_pct", "dk_fin_pct", "return_change_pct"]:
        if col in df:
            df.loc[df[col].abs() < ZERO_PCT_TOL, col] = 0.0


def positive_count(values: pd.Series) -> int:
    return int((values > ZERO_PCT_TOL).sum())


def available_flow_scenarios(flows: pd.DataFrame) -> pd.DataFrame:
    scenarios = flows[["scenario", "omega", "gamma", "theta"]].drop_duplicates().copy()
    preferred_order = {
        "baseline": 0,
        "most_probable": 1,
        "mid": 2,
        "max": 3,
    }
    scenarios["sort_order"] = scenarios["scenario"].map(preferred_order).fillna(99)
    return scenarios.sort_values(["sort_order", "omega", "gamma", "theta"]).drop(columns="sort_order")


def calculate_capital_returns(
    output_country: pd.DataFrame,
    flows: pd.DataFrame,
    return_scenario: pd.Series,
) -> pd.DataFrame:
    baseline_country = scenario_slice(output_country, 0.0, 0.0, 0.0)
    scenario_country = scenario_slice(
        output_country,
        float(return_scenario["omega"]),
        float(return_scenario["gamma"]),
        float(return_scenario["theta"]),
    )

    baseline_flows = flows[flows["scenario"].eq("baseline")].copy()
    scenario_flows = flows[
        flows["scenario"].eq(return_scenario["scenario"])
        & np.isclose(flows["omega"], return_scenario["omega"])
        & np.isclose(flows["gamma"], return_scenario["gamma"])
        & np.isclose(flows["theta"], return_scenario["theta"])
    ].copy()

    baseline_mpk = baseline_country[["country", "mpk_baseline"]].rename(
        columns={"country": "iso3_j", "mpk_baseline": "mpk"}
    )
    scenario_mpk_col = "mpk_baseline" if return_scenario["scenario"] == "baseline" else "mpk_cmu"
    scenario_mpk = scenario_country[["country", scenario_mpk_col]].rename(
        columns={"country": "iso3_j", scenario_mpk_col: "mpk"}
    )

    baseline_returns = baseline_flows.merge(baseline_mpk, on="iso3_j", how="left")
    baseline_returns["return_on_capital_baseline"] = baseline_returns["k_flow"] * baseline_returns["mpk"]
    baseline_returns = (
        baseline_returns.groupby("iso3_i", as_index=False)["return_on_capital_baseline"].sum()
    )

    scenario_returns = scenario_flows.merge(scenario_mpk, on="iso3_j", how="left")
    scenario_returns["return_on_capital_cmu"] = scenario_returns["k_flow"] * scenario_returns["mpk"]
    scenario_returns = scenario_returns.groupby("iso3_i", as_index=False)["return_on_capital_cmu"].sum()

    returns = baseline_returns.merge(scenario_returns, on="iso3_i", how="outer").fillna(0)
    returns["return_change"] = returns["return_on_capital_cmu"] - returns["return_on_capital_baseline"]
    returns["return_change_pct"] = np.where(
        np.isclose(returns["return_on_capital_baseline"], 0.0),
        np.nan,
        returns["return_change"] / returns["return_on_capital_baseline"] * 100,
    )
    zero_small_changes(returns)

    return scenario_country.merge(returns, left_on="country", right_on="iso3_i", how="left")


def metric_delta(value: float, suffix: str = "%") -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < ZERO_PCT_TOL:
        value = 0.0
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}{suffix}"


def format_usd_mn(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < ZERO_ABS_TOL:
        value = 0.0
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.2f}tn"
    if abs_value >= 1_000:
        return f"{sign}${abs_value / 1_000:.2f}bn"
    return f"{sign}${abs_value:.1f}mn"


def format_table_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return ""
    return f"{value:,.{decimals}f}"


def render_logo(path: Path) -> None:
    if not path.exists():
        return

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<img class="dashboard-logo" src="data:image/png;base64,{encoded}" alt="CBS logo">',
        unsafe_allow_html=True,
    )


def symmetric_range(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if finite.empty:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 95))
    if not np.isfinite(limit) or limit == 0:
        limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    return -limit, limit


def make_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    value_col: str,
    title: str,
    colorbar_title: str,
    hover_template: str,
    custom_data: list[str],
):
    range_color = symmetric_range(df[value_col])

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="country",
        featureidkey="properties.dashboard_iso3",
        color=value_col,
        color_continuous_scale="PiYG",
        color_continuous_midpoint=0,
        range_color=range_color,
        hover_name="country",
        custom_data=custom_data,
    )

    fig.update_traces(
        marker_line_color="#CBD5E1",
        marker_line_width=0.45,
        hovertemplate=hover_template,
    )
    fig.update_geos(
        visible=False,
        showland=True,
        landcolor="#E5E7EB",
        showcountries=True,
        countrycolor="#CBD5E1",
        lonaxis_range=[-15, 40],
        lataxis_range=[35, 70],
        projection_type="mercator",
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 18}},
        height=672,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        coloraxis_colorbar={
            "title": colorbar_title,
            "thickness": 12,
            "len": 0.78,
            "x": 1.01,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def make_country_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        [
            "country",
            "region",
            "dy_pct",
            "dy",
            "dk_fin_pct",
            "dk_fin",
            "k_fin_baseline",
        ]
    ].copy()
    table = table.sort_values("dy_pct", ascending=False)
    table = table.rename(
        columns={
            "country": "Country",
            "region": "Region",
            "dy_pct": "GDP change (%)",
            "dy": "GDP change (USD mn)",
            "dk_fin_pct": "Net portfolio flow change (%)",
            "dk_fin": "Net portfolio flow change (USD mn)",
            "k_fin_baseline": "Baseline portfolio capital (USD mn)",
        }
    )

    zero_decimal_cols = [
        "GDP change (USD mn)",
        "Net portfolio flow change (USD mn)",
        "Baseline portfolio capital (USD mn)",
    ]
    three_decimal_cols = [
        "GDP change (%)",
        "Net portfolio flow change (%)",
    ]
    for col in zero_decimal_cols:
        table[col] = table[col].map(lambda value: format_table_number(value, 0))
    for col in three_decimal_cols:
        table[col] = table[col].map(lambda value: format_table_number(value, 3))

    return table


def main() -> None:
    if not EXCEL_PATH.exists():
        st.error(f"Missing workbook: {EXCEL_PATH}")
        st.stop()
    if not GEOJSON_PATH.exists():
        st.error(f"Missing GeoJSON: {GEOJSON_PATH}")
        st.stop()

    df = load_output_country(EXCEL_PATH)
    flows = load_portfolio_flows(EXCEL_PATH)
    geojson = load_geojson(GEOJSON_PATH)
    flow_scenarios = available_flow_scenarios(flows)
    scenario_labels = flow_scenarios["scenario"].tolist()
    scenario_lookup = flow_scenarios.set_index("scenario", drop=False)
    scenario_display = {
        row["scenario"]: (
            f"{row['scenario']} "
            f"(omega={row['omega']:.3f}, gamma={row['gamma']:.3f}, tfp={row['theta']:.3f})"
        )
        for _, row in flow_scenarios.iterrows()
    }
    default_scenario_idx = (
        scenario_labels.index("most_probable")
        if "most_probable" in scenario_labels
        else 0
    )

    with st.sidebar:
        st.title("CMU Parameters")

        selected_scenario = st.selectbox(
            "Scenario",
            options=scenario_labels,
            index=default_scenario_idx,
            format_func=lambda label: scenario_display[label],
        )
        return_scenario = scenario_lookup.loc[selected_scenario]
        omega = float(return_scenario["omega"])
        gamma = float(return_scenario["gamma"])
        theta = float(return_scenario["theta"])
        st.caption(f"Selected: omega={omega:.3f}, gamma={gamma:.3f}, tfp={theta:.3f}")

    fixed_baseline = scenario_slice(df, 0.0, 0.0, 0.0)
    selected = compare_to_fixed_baseline(scenario_slice(df, omega, gamma, theta), fixed_baseline)
    return_data = calculate_capital_returns(df, flows, return_scenario)
    eu = selected[selected["in_EU27"].eq(1)]

    eu_gdp_base = eu["y_baseline"].sum()
    eu_gdp_change = eu["dy"].sum()
    eu_gdp_pct = eu_gdp_change / eu_gdp_base * 100 if eu_gdp_base else np.nan
    eu_flow_base = eu["k_fin_baseline"].sum()
    eu_flow_change = eu["dk_fin"].sum()
    eu_flow_pct = eu_flow_change / eu_flow_base * 100 if eu_flow_base else np.nan
    aggregate_return_base = return_data["return_on_capital_baseline"].sum()
    aggregate_return_change = return_data["return_change"].sum()
    aggregate_return_pct = (
        aggregate_return_change / aggregate_return_base * 100
        if aggregate_return_base
        else np.nan
    )
    if np.isfinite(aggregate_return_pct) and abs(aggregate_return_pct) < ZERO_PCT_TOL:
        aggregate_return_pct = 0.0

    render_logo(LOGO_PATH)
    st.title("CMU Simulation Dashboard")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric(
            "EU GDP change",
            format_usd_mn(eu_gdp_change),
            metric_delta(eu_gdp_pct),
        )
    with metric_cols[1]:
        st.metric(
            "EU net portfolio flow",
            format_usd_mn(eu_flow_change),
            metric_delta(eu_flow_pct),
        )
    with metric_cols[2]:
        st.metric("Countries with GDP gain", f"{positive_count(selected['dy_pct'])}/{len(selected)}")
    with metric_cols[3]:
        st.metric("Countries with inflow gain", f"{positive_count(selected['dk_fin_pct'])}/{len(selected)}")
    with metric_cols[4]:
        st.metric(
            "Aggregate return / year",
            format_usd_mn(aggregate_return_change),
            metric_delta(aggregate_return_pct),
        )

    country_custom_data = [
        "country",
        "dy_pct",
        "dy",
        "dk_fin_pct",
        "dk_fin",
        "y_baseline",
        "k_fin_baseline",
        "region",
    ]
    gdp_hover = (
        "<b>%{customdata[0]}</b><br>"
        "Region: %{customdata[7]}<br>"
        "GDP change: %{customdata[1]:+.3f}%<br>"
        "GDP change: %{customdata[2]:+,.0f} USD mn<br>"
        "Baseline GDP: %{customdata[5]:,.0f} USD mn"
        "<extra></extra>"
    )
    flow_hover = (
        "<b>%{customdata[0]}</b><br>"
        "Region: %{customdata[7]}<br>"
        "Net portfolio flow: %{customdata[3]:+.3f}%<br>"
        "Net portfolio flow: %{customdata[4]:+,.0f} USD mn<br>"
        "Baseline portfolio capital: %{customdata[6]:,.0f} USD mn"
        "<extra></extra>"
    )
    return_custom_data = [
        "country",
        "region",
        "return_change_pct",
        "return_change",
        "return_on_capital_baseline",
        "return_on_capital_cmu",
    ]
    return_hover = (
        "<b>%{customdata[0]}</b><br>"
        "Region: %{customdata[1]}<br>"
        "Annual return change: %{customdata[2]:+.3f}%<br>"
        "Annual return change: %{customdata[3]:+,.0f} USD mn<br>"
        "Baseline annual return: %{customdata[4]:,.0f} USD mn<br>"
        "Scenario annual return: %{customdata[5]:,.0f} USD mn"
        "<extra></extra>"
    )

    left, middle, right = st.columns(3, gap="small")
    with left:
        st.plotly_chart(
            make_choropleth(
                selected,
                geojson,
                "dy_pct",
                "Change in GDP",
                "Delta GDP (%)",
                gdp_hover,
                country_custom_data,
            ),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    with middle:
        st.plotly_chart(
            make_choropleth(
                selected,
                geojson,
                "dk_fin_pct",
                "Change in Net Portfolio Flow",
                "Delta flow (%)",
                flow_hover,
                country_custom_data,
            ),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    with right:
        st.plotly_chart(
            make_choropleth(
                return_data,
                geojson,
                "return_change_pct",
                "Change in Annual Return on Capital",
                "Delta return (%)",
                return_hover,
                return_custom_data,
            ),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.dataframe(
        make_country_table(selected),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
