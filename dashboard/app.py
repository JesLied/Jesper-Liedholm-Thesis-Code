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
        max-width: 130rem;
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


def nearest_value(values: np.ndarray, target: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(values[np.argmin(np.abs(values - target))])


def scenario_slice(df: pd.DataFrame, omega: float, gamma: float, theta: float) -> pd.DataFrame:
    mask = (
        np.isclose(df["omega"], omega)
        & np.isclose(df["gamma"], gamma)
        & np.isclose(df["theta"], theta)
    )
    scenario = df.loc[mask].copy()
    scenario["region"] = np.where(scenario["in_EU27"].eq(1), "EU27", "Outside")
    return scenario.sort_values("country")


def metric_delta(value: float, suffix: str = "%") -> str:
    if not np.isfinite(value):
        return ""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}{suffix}"


def format_usd_mn(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
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
        custom_data=[
            "country",
            "dy_pct",
            "dy",
            "dk_fin_pct",
            "dk_fin",
            "y_baseline",
            "k_fin_baseline",
            "region",
        ],
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
    geojson = load_geojson(GEOJSON_PATH)

    omegas = [float(v) for v in np.sort(df["omega"].dropna().unique())]
    gammas = [float(v) for v in np.sort(df["gamma"].dropna().unique())]
    thetas = [float(v) for v in np.sort(df["theta"].dropna().unique())]

    default_omega = nearest_value(omegas, 0.30)
    default_gamma = nearest_value(gammas, 0.10)
    default_theta = nearest_value(thetas, 0.03)

    with st.sidebar:
        st.title("CMU Parameters")

        omega = st.select_slider(
            "Omega (hard integration)",
            options=omegas,
            value=default_omega,
            format_func=lambda v: f"{v:.3f}",
        )

        gamma = st.select_slider(
            "Gamma (soft integration)",
            options=gammas,
            value=default_gamma,
            format_func=lambda v: f"{v:.3f}",
        )

        theta = st.select_slider(
            "TFP spillover",
            options=thetas,
            value=default_theta,
            format_func=lambda v: f"{v:.3f}",
        )

        st.caption(f"Scenario grid: omega={omega:.3f}, gamma={gamma:.3f}, tfp={theta:.3f}")

    selected = scenario_slice(df, omega, gamma, theta)
    eu = selected[selected["in_EU27"].eq(1)]

    eu_gdp_base = eu["y_baseline"].sum()
    eu_gdp_cmu = eu["y_cmu"].sum()
    eu_gdp_pct = (eu_gdp_cmu - eu_gdp_base) / eu_gdp_base * 100 if eu_gdp_base else np.nan
    eu_flow_base = eu["k_fin_baseline"].sum()
    eu_flow_change = eu["dk_fin"].sum()
    eu_flow_pct = eu_flow_change / eu_flow_base * 100 if eu_flow_base else np.nan

    render_logo(LOGO_PATH)
    st.title("CMU Simulation Dashboard")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(
            "EU GDP change",
            format_usd_mn(eu_gdp_cmu - eu_gdp_base),
            metric_delta(eu_gdp_pct),
        )
    with metric_cols[1]:
        st.metric(
            "EU net portfolio flow",
            format_usd_mn(eu_flow_change),
            metric_delta(eu_flow_pct),
        )
    with metric_cols[2]:
        st.metric("Countries with GDP gain", f"{int((selected['dy_pct'] > 0).sum())}/{len(selected)}")
    with metric_cols[3]:
        st.metric("Countries with inflow gain", f"{int((selected['dk_fin'] > 0).sum())}/{len(selected)}")

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

    left, right = st.columns(2, gap="small")
    with left:
        st.plotly_chart(
            make_choropleth(
                selected,
                geojson,
                "dy_pct",
                "Change in GDP",
                "Delta GDP (%)",
                gdp_hover,
            ),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    with right:
        st.plotly_chart(
            make_choropleth(
                selected,
                geojson,
                "dk_fin_pct",
                "Change in Net Portfolio Flow",
                "Delta flow (%)",
                flow_hover,
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
