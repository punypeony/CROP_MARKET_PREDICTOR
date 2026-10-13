import re

import numpy as np
import pandas as pd
import streamlit as st

from src.visualization import (
    create_comparison_chart,
    create_forecast_chart,
)


def apply_custom_styles():
    """Apply the shared dark analytics-dashboard visual system."""
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0B0B0C;
            --card-bg: #151517;
            --card-secondary: #1A1A1D;
            --surface-elevated: #202024;
            --text-primary: #F5F5F5;
            --text-secondary: #A1A1AA;
            --border: #29292E;
            --border-strong: #3F3F46;
            --accent: #3B82F6;
            --accent-hover: #60A5FA;
        }

        .stApp {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text-primary);
            background: var(--app-bg);
        }

        [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
        }

        [data-testid="stHeader"] {
            background: rgba(11, 11, 12, 0.92);
            border-bottom: 1px solid rgba(41, 41, 46, 0.55);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1360px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1, h2, h3 {
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: clamp(2rem, 3vw, 2.65rem) !important;
            line-height: 1.1 !important;
            font-weight: 720 !important;
            margin-bottom: 0.35rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
            font-weight: 650 !important;
        }

        h3 {
            font-size: 1.1rem !important;
            font-weight: 620 !important;
        }

        p, [data-testid="stCaptionContainer"] {
            color: var(--text-secondary);
        }

        [data-testid="stCaptionContainer"] {
            line-height: 1.5;
        }

        label, [data-testid="stWidgetLabel"] p {
            color: #D4D4D8 !important;
            font-weight: 540;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: none;
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.2rem;
        }

        [data-testid="stMetric"] {
            min-height: 112px;
            background: var(--card-secondary);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: none;
            padding: 0.9rem 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: clamp(1.45rem, 2.2vw, 2rem);
            font-weight: 640;
            letter-spacing: -0.025em;
        }

        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] span {
            color: #FFFFFF !important;
        }

        [data-testid="stBaseButton-primary"] {
            min-height: 2.65rem;
            background: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 10px;
            box-shadow: none;
            font-weight: 600;
            transition: background 150ms ease, border-color 150ms ease;
        }

        [data-testid="stBaseButton-primary"]:hover {
            background: var(--accent-hover);
            border-color: var(--accent-hover);
            color: #FFFFFF !important;
        }

        [data-testid="stBaseButton-secondary"],
        [data-testid="stDownloadButton"] button {
            background: var(--surface-elevated);
            color: #E4E4E7;
            border: 1px solid var(--border-strong);
            border-radius: 10px;
            box-shadow: none;
        }

        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stDownloadButton"] button:hover {
            color: #FFFFFF;
            border-color: #52525B;
            background: #27272A;
        }

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            min-height: 2.55rem;
            color: var(--text-primary) !important;
            background: var(--surface-elevated) !important;
            border-color: var(--border-strong) !important;
            border-radius: 10px;
            box-shadow: none !important;
        }

        [data-baseweb="select"] > div:focus-within,
        [data-testid="stNumberInput"] > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        [data-baseweb="popover"], [role="listbox"] {
            background: var(--surface-elevated) !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--text-primary) !important;
        }

        [role="option"] {
            color: var(--text-primary) !important;
            background: var(--surface-elevated) !important;
        }

        [role="option"]:hover,
        [aria-selected="true"][role="option"] {
            background: #27272A !important;
        }

        [data-testid="stRadio"] label {
            color: #D4D4D8 !important;
        }

        [data-testid="stPlotlyChart"] {
            background: transparent;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] {
            background: var(--card-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stAlert"] {
            color: #E4E4E7;
            background: var(--card-secondary);
            border-color: var(--border-strong);
            border-radius: 12px;
        }

        [data-testid="stTooltipContent"] {
            color: var(--text-primary);
            background: var(--surface-elevated);
            border: 1px solid var(--border-strong);
        }

        [data-testid="stExpander"] {
            background: var(--card-bg);
            border-color: var(--border);
            border-radius: 12px;
        }

        hr {
            border-color: var(--border) !important;
        }

        .dashboard-header {
            max-width: 760px;
            margin-bottom: 1.5rem;
        }

        .dashboard-header p {
            margin: 0;
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .empty-state {
            padding: 2.4rem 1rem;
            text-align: center;
        }

        .empty-state h3 {
            margin: 0 0 0.45rem;
        }

        .empty-state p {
            max-width: 620px;
            margin: 0 auto;
            color: var(--text-secondary);
        }

        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1.4rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 14px;
            }

            [data-testid="stMetric"] {
                min-height: 96px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header():
    """Render the application identity and concise purpose statement."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>Crop Market Price Predictor</h1>
            <p>Historical WFP market-price analysis and Random Forest forecasting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_price(value):
    """Format a value as Philippine peso currency or an unavailable state."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"₱{value:,.2f}"


def slugify(value):
    """Convert selection text into a safe, readable filename component."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def render_metric_grid(metrics, columns_per_row=3):
    """Render responsive rows of lightweight native metric cards."""
    for offset in range(0, len(metrics), columns_per_row):
        row = metrics[offset:offset + columns_per_row]
        columns = st.columns(len(row))
        for column, (label, value) in zip(columns, row):
            with column:
                st.metric(label, value, border=True)


def render_empty_state(title="No graph generated", message=None):
    """Render a compact, useful state before results are available."""
    if message is None:
        message = (
            "Select a commodity, market, and period above, then generate a "
            "graph to begin."
        )
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="empty-state">
                <h3>{title}</h3>
                <p>{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_result(result):
    """Render a forecast graph, dashboard, table, and CSV download."""
    forecast_df = result["data"]
    selection = result["selection"]
    details = result["details"]

    st.subheader("Monthly Forecast")
    st.caption(
        f"{selection['commodity']} · {selection['market']} · "
        f"{selection['year']}"
    )
    st.caption(
        "Random Forest predictions; these values are not historical "
        "observations."
    )
    if selection["year"] > result["latest_data_year"]:
        st.warning(
            "The selected year is beyond the latest WFP record. Long-range "
            "model extrapolation should be interpreted cautiously."
        )

    with st.container(border=True):
        figure = create_forecast_chart(
            forecast_df, selection["year"], details["unit"]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="forecast_chart",
            config={"displaylogo": False},
            theme=None,
        )

    lowest = forecast_df.loc[forecast_df["predicted_price"].idxmin()]
    highest = forecast_df.loc[forecast_df["predicted_price"].idxmax()]
    st.subheader("Forecast Summary")
    render_metric_grid([
        (
            "Average Predicted Price",
            format_price(forecast_df["predicted_price"].mean()),
        ),
        (
            "Lowest Predicted Month",
            f"{lowest['month_name']} · {format_price(lowest['predicted_price'])}",
        ),
        (
            "Highest Predicted Month",
            f"{highest['month_name']} · {format_price(highest['predicted_price'])}",
        ),
        ("Forecast Year", str(selection["year"])),
    ], columns_per_row=4)
    st.caption(
        f"Commodity: {selection['commodity']} | Market: {selection['market']} | "
        f"Unit: {details['unit']} | Price Type: {details['pricetype']} | "
        "Data Type: Random Forest Model Prediction"
    )

    display_df = forecast_df[["month_name", "predicted_price"]].rename(
        columns={
            "month_name": "Month",
            "predicted_price": "Predicted Price (PHP)",
        }
    )
    export_df = forecast_df[["month", "predicted_price"]].copy()
    with st.container(border=True):
        st.subheader("Forecast Data")
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Predicted Price (PHP)": st.column_config.NumberColumn(
                    format="₱ %.2f"
                )
            },
        )
        filename = (
            f"{slugify(selection['commodity'])}_"
            f"{slugify(selection['market'])}_"
            f"{selection['year']}_forecast.csv"
        )
        st.download_button(
            "Download Forecast Data (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            on_click="ignore",
        )


def render_comparison_result(result):
    """Render same-graph actual/predicted history and fit-aware metrics."""
    comparison_df = result["data"]
    selection = result["selection"]
    details = result["details"]
    metrics = result["metrics"]
    context = result["evaluation"]
    aggregation = selection["aggregation"]

    st.subheader(context["title"])
    st.caption(
        f"{selection['commodity']} · {selection['market']} · "
        f"{selection['start_year']}–{selection['end_year']}"
    )
    st.caption(
        "Actual WFP observations and Random Forest predictions use matching "
        "commodity, market, unit, price type, price flag, and time period."
    )
    st.info(context["note"])
    if details["priceflag"] != "actual":
        st.warning(
            "No exact-actual flag is available for this compatible series. "
            f"The WFP '{details['priceflag']}' class is used without mixing "
            "observation classes."
        )

    with st.container(border=True):
        figure = create_comparison_chart(
            comparison_df,
            aggregation,
            details["unit"],
            context["title"],
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="comparison_chart",
            config={"displaylogo": False},
            theme=None,
        )

    st.subheader(context["metrics_heading"])
    render_metric_grid([
        ("Actual Average", format_price(metrics["actual_average"])),
        (
            "Predicted Average",
            format_price(metrics["predicted_average"]),
        ),
        ("MAE", format_price(metrics["mae"])),
        ("RMSE", format_price(metrics["rmse"])),
        (
            "R²",
            f"{metrics['r2']:.3f}"
            if not np.isnan(metrics["r2"])
            else "Not available",
        ),
        ("Observations", f"{metrics['observation_count']:,}"),
    ])
    st.caption(
        f"Comparable periods: {metrics['period_count']:,} | Selected Period: "
        f"{selection['start_year']}–{selection['end_year']} | "
        f"Data Source: WFP Historical Observations"
    )
    st.caption(
        f"Compatible series: {details['unit']} | {details['pricetype']} | "
        f"WFP price flag: {details['priceflag']} | Difference: predicted − actual"
    )

    if aggregation == "Monthly Average":
        display_df = comparison_df.copy()
        display_df["Month"] = display_df["period"].dt.strftime("%B %Y")
        display_df = display_df[
            [
                "Month",
                "actual_price",
                "predicted_price",
                "difference",
                "observation_count",
            ]
        ]
        export_df = comparison_df.copy()
        export_df.insert(0, "month", export_df["period"].dt.strftime("%Y-%m"))
        export_df = export_df[
            [
                "month",
                "actual_price",
                "predicted_price",
                "difference",
                "observation_count",
            ]
        ]
        aggregation_slug = "monthly_comparison"
    else:
        display_df = comparison_df[
            [
                "year",
                "actual_price",
                "predicted_price",
                "difference",
                "observation_count",
            ]
        ].rename(columns={"year": "Year"})
        export_df = comparison_df[
            [
                "year",
                "actual_price",
                "predicted_price",
                "difference",
                "observation_count",
            ]
        ].copy()
        aggregation_slug = "yearly_comparison"

    display_df = display_df.rename(columns={
        "actual_price": "Actual WFP Average",
        "predicted_price": "Random Forest Prediction",
        "difference": "Difference",
        "observation_count": "Observation Count",
    })
    price_columns = {
        name: st.column_config.NumberColumn(format="₱ %.2f")
        for name in (
            "Actual WFP Average",
            "Random Forest Prediction",
            "Difference",
        )
    }
    price_columns["Observation Count"] = st.column_config.NumberColumn(
        format="%d"
    )

    with st.container(border=True):
        st.subheader("Comparison Data")
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config=price_columns,
        )
        filename = (
            f"{slugify(selection['commodity'])}_"
            f"{slugify(selection['market'])}_"
            f"{selection['start_year']}_{selection['end_year']}_"
            f"{aggregation_slug}.csv"
        )
        st.download_button(
            "Download Comparison Data (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            on_click="ignore",
        )
