import sys
from calendar import month_name
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import (
    get_available_commodities,
    get_available_markets,
    get_available_years,
    get_historical_series_details,
    get_market_details,
    load_processed_data,
)
from src.forecast import load_model, predict_monthly_prices
from src.historical import (
    build_historical_comparison,
    calculate_comparison_metrics,
    get_evaluation_context,
)
from src.ui import (
    apply_custom_styles,
    render_comparison_result,
    render_empty_state,
    render_forecast_result,
    render_page_header,
)


MONTH_NAMES = list(month_name[1:])


st.set_page_config(
    page_title="Crop Market Price Predictor",
    layout="wide",
)
apply_custom_styles()


def forecast_controls(df, commodities):
    """Render forecast selections and return their current state."""
    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        commodity = st.selectbox(
            "Commodity", commodities, key="forecast_commodity"
        )
    markets = get_available_markets(df, commodity)
    with control_2:
        market = st.selectbox(
            "Market", markets, key="forecast_market"
        )

    latest_year = int(df["year"].max())
    with control_3:
        year = int(st.number_input(
            "Forecast Year",
            min_value=int(df["year"].min()),
            max_value=latest_year + 10,
            value=latest_year + 1,
            step=1,
        ))
    with control_4:
        focus_month = st.selectbox(
            "Focus Month",
            MONTH_NAMES,
            key="forecast_focus_month",
            help=(
                "Highlights one month in the full-year forecast and shows "
                "its prediction in detail."
            ),
        )
    _, action = st.columns([2, 1])
    with action:
        generate = st.button(
            "Generate Forecast",
            type="primary",
            width="stretch",
            key="generate_forecast",
        )
    return {
        "commodity": commodity,
        "market": market,
        "year": year,
    }, focus_month, generate, latest_year


def comparison_controls(df, commodities):
    """Render historical comparison selections and validation state."""
    control_1, control_2 = st.columns(2)
    with control_1:
        commodity = st.selectbox(
            "Commodity", commodities, key="comparison_commodity"
        )
    markets = get_available_markets(df, commodity)
    with control_2:
        market = st.selectbox(
            "Market", markets, key="comparison_market"
        )

    try:
        details = get_historical_series_details(df, commodity, market)
        years = get_available_years(df, commodity, market)
    except Exception as exc:
        st.error(f"Could not resolve comparable observations: {exc}")
        return None, False

    if not years:
        st.warning("No comparable historical observations are available.")
        return None, False

    year_1, year_2, aggregation_column, month_column = st.columns(4)
    with year_1:
        start_year = st.selectbox(
            "Start Year",
            years,
            index=max(0, len(years) - 5),
            key="comparison_start_year",
        )
    with year_2:
        end_year = st.selectbox(
            "End Year",
            years,
            index=len(years) - 1,
            key="comparison_end_year",
        )
    with aggregation_column:
        aggregation = st.selectbox(
            "Aggregation",
            ["Monthly Average", "Yearly Average"],
            key="comparison_aggregation",
        )
    with month_column:
        selected_month = st.selectbox(
            "Month Filter",
            ["All Months", *MONTH_NAMES],
            key="comparison_month_filter",
            disabled=aggregation == "Yearly Average",
            help=(
                "For monthly comparisons, select one calendar month to "
                "compare it across the chosen years."
            ),
        )
    month_filter = (
        selected_month
        if aggregation == "Monthly Average"
        else "All Months"
    )

    invalid_range = start_year > end_year
    insufficient_periods = (
        start_year == end_year
        and (
            aggregation == "Yearly Average"
            or month_filter != "All Months"
        )
    )
    if invalid_range:
        st.warning("Start Year must be earlier than or equal to End Year.")
    elif insufficient_periods:
        st.warning(
            "Select at least two years when using a specific month or "
            "yearly aggregation."
        )
    st.caption(
        f"Comparable series: {details['unit']} | {details['pricetype']} | "
        f"WFP price flag: {details['priceflag']}"
    )
    _, action = st.columns([2, 1])
    with action:
        generate = st.button(
            "Generate Comparison",
            type="primary",
            width="stretch",
            disabled=invalid_range or insufficient_periods,
            key="generate_comparison",
        )
    return {
        "commodity": commodity,
        "market": market,
        "start_year": start_year,
        "end_year": end_year,
        "aggregation": aggregation,
        "month_filter": month_filter,
    }, generate


def generate_forecast(df, selection, latest_year):
    """Generate and store a forecast result for stable Streamlit reruns."""
    model = load_model()
    details = get_market_details(
        df, selection["commodity"], selection["market"]
    )
    prediction_details = {
        **details,
        "commodity": selection["commodity"],
        "market": selection["market"],
    }
    data = predict_monthly_prices(
        model, prediction_details, selection["year"]
    )
    st.session_state["forecast_result"] = {
        "selection": selection,
        "details": details,
        "data": data,
        "latest_data_year": latest_year,
    }


def generate_comparison(df, selection):
    """Generate and store a comparable actual-versus-predicted result."""
    model = load_model()
    data, details = build_historical_comparison(
        df=df,
        model=model,
        commodity=selection["commodity"],
        market=selection["market"],
        start_year=selection["start_year"],
        end_year=selection["end_year"],
        aggregation=selection["aggregation"],
        month=(
            MONTH_NAMES.index(selection["month_filter"]) + 1
            if selection["month_filter"] != "All Months"
            else None
        ),
    )
    if data.empty:
        st.session_state.pop("comparison_result", None)
        st.warning(
            "No comparable WFP observations are available for this period."
        )
        return


    st.session_state["comparison_result"] = {
        "selection": selection,
        "details": details,
        "data": data,
        "metrics": calculate_comparison_metrics(data),
        "evaluation": get_evaluation_context(
            selection["start_year"], selection["end_year"]
        ),
    }


render_page_header()

try:
    dataset = load_processed_data()
except Exception as exc:
    st.error(f"Could not load the processed WFP dataset: {exc}")
    st.stop()

commodities = get_available_commodities(dataset)
if not commodities:
    st.error("No valid commodities are available in the processed dataset.")
    st.stop()

with st.container(border=True):
    st.subheader("Graph Generator")
    graph_mode = st.radio(
        "Graph Type",
        ["Monthly Forecast", "Historical Comparison"],
        horizontal=True,
        format_func=lambda mode: (
            "Historical Model Fit"
            if mode == "Historical Comparison"
            else mode
        ),
        help=(
            "Monthly Forecast produces model predictions. Historical "
            "Comparison plots actual WFP values and model predictions together."
        ),
    )
    if graph_mode == "Monthly Forecast":
        selection, focus_month, generate, latest_year = forecast_controls(
            dataset, commodities
        )
    else:
        selection, generate = comparison_controls(dataset, commodities)

if graph_mode == "Monthly Forecast":
    if generate:
        try:
            generate_forecast(dataset, selection, latest_year)
        except Exception as exc:
            st.error(f"Could not generate the model forecast: {exc}")

    result = st.session_state.get("forecast_result")
    if result and result["selection"] == selection:
        render_forecast_result(result, focus_month)
    elif result:
        render_empty_state(
            "Selections changed",
            "Generate the forecast again to apply the current selections.",
        )
    else:
        render_empty_state(
            message=(
                "Select a commodity, market, and forecast year above, then "
                "generate a forecast to begin."
            )
        )
else:
    if selection is not None and generate:
        try:
            generate_comparison(dataset, selection)
        except Exception as exc:
            st.error(f"Could not generate the historical comparison: {exc}")

    result = st.session_state.get("comparison_result")
    if selection is not None and result and result["selection"] == selection:
        render_comparison_result(result)
    elif result:
        render_empty_state(
            "Selections changed",
            "Generate the comparison again to apply the current selections.",
        )
    else:
        render_empty_state(
            message=(
                "Select a commodity, market, period, and aggregation above, "
                "then generate a comparison to begin."
            )
        )
