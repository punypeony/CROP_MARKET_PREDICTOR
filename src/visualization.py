import pandas as pd
import plotly.graph_objects as go


ACCENT_BLUE = "#3B82F6"
PREDICTION_GRAY = "#CBD5E1"
GRID_COLOR = "#27272A"
AXIS_COLOR = "#3F3F46"
TEXT_COLOR = "#F5F5F5"
SECONDARY_TEXT = "#A1A1AA"
HOVER_BACKGROUND = "#202024"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_ABBREVIATIONS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _price_axis_title(unit):
    """Return a concise price-axis label for the selected WFP unit."""
    if not unit:
        return "Price (PHP)"
    if str(unit).upper() == "KG":
        return "Price (PHP/kg)"
    if str(unit).lower() == "unit":
        return "Price (PHP/unit)"
    return f"Price (PHP/{unit})"


def _apply_price_layout(fig, title, xaxis_title, unit, showlegend=False):
    """Apply the shared dark visual system to a price chart."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": TEXT_COLOR},
        },
        template="plotly_dark",
        height=500,
        margin={"l": 26, "r": 20, "t": 78, "b": 34},
        hovermode="closest",
        hoverlabel={
            "bgcolor": HOVER_BACKGROUND,
            "bordercolor": AXIS_COLOR,
            "font": {"color": TEXT_COLOR, "size": 13},
        },
        showlegend=showlegend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": SECONDARY_TEXT},
        },
        xaxis_title=xaxis_title,
        yaxis_title=_price_axis_title(unit),
        font={
            "family": (
                "Inter, Segoe UI, ui-sans-serif, sans-serif"
            ),
            "color": TEXT_COLOR,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showgrid=False,
        title={"font": {"color": SECONDARY_TEXT}},
        linecolor=AXIS_COLOR,
        zerolinecolor=AXIS_COLOR,
        tickfont={"color": SECONDARY_TEXT},
    )
    fig.update_yaxes(
        rangemode="tozero",
        tickprefix="₱",
        tickformat=",.2f",
        gridcolor=GRID_COLOR,
        gridwidth=1,
        title={"font": {"color": SECONDARY_TEXT}},
        zerolinecolor=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        tickfont={"color": SECONDARY_TEXT},
    )
    return fig


def create_forecast_chart(forecast_df, year, unit=None):
    """Create a January-to-December Random Forest forecast chart."""
    chart_df = forecast_df.copy()
    chart_df["month_name"] = pd.Categorical(
        chart_df["month_name"], categories=MONTHS, ordered=True
    )
    chart_df = chart_df.sort_values("month_name")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["month_name"],
        y=chart_df["predicted_price"],
        mode="lines+markers",
        name="Random Forest prediction",
        line={"color": ACCENT_BLUE, "width": 2.5},
        marker={"color": ACCENT_BLUE, "size": 6},
        hovertemplate=(
            "%{x}<br>Predicted Price: ₱%{y:,.2f}"
            "<extra>Random Forest prediction</extra>"
        ),
    ))
    fig.update_xaxes(categoryorder="array", categoryarray=MONTHS)
    return _apply_price_layout(
        fig,
        f"Monthly Price Forecast — {int(year)}",
        "Month",
        unit,
    )


def _comparison_hover_text(row, aggregation):
    """Build a complete comparison tooltip for one displayed period."""
    if aggregation == "Monthly Average":
        period = row["period"].strftime("%B %Y")
        actual_label = "Actual WFP Monthly Average"
        predicted_label = "Random Forest Predicted Price"
    else:
        period = str(int(row["year"]))
        actual_label = "Actual WFP Yearly Average"
        predicted_label = "Random Forest Predicted Average"
    actual = (
        f"₱{row['actual_price']:,.2f}"
        if pd.notna(row["actual_price"])
        else "Not available"
    )
    difference = (
        f"₱{row['difference']:+,.2f}"
        if pd.notna(row["difference"])
        else "Not available"
    )
    return (
        f"{period}<br>{actual_label}: {actual}"
        f"<br>{predicted_label}: ₱{row['predicted_price']:,.2f}"
        f"<br>Difference: {difference}"
        f"<br>Observations: {int(row['observation_count'])}"
    )


def _apply_monthly_comparison_ticks(fig, periods):
    """Show readable date ticks without changing the monthly trace data."""
    periods = pd.to_datetime(periods)
    if periods.empty:
        return fig

    first_period = periods.min().to_period("M").to_timestamp()
    last_period = periods.max().to_period("M").to_timestamp()
    first_year = int(first_period.year)
    last_year = int(last_period.year)

    if first_year == last_year:
        tickvals = pd.date_range(
            f"{first_year}-01-01",
            f"{first_year}-12-01",
            freq="MS",
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=MONTH_ABBREVIATIONS,
            tickangle=0,
            range=[
                pd.Timestamp(first_year, 1, 1),
                pd.Timestamp(first_year, 12, 31),
            ],
        )
        return fig

    month_count = (
        (last_period.year - first_period.year) * 12
        + last_period.month
        - first_period.month
        + 1
    )
    if month_count <= 48:
        tick_interval = 3
    elif month_count <= 96:
        tick_interval = 6
    else:
        tick_interval = 12

    tickvals = pd.date_range(
        f"{first_year}-01-01",
        f"{last_year}-12-01",
        freq=f"{tick_interval}MS",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals.strftime("%b %Y"),
        tickangle=0,
    )
    return fig


def create_comparison_chart(
    comparison_df,
    aggregation,
    unit,
    context_title,
):
    """Plot actual WFP prices and comparable RF predictions together."""
    chart_df = comparison_df.copy()
    x_column = "period" if aggregation == "Monthly Average" else "year"
    chart_df["hover_text"] = chart_df.apply(
        _comparison_hover_text,
        axis=1,
        aggregation=aggregation,
    )
    marker_mode = (
        "lines+markers"
        if len(chart_df) <= 24
        else "lines"
    )
    if aggregation == "Monthly Average":
        actual_name = "Actual WFP Monthly Average"
        predicted_name = "Random Forest Predicted Price"
    else:
        actual_name = "Actual WFP Yearly Average"
        predicted_name = "Random Forest Predicted Average"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df[x_column],
        y=chart_df["actual_price"],
        mode=marker_mode,
        connectgaps=False,
        name=actual_name,
        line={"color": ACCENT_BLUE, "width": 2.5},
        marker={"color": ACCENT_BLUE, "size": 6},
        text=chart_df["hover_text"],
        hovertemplate=f"%{{text}}<extra>{actual_name}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df[x_column],
        y=chart_df["predicted_price"],
        mode=marker_mode,
        connectgaps=False,
        name=predicted_name,
        line={
            "color": PREDICTION_GRAY,
            "width": 2,
            "dash": "dash",
        },
        marker={"color": PREDICTION_GRAY, "size": 5},
        text=chart_df["hover_text"],
        hovertemplate=(
            f"%{{text}}<extra>{predicted_name}</extra>"
        ),
    ))

    if aggregation == "Yearly Average":
        fig.update_xaxes(dtick=1, tickformat="d")
        xaxis_title = "Year"
    else:
        _apply_monthly_comparison_ticks(fig, chart_df["period"])
        xaxis_title = "Month"

    return _apply_price_layout(
        fig,
        f"{context_title} — {aggregation}",
        xaxis_title,
        unit,
        showlegend=True,
    )
