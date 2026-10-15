import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import (
    get_compatible_historical_records,
    get_market_details,
)
from src.forecast import predict_period_prices


# The saved-model notebooks use a chronological 80/20 split. The training
# portion spans 2000-01-15 through a row-level cutoff on 2025-02-15; the
# holdout begins on that same date. February 2025 is therefore ambiguous at
# monthly aggregation and is never described as independently held out.
TRAINING_LAST_FULL_YEAR = 2024
HOLDOUT_FIRST_FULL_YEAR = 2026


@st.cache_data
def get_monthly_actual_prices(
    df,
    commodity,
    market,
    start_year,
    end_year,
):
    """Aggregate compatible WFP observations by calendar month."""
    if int(start_year) > int(end_year):
        raise ValueError(
            "Start Year must be earlier than or equal to End Year."
        )

    records, details = get_compatible_historical_records(
        df, commodity, market
    )
    records = records.loc[
        records["year"].between(int(start_year), int(end_year))
    ].copy()
    if records.empty:
        return pd.DataFrame(), details

    records["period"] = (
        records["date"].dt.to_period("M").dt.to_timestamp()
    )
    actual = (
        records.groupby("period", as_index=False)
        .agg(
            actual_price=("price", "mean"),
            observation_count=("price", "size"),
        )
        .sort_values("period")
        .reset_index(drop=True)
    )
    actual["observation_count"] = actual["observation_count"].astype(int)
    return actual, details


def build_historical_comparison(
    df,
    model,
    commodity,
    market,
    start_year,
    end_year,
    aggregation,
    month=None,
):
    """Build comparable actual and Random Forest historical price series."""
    if aggregation not in {"Monthly Average", "Yearly Average"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if month is not None and int(month) not in range(1, 13):
        raise ValueError("Month must be between 1 and 12.")
    if aggregation == "Yearly Average" and month is not None:
        raise ValueError("A month filter requires monthly aggregation.")
    if int(start_year) == int(end_year) and (
        aggregation == "Yearly Average" or month is not None
    ):
        raise ValueError(
            "Select at least two years when using a specific month or "
            "yearly aggregation."
        )

    actual, series_details = get_monthly_actual_prices(
        df, commodity, market, start_year, end_year
    )
    if actual.empty:
        return pd.DataFrame(), series_details

    market_details = get_market_details(df, commodity, market)
    prediction_details = {
        **market_details,
        **series_details,
        "commodity": commodity,
        "market": market,
    }
    periods = pd.date_range(
        f"{int(start_year)}-01-01",
        f"{int(end_year)}-12-01",
        freq="MS",
    )
    predicted = predict_period_prices(
        model, prediction_details, periods
    )
    monthly = predicted.merge(actual, on="period", how="left")
    monthly["difference"] = (
        monthly["predicted_price"] - monthly["actual_price"]
    )
    monthly["observation_count"] = (
        monthly["observation_count"].fillna(0).astype(int)
    )

    if aggregation == "Monthly Average":
        if month is not None:
            monthly = monthly.loc[
                monthly["period"].dt.month == int(month)
            ].reset_index(drop=True)
        return monthly, series_details

    comparable = monthly.loc[monthly["actual_price"].notna()].copy()
    comparable["year"] = comparable["period"].dt.year
    comparable["actual_weighted"] = (
        comparable["actual_price"] * comparable["observation_count"]
    )
    comparable["predicted_weighted"] = (
        comparable["predicted_price"]
        * comparable["observation_count"]
    )
    yearly = (
        comparable.groupby("year", as_index=False)
        .agg(
            actual_total=("actual_weighted", "sum"),
            predicted_total=("predicted_weighted", "sum"),
            observation_count=("observation_count", "sum"),
        )
        .sort_values("year")
    )
    yearly["actual_price"] = (
        yearly["actual_total"] / yearly["observation_count"]
    )
    yearly["predicted_price"] = (
        yearly["predicted_total"] / yearly["observation_count"]
    )
    yearly["difference"] = (
        yearly["predicted_price"] - yearly["actual_price"]
    )
    yearly = yearly[
        [
            "year",
            "actual_price",
            "predicted_price",
            "difference",
            "observation_count",
        ]
    ]

    year_scaffold = pd.DataFrame({
        "year": range(int(start_year), int(end_year) + 1)
    })
    yearly = year_scaffold.merge(yearly, on="year", how="left")
    yearly["observation_count"] = (
        yearly["observation_count"].fillna(0).astype(int)
    )
    return yearly, series_details


def calculate_comparison_metrics(comparison_df):
    """Calculate metrics only where both actual and prediction are present."""
    matched = comparison_df.dropna(
        subset=["actual_price", "predicted_price"]
    )
    if matched.empty:
        return {
            "actual_average": np.nan,
            "predicted_average": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "observation_count": 0,
            "period_count": 0,
        }

    actual = matched["actual_price"]
    predicted = matched["predicted_price"]
    r2 = (
        r2_score(actual, predicted)
        if len(matched) >= 2 and actual.nunique() > 1
        else np.nan
    )
    return {
        "actual_average": float(actual.mean()),
        "predicted_average": float(predicted.mean()),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2) if not pd.isna(r2) else np.nan,
        "observation_count": int(matched["observation_count"].sum()),
        "period_count": int(len(matched)),
    }


def get_evaluation_context(start_year, end_year):
    """Describe training overlap without overstating model validation."""
    start_year = int(start_year)
    end_year = int(end_year)
    if end_year <= TRAINING_LAST_FULL_YEAR:
        return {
            "kind": "fit",
            "title": "Historical Model Fit",
            "metrics_heading": "Historical Fit Metrics",
            "note": (
                "The selected years overlap the data used to train the saved "
                "model. These metrics describe historical fit, not independent "
                "validation or future performance."
            ),
        }
    if start_year >= HOLDOUT_FIRST_FULL_YEAR:
        return {
            "kind": "held_out",
            "title": "Held-Out Historical Comparison",
            "metrics_heading": "Held-Out Period Metrics",
            "note": (
                "The selected full years fall after the model's February 2025 "
                "training cutoff. Metrics use only periods with WFP actuals."
            ),
        }
    return {
        "kind": "mixed",
        "title": "Historical Comparison",
        "metrics_heading": "Mixed-Period Metrics",
        "note": (
            "The selection crosses the original training/holdout boundary. "
            "February 2025 was split at row level, so these metrics combine "
            "historical fit and held-out periods and are not independent "
            "validation metrics."
        ),
    }
