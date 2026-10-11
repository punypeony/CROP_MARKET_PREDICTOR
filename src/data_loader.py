from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "wfp_food_prices_phl_processed.csv"
)

REQUIRED_COLUMNS = {
    "date",
    "admin1",
    "admin2",
    "market",
    "latitude",
    "longitude",
    "category",
    "commodity",
    "unit",
    "priceflag",
    "pricetype",
    "price",
    "year",
    "month",
    "quarter",
}

# Prefer exact observations and use a broader WFP observation class only when
# an exact-actual series does not exist for the compatible unit/price type.
PRICEFLAG_PREFERENCE = ("actual", "actual,aggregate", "aggregate")


@st.cache_data(show_spinner="Loading WFP price data...")
def load_processed_data():
    """Load, validate, normalize, and cache the processed WFP dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in (
        "latitude",
        "longitude",
        "price",
        "year",
        "month",
        "quarter",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data
def get_available_commodities(df):
    """Return sorted commodities that have valid price records."""
    values = df.loc[df["price"].notna(), "commodity"]
    return sorted(values.dropna().unique().tolist())


@st.cache_data
def get_available_markets(df, commodity):
    """Return sorted markets with valid records for one commodity."""
    values = df.loc[
        (df["commodity"] == commodity) & df["price"].notna(),
        "market",
    ]
    return sorted(values.dropna().unique().tolist())


@st.cache_data
def get_market_details(df, commodity, market):
    """Return the established market features used for model prediction."""
    subset = df.loc[
        (df["commodity"] == commodity)
        & (df["market"] == market)
        & df["date"].notna()
    ]
    if subset.empty:
        raise ValueError("No WFP records were found for this selection.")

    # Preserve the existing forecast workflow: categorical fields come from
    # the latest row and coordinates use historical medians.
    latest = subset.loc[subset["date"].idxmax()]
    return {
        "admin1": latest["admin1"],
        "admin2": latest["admin2"],
        "latitude": float(subset["latitude"].median()),
        "longitude": float(subset["longitude"].median()),
        "category": latest["category"],
        "unit": latest["unit"],
        "priceflag": latest["priceflag"],
        "pricetype": latest["pricetype"],
    }


@st.cache_data
def get_historical_series_details(df, commodity, market):
    """Resolve one compatible unit, price type, and WFP price flag."""
    forecast_details = get_market_details(df, commodity, market)
    compatible = df.loc[
        (df["commodity"] == commodity)
        & (df["market"] == market)
        & (df["unit"] == forecast_details["unit"])
        & (df["pricetype"] == forecast_details["pricetype"])
        & df["price"].notna()
        & df["date"].notna()
    ]
    if compatible.empty:
        raise ValueError(
            "No valid WFP price records were found for this selection."
        )

    flags = set(compatible["priceflag"].dropna().unique())
    selected_flag = next(
        (flag for flag in PRICEFLAG_PREFERENCE if flag in flags),
        None,
    )
    if selected_flag is None:
        raise ValueError(
            "No supported WFP observation classification was found."
        )

    return {
        "unit": forecast_details["unit"],
        "pricetype": forecast_details["pricetype"],
        "priceflag": selected_flag,
    }


@st.cache_data
def get_compatible_historical_records(df, commodity, market):
    """Return valid records from exactly one comparable WFP price series."""
    details = get_historical_series_details(df, commodity, market)
    records = df.loc[
        (df["commodity"] == commodity)
        & (df["market"] == market)
        & (df["unit"] == details["unit"])
        & (df["pricetype"] == details["pricetype"])
        & (df["priceflag"] == details["priceflag"])
        & df["price"].notna()
        & df["date"].notna(),
        ["date", "year", "month", "price"],
    ].copy()
    return records, details


@st.cache_data
def get_available_years(df, commodity, market):
    """Return years available in the selected compatible WFP series."""
    records, _ = get_compatible_historical_records(
        df, commodity, market
    )
    years = pd.to_numeric(records["year"], errors="coerce").dropna()
    return sorted(years.astype(int).unique().tolist())
