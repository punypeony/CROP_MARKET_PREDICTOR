from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"


@st.cache_resource(show_spinner="Loading the saved Random Forest model...")
def load_model():
    """Load and cache the saved preprocessing and Random Forest pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_prediction_input(market_details, periods):
    """Build model-ready rows for a sequence of monthly timestamps."""
    records = []
    for period in pd.DatetimeIndex(periods):
        records.append({
            "admin1": market_details["admin1"],
            "admin2": market_details["admin2"],
            "market": market_details["market"],
            "latitude": market_details["latitude"],
            "longitude": market_details["longitude"],
            "category": market_details["category"],
            "commodity": market_details["commodity"],
            "unit": market_details["unit"],
            "priceflag": market_details["priceflag"],
            "pricetype": market_details["pricetype"],
            "year": int(period.year),
            "month": int(period.month),
            "quarter": (int(period.month) - 1) // 3 + 1,
        })
    return pd.DataFrame(records)


def predict_period_prices(model, market_details, periods):
    """Predict prices for monthly timestamps using one compatible series."""
    period_index = pd.DatetimeIndex(periods)
    if period_index.empty:
        return pd.DataFrame(columns=["period", "predicted_price"])

    input_df = build_prediction_input(market_details, period_index)
    predictions = model.predict(input_df)
    return pd.DataFrame({
        "period": period_index,
        "predicted_price": predictions,
    })


def predict_monthly_prices(model, market_details, year):
    """Predict January through December for one forecast year."""
    periods = pd.date_range(
        f"{int(year)}-01-01",
        f"{int(year)}-12-01",
        freq="MS",
    )
    result = predict_period_prices(model, market_details, periods)
    result["month"] = result["period"].dt.month
    result["month_name"] = result["period"].dt.strftime("%B")
    return result[["month", "month_name", "predicted_price"]]
