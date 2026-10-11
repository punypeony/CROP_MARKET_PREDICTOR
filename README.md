# Crop Market Price Predictor

A Streamlit dashboard for exploring historical Philippine agricultural commodity prices and generating Random Forest price estimates from World Food Programme (WFP) market data.

The application provides two workflows:

- **Monthly Forecast** — generates January-to-December Random Forest estimates for a selected commodity, market, and year.
- **Historical Model Fit** — plots compatible Actual WFP averages and Random Forest predictions on the same chart and reports MAE, RMSE, R², and observation counts.

The interface uses a restrained dark analytics-dashboard theme. Actual observations and model predictions are always identified separately.

## Current status

The application runs locally and has been tested with the included processed dataset and saved model pipeline. A public deployment URL is not currently included.

| Component | Status |
|---|---|
| Processed WFP dataset | Included |
| Original WFP dataset | Included |
| Streamlit application | Working locally |
| Random Forest pipeline | Required locally; ignored by Git because of its size |
| Monthly and yearly historical comparison | Working |
| CSV downloads | Working |
| Public deployment | Not yet configured |

## Features

- Professional dark dashboard with responsive controls and compact metric cards
- Cached dataset and model loading through `st.cache_data` and `st.cache_resource`
- Cascading commodity and market selections
- Twelve-month Random Forest forecast view
- Monthly or yearly Actual WFP versus Random Forest comparison
- Missing actual observations preserved as missing; no interpolation or fabricated prices
- Single-year monthly charts with all labels from `Jan` through `Dec`
- Adaptive quarterly, semiannual, or annual labels for longer date ranges while retaining all monthly points
- Price axes that start at zero and display the selected WFP unit
- Downloadable forecast and comparison CSV files
- Clear historical-fit, held-out-period, and mixed-period context messages

## Data

The project uses Philippine food-price records from the World Food Programme, distributed through the Humanitarian Data Exchange (HDX):

- Source page: [WFP food prices for the Philippines](https://data.humdata.org/dataset/wfp-food-prices-for-philippines)
- Original file: `data/original/wfp_food_prices_phl.csv`
- Processed file: `data/processed/wfp_food_prices_phl_processed.csv`

The processed dataset contains:

| Item | Value |
|---|---:|
| Observations | 242,212 |
| Columns | 15 |
| Date coverage | 2000-01-15 to 2026-08-15 |
| First-level administrative areas | 17 |
| Second-level administrative areas | 79 |
| Markets | 108 |
| Commodities | 73 |

The regression target is `price`, interpreted as nominal Philippine pesos per value of the associated `unit` field. Units include `KG`, `Unit`, and `750ML`; prices from different units should not be treated as directly interchangeable.

The source dataset remains subject to the attribution, licensing, and usage terms provided by WFP and HDX. A standalone repository-level dataset license note has not yet been added, so consult the source metadata before redistributing the data.

## Model

The deployed artifact is a scikit-learn pipeline containing categorical preprocessing and a `RandomForestRegressor`.

### Input features

| Type | Features |
|---|---|
| Categorical | `admin1`, `admin2`, `market`, `category`, `commodity`, `unit`, `priceflag`, `pricetype` |
| Numerical | `latitude`, `longitude`, `year`, `month`, `quarter` |
| Target | `price` |

Categorical values are transformed using `OneHotEncoder(handle_unknown="ignore")`. Numerical values pass through the preprocessing pipeline unchanged.

### Compared algorithms

Exactly three traditional regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The modeling workflow uses a chronological 80:20 split and five-fold `TimeSeriesSplit` validation. Random Forest obtained the lowest recorded mean validation MAE.

| Algorithm | Mean validation MAE (PHP) |
|---|---:|
| Random Forest Regressor | **23.216** |
| Gradient Boosting Regressor | 26.146 |
| Linear Regression | 29.153 |

The current saved Random Forest pipeline reproduced the following results on the repository's later 48,443-row partition:

| Metric | Result |
|---|---:|
| MAE | PHP 17.271 |
| RMSE | PHP 27.946 |
| R² | 0.956 |

These metrics describe the current repository experiment, not guaranteed future-market accuracy. The existing notebooks also report the later partition for all three models, so the evaluation chronology should be revised before claiming that the final test data were viewed only once.

## Requirements

- Windows, macOS, or Linux
- Python compatible with the pinned packages in `requirements.txt`
- Sufficient storage and memory for the approximately 1.67-GB saved model
- The required model at `models/best_model.joblib`

The current development environment uses Python 3.14.6, Streamlit 1.65.0, pandas 3.0.6, scikit-learn 1.9.1, Plotly 7.1.0, and joblib 1.6.0.

## Local setup

Run the following commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow scripts for only the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required model artifact

The application expects:

```text
models/best_model.joblib
```

This file is approximately 1.67 GB and is excluded by `models/*.joblib` in `.gitignore`. A fresh Git clone will therefore not contain it. Obtain the artifact from the submitted project package and place it at the path above before starting the app.

You can inspect the artifact without retraining or overwriting it:

```powershell
python scripts/inspect_saved_model.py
```

The notebooks document preprocessing, EDA, modeling, validation, and feature importance. Their current stored cell order is not yet a clean one-click `Run All` pipeline, so review and normalize their execution order before using them to regenerate the model.

## Run the application

```powershell
python -m streamlit run app/app.py
```

Streamlit normally opens `http://localhost:8501`. Stop the server with `Ctrl+C`.

## Using the dashboard

### Monthly Forecast

1. Select **Monthly Forecast** under **Graph Type**.
2. Choose a commodity and compatible market.
3. Enter the forecast year.
4. Select **Generate Forecast**.
5. Review the chart, summary cards, table, and downloadable CSV.

Forecast values are model-generated point estimates. When the selected year is later than the available WFP records, the application displays an extrapolation warning.

### Historical Model Fit

1. Select **Historical Model Fit** under **Graph Type**.
2. Choose a commodity and compatible market.
3. Select the start year, end year, and aggregation.
4. Select **Generate Comparison**.
5. Review the shared Actual WFP and Random Forest chart, fit-aware metrics, comparison table, and downloadable CSV.

For monthly aggregation, compatible original WFP observations are grouped by calendar month using their arithmetic mean. One source observation remains that observation's monthly average; multiple compatible observations are averaged. Missing months are not filled with invented actual values.

The model produces estimates for the complete selected calendar range, while metrics use only periods where both an actual value and a prediction exist.

## Interpretation of comparison labels

- **Historical Model Fit / Historical Fit Metrics** — the selected full years overlap data used during model training.
- **Held-Out Historical Comparison / Held-Out Period Metrics** — the selected full years begin after the documented February 2025 training cutoff.
- **Historical Comparison / Mixed-Period Metrics** — the selection crosses the training and later-period boundary.

February 2025 was divided by a row-level split, so it is not represented as an independently held-out complete month.

## Project structure

```text
CROP_MARKET_PREDICTOR/
├── .streamlit/
│   └── config.toml              Dark Streamlit theme configuration
├── app/
│   └── app.py                   Application orchestration and controls
├── data/
│   ├── original/                Original WFP CSV
│   └── processed/               Validated processed CSV used by the app
├── models/
│   └── best_model.joblib        Saved preprocessing + Random Forest pipeline
├── notebooks/
│   ├── 00_preprocessing.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_model_validation.ipynb
│   └── 04_feature_importance.ipynb
├── paper/                       Research draft and evidence figures
├── scripts/
│   └── inspect_saved_model.py   Read-only model inspection utility
├── src/
│   ├── data_loader.py           Data loading, validation, and selectors
│   ├── forecast.py              Model loading and prediction inputs
│   ├── historical.py            Actual-price aggregation and metrics
│   ├── ui.py                    Streamlit result rendering and dark styling
│   └── visualization.py         Plotly charts and adaptive axes
├── README.md
└── requirements.txt
```

## Implementation notes

- `load_processed_data()` validates required columns and caches the CSV.
- `load_model()` caches the serialized pipeline and does not retrain it.
- Actual WFP series selection prefers the exact `actual` flag and uses broader compatible WFP classes only when necessary.
- Actual and predicted historical lines use the same commodity, market, unit, price type, price flag, and period context.
- Monthly charts retain every monthly prediction even when visible date labels are reduced for readability.
- The Y-axis uses `rangemode="tozero"` and does not hard-code an upper bound.
- No PSA or weather integration is present.

## Known limitations

- Predictions do not include price lags, inflation adjustment, weather, production volume, fuel prices, trade changes, or policy events.
- The app reports point estimates without prediction intervals.
- Nominal prices across 2000–2026 are not inflation-adjusted.
- Pooled performance combines different WFP units and commodity price scales.
- Public market coverage is uneven across dates, commodities, and locations.
- A selected future year represents feature-based extrapolation, not a recursive time-series forecast.
- The 1.67-GB model may exceed limits on lightweight hosting services and increases first-load time.
- The notebook execution order needs cleanup before the full workflow can be described as one-click reproducible.
- No public deployment URL is currently documented.


No repository-level `LICENSE` file is currently present. Do not assume permission to redistribute the project code, model, or third-party data beyond applicable institutional rules and the original data terms. WFP and HDX retain the rights and attribution requirements associated with the source data.
