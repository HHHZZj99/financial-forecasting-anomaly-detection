# Financial Forecasting & Anomaly Detection

A reproducible portfolio project forecasting daily revenue and flagging unusual observations with Python, pandas, scikit-learn, and pytest. All data is synthetic and generated for this project.

## Business problem
Finance and commercial teams need an estimate of expected revenue and a way to prioritise unusual movements for investigation. This project builds a forecasting baseline, evaluates it on a chronological holdout period, and flags potential anomalies without treating alerts as proof of an error.

## What it does
- Generates two years of synthetic daily revenue with trend, weekly seasonality, noise, and planted unusual movements.
- Creates lag, rolling-window, and calendar features without using future information.
- Compares a `HistGradientBoostingRegressor` with a seven-day seasonal-naive baseline.
- Evaluates holdout forecasts using MAE, RMSE, and WAPE.
- Uses `IsolationForest` to flag unusual forecast residuals.
- Writes forecast results, anomaly alerts, and a JSON summary to `reports/`.
- Runs automated checks through GitHub Actions.

## Quick start
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python src/generate_data.py
python src/run_analysis.py
python -m pytest -q
```

## Project structure
```text
data/raw/daily_revenue.csv
reports/forecast_results.csv
reports/anomaly_alerts.csv
reports/model_summary.json
src/generate_data.py
src/run_analysis.py
tests/test_analysis.py
.github/workflows/ci.yml
```

## Outputs
- `forecast_results.csv`: actual revenue, both forecasts, residual, anomaly score, and evaluation-only event flag.
- `anomaly_alerts.csv`: observations flagged for investigation, ranked by anomaly score.
- `model_summary.json`: train/test sizes, model metrics, and anomaly-detection counts.

## Method
1. Generate deterministic fictional daily revenue.
2. Keep the time order and use the earliest 80% of observations for training.
3. Train a gradient-boosting regressor using past revenue lags, rolling statistics, day of week, and time index.
4. Compare against a seasonal-naive forecast using revenue from seven days earlier.
5. Score absolute forecast residual patterns with Isolation Forest on the holdout set.

## Limitations
This is a portfolio demonstration, not a production financial control. Data is synthetic; results do not represent a real business. Alerts are investigation candidates, not confirmed fraud or accounting errors. The current version does not model holidays, promotions, product mix, FX effects, missing business days, or approval workflows.

## Potential extensions
Rolling-origin backtesting, prediction intervals, ARIMA/SARIMAX comparison, calendar-event features, and a finance-facing dashboard.
