import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/daily_revenue.csv"
REPORTS = ROOT / "reports"


def add_features(df):
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["time_index"] = np.arange(len(out))
    for lag in (1, 7, 14):
        out[f"revenue_lag_{lag}"] = out["revenue"].shift(lag)
    out["rolling_mean_7"] = out["revenue"].shift(1).rolling(7).mean()
    out["rolling_std_7"] = out["revenue"].shift(1).rolling(7).std()
    return out.dropna().reset_index(drop=True)


def metrics(actual, predicted):
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denom = float(np.abs(actual).sum())
    wape = float(np.abs(actual - predicted).sum() / denom) if denom else float("nan")
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "wape": round(wape, 5)}


def run_analysis(input_path=INPUT, reports_dir=REPORTS):
    df = pd.read_csv(input_path, parse_dates=["date"])
    required = {"date", "revenue", "injected_event_for_evaluation_only"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["date"].isna().any() or df["revenue"].isna().any():
        raise ValueError("Date and revenue must not contain missing values.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Input must be sorted by date.")

    featured = add_features(df)
    cols = ["day_of_week", "time_index", "revenue_lag_1", "revenue_lag_7",
            "revenue_lag_14", "rolling_mean_7", "rolling_std_7"]
    split = int(len(featured) * 0.8)
    if split < 30 or len(featured)-split < 10:
        raise ValueError("Dataset is too small for the configured time split.")
    train, test = featured.iloc[:split].copy(), featured.iloc[split:].copy()

    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06,
                                          max_leaf_nodes=15, random_state=42)
    model.fit(train[cols], train["revenue"])
    test["model_forecast"] = model.predict(test[cols])
    test["seasonal_naive_forecast"] = test["revenue_lag_7"]
    test["model_residual"] = test["revenue"] - test["model_forecast"]
    test["absolute_residual"] = test["model_residual"].abs()

    model_metrics = metrics(test["revenue"], test["model_forecast"])
    naive_metrics = metrics(test["revenue"], test["seasonal_naive_forecast"])

    train_residual = train["revenue"].to_numpy() - model.predict(train[cols])
    anomaly = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    anomaly.fit(np.column_stack([train_residual, np.abs(train_residual)]))
    residual_features = np.column_stack([test["model_residual"], test["absolute_residual"]])
    test["anomaly_score"] = anomaly.decision_function(residual_features)
    test["is_anomaly"] = anomaly.predict(residual_features) == -1

    reports_dir.mkdir(parents=True, exist_ok=True)
    test[["date", "revenue", "model_forecast", "seasonal_naive_forecast",
          "model_residual", "anomaly_score", "is_anomaly",
          "injected_event_for_evaluation_only"]].to_csv(reports_dir/"forecast_results.csv", index=False)
    test.loc[test["is_anomaly"], ["date", "revenue", "model_forecast", "model_residual",
          "anomaly_score", "injected_event_for_evaluation_only"]].sort_values(
          "anomaly_score").to_csv(reports_dir/"anomaly_alerts.csv", index=False)

    injected = int(test["injected_event_for_evaluation_only"].sum())
    detected = int(((test["injected_event_for_evaluation_only"] == 1) & test["is_anomaly"]).sum())
    summary = {
        "dataset": "synthetic_daily_revenue",
        "rows_total": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "forecast_metrics": {"hist_gradient_boosting": model_metrics, "seasonal_naive_7_day": naive_metrics},
        "anomaly_detection": {"alerts_in_test_period": int(test["is_anomaly"].sum()),
                              "injected_events_in_test_period": injected,
                              "injected_events_flagged": detected,
                              "note": "Alerts are investigation candidates, not confirmed errors."}
    }
    (reports_dir/"model_summary.json").write_text(json.dumps(summary, indent=2)+"\\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    if not INPUT.exists():
        raise FileNotFoundError("Run `python src/generate_data.py` first.")
    print(json.dumps(run_analysis(), indent=2))
