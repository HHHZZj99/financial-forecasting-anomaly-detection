import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generate_data import generate_data
from src.run_analysis import add_features, metrics, run_analysis


def test_data_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_data(), generate_data())
    assert len(generate_data()) == 730
    assert generate_data()["injected_event_for_evaluation_only"].sum() == 4


def test_features_have_no_missing_values():
    featured = add_features(generate_data())
    assert not featured.isna().any().any()
    assert len(featured) == 716


def test_metrics_are_non_negative():
    result = metrics(pd.Series([100., 110., 90.]), pd.Series([105., 100., 95.]))
    assert result["mae"] >= 0 and result["rmse"] >= 0 and result["wape"] >= 0


def test_analysis_writes_reports(tmp_path):
    input_path = tmp_path / "daily_revenue.csv"
    generate_data().to_csv(input_path, index=False)
    summary = run_analysis(input_path, tmp_path / "reports")
    assert summary["rows_total"] == 730
    assert summary["test_rows"] > 0
    assert (tmp_path / "reports/forecast_results.csv").exists()
    assert (tmp_path / "reports/anomaly_alerts.csv").exists()
    assert (tmp_path / "reports/model_summary.json").exists()
