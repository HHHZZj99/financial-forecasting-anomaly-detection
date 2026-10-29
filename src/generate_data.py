from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "daily_revenue.csv"


def generate_data(n_days=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    dow = dates.dayofweek.to_numpy()
    weekly = np.select(
        [dow == 0, dow == 1, dow == 2, dow == 3, dow == 4, dow == 5, dow == 6],
        [-90, -40, 5, 45, 110, 180, 80], default=0
    )
    revenue = 1200 + 0.75*t + weekly + 90*np.sin(2*np.pi*t/365.25) + rng.normal(0, 65, n_days)
    injected = np.zeros(n_days, dtype=int)
    for idx, magnitude in zip([n_days-105, n_days-63, n_days-28, n_days-12], [420, -500, 550, -430]):
        if 0 <= idx < n_days:
            revenue[idx] += magnitude
            injected[idx] = 1
    return pd.DataFrame({
        "date": dates,
        "revenue": np.round(revenue, 2),
        "injected_event_for_evaluation_only": injected,
    })


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    generate_data().to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")
