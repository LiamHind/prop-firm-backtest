from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_intraday_data(
    output_path: str | Path,
    start_date: str = "2024-01-02",
    sessions: int = 80,
    bars_per_session: int = 180,
    seed: int = 7,
    instrument: str = "MES",
) -> Path:
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)
    rows = []
    base_price = 5000.0
    current_day = start
    generated_sessions = 0
    while generated_sessions < sessions:
        if current_day.weekday() >= 5:
            current_day += timedelta(days=1)
            continue
        session_open = datetime.combine(current_day.date(), datetime.strptime("09:00", "%H:%M").time())
        price = base_price + rng.normal(0, 8)
        day_drift = rng.normal(0.015, 0.08)
        for bar in range(bars_per_session):
            ts = session_open + timedelta(minutes=bar)
            noise = rng.normal(0, 0.7)
            open_price = price
            close_price = price + noise + day_drift
            high_price = max(open_price, close_price) + abs(rng.normal(0, 0.35))
            low_price = min(open_price, close_price) - abs(rng.normal(0, 0.35))
            volume = int(max(1, rng.normal(250, 75)))
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts, tz="America/New_York").tz_convert("UTC").isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                    "instrument": instrument,
                }
            )
            price = close_price
        base_price = price + rng.normal(0, 4)
        generated_sessions += 1
        current_day += timedelta(days=1)
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
