from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    price_volume = typical_price * df["volume"]
    cumulative_pv = price_volume.groupby(df["session_date"]).cumsum()
    cumulative_volume = df["volume"].groupby(df["session_date"]).cumsum().replace(0, np.nan)
    return cumulative_pv / cumulative_volume


def higher_timeframe_trend(
    df: pd.DataFrame,
    higher_timeframe_minutes: int,
    lookback: int,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="int64")
    outputs = []
    for _, day_df in df.groupby("session_date", sort=True):
        working = day_df.set_index("timestamp").copy()
        higher_tf = working.resample(f"{higher_timeframe_minutes}min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        higher_tf = higher_tf.dropna()
        ema = higher_tf["close"].ewm(span=max(lookback, 2), adjust=False).mean()
        slope = ema.diff()
        trend = pd.Series(0, index=higher_tf.index, dtype="int64")
        trend = trend.where(~((higher_tf["close"] > ema) & (slope > 0)), 1)
        trend = trend.where(~((higher_tf["close"] < ema) & (slope < 0)), -1)
        trend = trend.reindex(working.index, method="ffill").fillna(0)
        outputs.append(trend.reset_index(drop=True))
    if not outputs:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    return pd.concat(outputs, ignore_index=True)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def assign_regimes(df: pd.DataFrame) -> pd.Series:
    daily = (
        df.groupby("session_date")
        .agg(
            day_open=("open", "first"),
            day_close=("close", "last"),
            day_high=("high", "max"),
            day_low=("low", "min"),
        )
        .sort_index()
    )
    daily["day_range"] = daily["day_high"] - daily["day_low"]
    daily["vol_median"] = daily["day_range"].rolling(20, min_periods=5).median()
    daily["high_vol"] = daily["day_range"] >= daily["vol_median"].fillna(daily["day_range"].median())
    daily["direction"] = np.where(daily["day_close"] >= daily["day_open"], "up", "down")
    daily["volatility"] = np.where(daily["high_vol"], "highvol", "lowvol")
    daily["regime"] = daily["volatility"] + "_" + daily["direction"]
    return df["session_date"].map(daily["regime"]).fillna("unknown")
