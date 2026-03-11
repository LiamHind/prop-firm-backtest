from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from topstep_research.config import DataConfig
from topstep_research.indicators import assign_regimes


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def _load_frame(config: DataConfig) -> pd.DataFrame:
    path = Path(config.path)
    if config.format.lower() == "csv" or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if config.format.lower() == "parquet" or path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data format: {config.format}")


def _normalize_columns(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    rename_map = {source: target for source, target in config.column_map.items() if source in df.columns}
    normalized = df.rename(columns=rename_map).copy()
    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return normalized


def _normalize_timestamp(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    ts = pd.to_datetime(df[config.timestamp_column], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ZoneInfo(config.source_timezone))
    ts = ts.dt.tz_convert(ZoneInfo(config.exchange_timezone))
    normalized = df.copy()
    normalized["timestamp"] = ts
    return normalized


def _apply_filters(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    filtered = df.copy()
    if "instrument" in filtered.columns:
        filtered = filtered[filtered["instrument"].fillna(config.instrument) == config.instrument]
    if config.session_start and config.session_end:
        filtered = filtered.set_index("timestamp")
        filtered = filtered.between_time(config.session_start, config.session_end)
        filtered = filtered.reset_index()
    if filtered.empty:
        return filtered
    filtered["session_date"] = filtered["timestamp"].dt.strftime("%Y-%m-%d")
    if config.exclude_event_dates:
        excluded = set(config.exclude_event_dates)
        filtered = filtered[~filtered["session_date"].isin(excluded)]
    if config.event_calendar_path:
        event_df = pd.read_csv(config.event_calendar_path)
        if "date" not in event_df.columns:
            raise ValueError("Event calendar must contain a 'date' column.")
        event_df["date"] = pd.to_datetime(event_df["date"]).dt.strftime("%Y-%m-%d")
        if config.exclude_event_tags and "event_name" in event_df.columns:
            event_df = event_df[event_df["event_name"].isin(config.exclude_event_tags)]
        excluded = set(event_df["date"])
        filtered = filtered[~filtered["session_date"].isin(excluded)]
    return filtered


def load_market_data(config: DataConfig) -> pd.DataFrame:
    raw = _load_frame(config)
    normalized = _normalize_columns(raw, config)
    normalized = _normalize_timestamp(normalized, config)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    filtered = _apply_filters(normalized, config)
    if filtered.empty:
        raise ValueError("No market data remained after applying filters.")
    filtered["time"] = filtered["timestamp"].dt.strftime("%H:%M")
    filtered["date"] = filtered["timestamp"].dt.date
    filtered["session_date"] = filtered["timestamp"].dt.strftime("%Y-%m-%d")
    filtered["regime"] = assign_regimes(filtered)
    return filtered.reset_index(drop=True)
