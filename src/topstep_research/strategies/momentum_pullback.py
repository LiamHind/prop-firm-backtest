from __future__ import annotations

import numpy as np
import pandas as pd

from topstep_research.indicators import average_true_range, higher_timeframe_trend, intraday_vwap
from topstep_research.strategies.base import BaseStrategy


class MomentumPullbackStrategy(BaseStrategy):
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        cfg = self.strategy_config
        working["atr"] = average_true_range(working, cfg.atr_period)
        working["vwap"] = intraday_vwap(working)
        working["vwap_slope"] = (
            working.groupby("session_date")["vwap"].diff(cfg.vwap_slope_lookback).fillna(0.0)
        )
        working["htf_trend"] = higher_timeframe_trend(
            working,
            higher_timeframe_minutes=cfg.higher_timeframe_minutes,
            lookback=cfg.higher_timeframe_lookback,
        )
        working["rolling_pullback_low"] = (
            working.groupby("session_date")["low"]
            .rolling(cfg.pullback_lookback_bars, min_periods=cfg.pullback_lookback_bars)
            .min()
            .reset_index(level=0, drop=True)
        )
        working["rolling_pullback_high"] = (
            working.groupby("session_date")["high"]
            .rolling(cfg.pullback_lookback_bars, min_periods=cfg.pullback_lookback_bars)
            .max()
            .reset_index(level=0, drop=True)
        )
        close_diff = working.groupby("session_date")["close"].diff()
        working["recent_down_bars"] = (
            (close_diff < 0)
            .groupby(working["session_date"])
            .rolling(cfg.pullback_lookback_bars, min_periods=cfg.pullback_lookback_bars)
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        working["recent_up_bars"] = (
            (close_diff > 0)
            .groupby(working["session_date"])
            .rolling(cfg.pullback_lookback_bars, min_periods=cfg.pullback_lookback_bars)
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        working["distance_to_vwap_atr"] = ((working["close"] - working["vwap"]) / working["atr"]).replace(
            [np.inf, -np.inf], np.nan
        )
        return working

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.strategy_config
        signals = self.zero_signals(df)
        confirmation_high = df["high"].shift(cfg.confirmation_lookback_bars)
        confirmation_low = df["low"].shift(cfg.confirmation_lookback_bars)
        pullback_long = (
            (df["rolling_pullback_low"] <= df["vwap"] + (df["atr"] * cfg.pullback_touch_tolerance_atr))
            & (df["recent_down_bars"] >= 1)
        )
        pullback_short = (
            (df["rolling_pullback_high"] >= df["vwap"] - (df["atr"] * cfg.pullback_touch_tolerance_atr))
            & (df["recent_up_bars"] >= 1)
        )
        long_signal = (
            (df["htf_trend"] == 1)
            & (df["close"] > df["vwap"])
            & (df["vwap_slope"] > 0)
            & pullback_long
            & (df["close"] > confirmation_high)
        )
        short_signal = (
            (df["htf_trend"] == -1)
            & (df["close"] < df["vwap"])
            & (df["vwap_slope"] < 0)
            & pullback_short
            & (df["close"] < confirmation_low)
        )
        signals = signals.mask(long_signal, 1)
        signals = signals.mask(short_signal, -1)
        return signals.fillna(0).astype(int)
