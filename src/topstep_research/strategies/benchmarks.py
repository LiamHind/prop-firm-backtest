from __future__ import annotations

import numpy as np
import pandas as pd

from topstep_research.indicators import average_true_range, higher_timeframe_trend, intraday_vwap
from topstep_research.strategies.base import BaseStrategy


class RandomEntryStrategy(BaseStrategy):
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        working["atr"] = average_true_range(working, self.strategy_config.atr_period)
        return working

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self.context.random_seed)
        signals = self.zero_signals(df)
        max_trades = self.strategy_config.max_trades_per_day
        for _, day_df in df.groupby("session_date", sort=True):
            if len(day_df) < 3:
                continue
            eligible_positions = np.arange(0, len(day_df) - 1)
            if len(eligible_positions) == 0:
                continue
            choose_count = min(max_trades, len(eligible_positions))
            chosen = rng.choice(eligible_positions, size=choose_count, replace=False)
            directions = rng.choice([-1, 1], size=choose_count)
            for position, direction in zip(chosen, directions):
                signals.loc[day_df.index[position]] = int(direction)
        return signals.astype(int)


class BreakoutStrategy(BaseStrategy):
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        lookback = int(self.context.extra_params.get("lookback_bars", 20))
        working["atr"] = average_true_range(working, self.strategy_config.atr_period)
        working["rolling_high"] = (
            working.groupby("session_date")["high"]
            .shift(1)
            .groupby(working["session_date"])
            .rolling(lookback, min_periods=lookback)
            .max()
            .reset_index(level=0, drop=True)
        )
        working["rolling_low"] = (
            working.groupby("session_date")["low"]
            .shift(1)
            .groupby(working["session_date"])
            .rolling(lookback, min_periods=lookback)
            .min()
            .reset_index(level=0, drop=True)
        )
        working["vwap"] = intraday_vwap(working)
        working["htf_trend"] = higher_timeframe_trend(
            working,
            higher_timeframe_minutes=self.strategy_config.higher_timeframe_minutes,
            lookback=self.strategy_config.higher_timeframe_lookback,
        )
        return working

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = self.zero_signals(df)
        long_signal = (df["close"] > df["rolling_high"]) & (df["htf_trend"] >= 0)
        short_signal = (df["close"] < df["rolling_low"]) & (df["htf_trend"] <= 0)
        signals = signals.mask(long_signal, 1)
        signals = signals.mask(short_signal, -1)
        return signals.fillna(0).astype(int)


class MeanReversionStrategy(BaseStrategy):
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        working["atr"] = average_true_range(working, self.strategy_config.atr_period)
        working["vwap"] = intraday_vwap(working)
        threshold = float(self.context.extra_params.get("distance_from_vwap_atr", 1.0))
        working["distance_from_vwap_atr"] = ((working["close"] - working["vwap"]) / working["atr"]).replace(
            [np.inf, -np.inf], np.nan
        )
        working["threshold"] = threshold
        return working

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = self.zero_signals(df)
        long_signal = df["distance_from_vwap_atr"] <= -df["threshold"]
        short_signal = df["distance_from_vwap_atr"] >= df["threshold"]
        signals = signals.mask(long_signal, 1)
        signals = signals.mask(short_signal, -1)
        return signals.fillna(0).astype(int)
