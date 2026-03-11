from __future__ import annotations

from dataclasses import asdict
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from topstep_research.config import ExperimentConfig, StrategyConfig
from topstep_research.domain import BacktestResult, TradeRecord
from topstep_research.metrics import compute_trade_metrics
from topstep_research.strategies import (
    BaseStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumPullbackStrategy,
    RandomEntryStrategy,
    StrategyContext,
)


STRATEGY_MAP = {
    "momentum_pullback": MomentumPullbackStrategy,
    "random_entry": RandomEntryStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def _time_string_to_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _build_strategy(name: str, config: ExperimentConfig, seed_offset: int = 0) -> BaseStrategy:
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{name}'.")
    extra_params = config.benchmarks.get(name, {})
    context = StrategyContext(
        strategy_name=name,
        strategy_config=config.strategy,
        extra_params=extra_params,
        random_seed=config.experiment.random_seed + seed_offset,
    )
    return STRATEGY_MAP[name](context)


def _entry_allowed(ts: pd.Timestamp, config: StrategyConfig) -> bool:
    session_start = _time_string_to_time(config.session_start)
    session_end = _time_string_to_time(config.session_end)
    return session_start <= ts.timetz().replace(tzinfo=None) <= session_end


def _exit_fill(price: float, direction: int, slippage: float) -> float:
    return price - (direction * slippage)


def _pnl_dollars(entry: float, exit_price: float, direction: int, config: StrategyConfig) -> float:
    gross = (exit_price - entry) * direction * config.point_value * config.contracts
    fees = 2.0 * config.commission_per_contract * config.contracts
    return gross - fees


def _stop_target_hit(
    direction: int,
    low: float,
    high: float,
    stop_price: float,
    target_price: float | None,
) -> tuple[bool, bool]:
    if direction == 1:
        hit_stop = low <= stop_price
        hit_target = target_price is not None and high >= target_price
    else:
        hit_stop = high >= stop_price
        hit_target = target_price is not None and low <= target_price
    return hit_stop, hit_target


def _simulate_trade(
    day_df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    strategy_name: str,
    config: ExperimentConfig,
    trade_number: int,
) -> tuple[dict[str, Any] | None, int]:
    strategy = config.strategy
    entry_bar = day_df.iloc[entry_idx]
    atr = float(entry_bar.get("atr", np.nan))
    if np.isnan(atr) or atr <= 0:
        return None, entry_idx
    slippage = strategy.slippage_points_per_side
    entry_price = float(entry_bar["open"]) + (direction * slippage)
    stop_distance = atr * strategy.stop_atr
    target_distance = stop_distance * strategy.target_r if strategy.target_r > 0 else 0.0
    stop_price = entry_price - (direction * stop_distance)
    target_price = entry_price + (direction * target_distance) if target_distance > 0 else None
    trailing_stop = stop_price
    best_price = entry_price
    exit_reason = "session_close"
    exit_fill = _exit_fill(float(day_df.iloc[-1]["close"]), direction, slippage)
    exit_idx = len(day_df) - 1
    for i in range(entry_idx, len(day_df)):
        bar = day_df.iloc[i]
        hit_stop, hit_target = _stop_target_hit(direction, float(bar["low"]), float(bar["high"]), trailing_stop, target_price)
        if hit_stop and hit_target:
            first = strategy.intrabar_fill_priority
            exit_reason = "stop" if first == "stop_first" else "target"
            chosen_price = trailing_stop if exit_reason == "stop" else float(target_price)
            exit_fill = _exit_fill(chosen_price, direction, slippage)
            exit_idx = i
            break
        if hit_stop:
            exit_reason = "stop"
            exit_fill = _exit_fill(trailing_stop, direction, slippage)
            exit_idx = i
            break
        if hit_target:
            exit_reason = "target"
            exit_fill = _exit_fill(float(target_price), direction, slippage)
            exit_idx = i
            break
        if strategy.trailing:
            if direction == 1:
                best_price = max(best_price, float(bar["high"]))
                trailing_stop = max(trailing_stop, best_price - (stop_distance * strategy.trailing_r))
            else:
                best_price = min(best_price, float(bar["low"]))
                trailing_stop = min(trailing_stop, best_price + (stop_distance * strategy.trailing_r))
    trade_slice = day_df.iloc[entry_idx : exit_idx + 1]
    if direction == 1:
        adverse_points = max(0.0, entry_price - float(trade_slice["low"].min()))
        favorable_points = max(0.0, float(trade_slice["high"].max()) - entry_price)
    else:
        adverse_points = max(0.0, float(trade_slice["high"].max()) - entry_price)
        favorable_points = max(0.0, entry_price - float(trade_slice["low"].min()))
    pnl = _pnl_dollars(entry_price, exit_fill, direction, strategy)
    record = TradeRecord(
        strategy_name=strategy_name,
        config_id=config.config_id,
        day_id=str(entry_bar["session_date"]),
        entry_timestamp=str(entry_bar["timestamp"]),
        exit_timestamp=str(day_df.iloc[exit_idx]["timestamp"]),
        direction=int(direction),
        entry_price=float(entry_price),
        exit_price=float(exit_fill),
        stop_distance=float(stop_distance),
        target_distance=float(target_distance),
        pnl=float(pnl),
        mae=float(adverse_points * strategy.point_value * strategy.contracts),
        mfe=float(favorable_points * strategy.point_value * strategy.contracts),
        reason_for_exit=exit_reason,
        contracts=strategy.contracts,
        point_value=strategy.point_value,
        trade_number=trade_number,
        regime=str(entry_bar.get("regime", "unknown")),
    )
    return record.to_dict(), exit_idx


def _summarize_daily(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["session_date", "pnl", "trade_count", "largest_trade", "worst_trade", "regime"]
        )
    daily = (
        trades.groupby("day_id", as_index=False)
        .agg(
            pnl=("pnl", "sum"),
            trade_count=("pnl", "size"),
            largest_trade=("pnl", "max"),
            worst_trade=("pnl", "min"),
            regime=("regime", "first"),
        )
        .rename(columns={"day_id": "session_date"})
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    daily["cum_pnl"] = daily["pnl"].cumsum()
    return daily


def run_backtest(data: pd.DataFrame, strategy_name: str, config: ExperimentConfig, seed_offset: int = 0) -> BacktestResult:
    strategy = _build_strategy(strategy_name, config, seed_offset=seed_offset)
    prepared = strategy.prepare_features(data)
    signals = strategy.generate_signals(prepared)
    prepared = prepared.copy()
    prepared["signal"] = signals
    records: list[dict[str, Any]] = []
    global_trade_number = 0
    for day_id, day_df in prepared.groupby("session_date", sort=True):
        if day_id in set(config.strategy.no_trade_dates):
            continue
        day_df = day_df.reset_index(drop=True)
        daily_realized = 0.0
        trades_today = 0
        i = 0
        while i < len(day_df) - 1:
            ts = day_df.iloc[i]["timestamp"]
            entry_ts = day_df.iloc[i + 1]["timestamp"]
            if not _entry_allowed(ts, config.strategy) or not _entry_allowed(entry_ts, config.strategy):
                i += 1
                continue
            if trades_today >= config.strategy.max_trades_per_day:
                break
            if daily_realized <= config.strategy.daily_stop:
                break
            if daily_realized >= config.strategy.daily_profit_cap:
                break
            direction = int(day_df.iloc[i]["signal"])
            if direction == 0:
                i += 1
                continue
            trade, exit_idx = _simulate_trade(
                day_df=day_df,
                entry_idx=i + 1,
                direction=direction,
                strategy_name=strategy_name,
                config=config,
                trade_number=global_trade_number + 1,
            )
            if trade is None:
                i += 1
                continue
            records.append(trade)
            global_trade_number += 1
            trades_today += 1
            daily_realized += float(trade["pnl"])
            i = max(exit_idx + 1, i + 1)
    trades = pd.DataFrame.from_records(records)
    daily = _summarize_daily(trades)
    metrics = compute_trade_metrics(trades, daily)
    return BacktestResult(
        strategy_name=strategy_name,
        config_id=config.config_id,
        trades=trades,
        daily=daily,
        trade_metrics=metrics,
    )
