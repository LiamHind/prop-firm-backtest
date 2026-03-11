from __future__ import annotations

from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd

from topstep_research.domain import CombineAttemptResult


def _longest_streak(values: pd.Series, positive: bool) -> int:
    if values.empty:
        return 0
    flags = values.gt(0) if positive else values.lt(0)
    longest = current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    equity = pnl_series.cumsum()
    peaks = equity.cummax()
    drawdowns = equity - peaks
    return float(drawdowns.min())


def compute_trade_metrics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "average_trades_per_day": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "expectancy_per_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "longest_losing_streak": 0,
            "longest_winning_streak": 0,
            "average_mae": 0.0,
            "average_mfe": 0.0,
            "total_pnl": 0.0,
        }
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    trade_count = len(trades)
    day_count = len(daily)
    return {
        "total_trades": int(trade_count),
        "average_trades_per_day": float(trade_count / max(day_count, 1)),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy_per_trade": float(trades["pnl"].mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
        "max_drawdown": _max_drawdown(trades["pnl"]),
        "longest_losing_streak": _longest_streak(trades["pnl"], positive=False),
        "longest_winning_streak": _longest_streak(trades["pnl"], positive=True),
        "average_mae": float(trades["mae"].mean()),
        "average_mfe": float(trades["mfe"].mean()),
        "total_pnl": float(trades["pnl"].sum()),
    }


def summarize_attempts(
    attempts: list[CombineAttemptResult],
    profit_target: float,
    assumed_pass_value: float = 0.0,
) -> dict[str, Any]:
    if not attempts:
        return {
            "attempt_count": 0,
            "pass_probability": 0.0,
            "fail_probability": 0.0,
            "unresolved_probability": 0.0,
            "average_days_to_pass": None,
            "median_days_to_pass": None,
            "average_days_to_fail": None,
            "median_days_to_fail": None,
            "largest_day_gt_half_target_frequency": 0.0,
            "expected_monthly_fees": 0.0,
            "expected_value_per_attempt": 0.0,
        }
    total = len(attempts)
    passed = [a for a in attempts if a.passed]
    failed = [a for a in attempts if a.failed]
    unresolved = [a for a in attempts if not a.resolved]
    pass_days = [a.days_used for a in passed]
    fail_days = [a.days_used for a in failed]
    fee_values = [a.monthly_fees_paid for a in attempts]
    pass_probability = len(passed) / total
    avg_fees = mean(fee_values) if fee_values else 0.0
    return {
        "attempt_count": total,
        "pass_probability": float(pass_probability),
        "fail_probability": float(len(failed) / total),
        "unresolved_probability": float(len(unresolved) / total),
        "average_days_to_pass": float(mean(pass_days)) if pass_days else None,
        "median_days_to_pass": float(median(pass_days)) if pass_days else None,
        "average_days_to_fail": float(mean(fail_days)) if fail_days else None,
        "median_days_to_fail": float(median(fail_days)) if fail_days else None,
        "largest_day_gt_half_target_frequency": float(
            mean([(a.best_day_profit >= (profit_target * 0.5)) for a in attempts])
        ),
        "expected_monthly_fees": float(avg_fees),
        "expected_value_per_attempt": float((pass_probability * assumed_pass_value) - avg_fees),
    }


def regime_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or "regime" not in daily.columns:
        return pd.DataFrame(columns=["regime", "days", "total_pnl", "avg_pnl", "passive_hit_rate"])
    summary = (
        daily.groupby("regime", as_index=False)
        .agg(days=("pnl", "size"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"))
        .sort_values("total_pnl", ascending=False)
        .reset_index(drop=True)
    )
    summary["passive_hit_rate"] = np.where(summary["avg_pnl"] > 0, 1.0, 0.0)
    return summary
