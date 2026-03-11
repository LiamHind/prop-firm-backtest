from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from topstep_research.config import TopstepConfig
from topstep_research.domain import CombineAttemptResult


def _effective_profit_target(
    base_profit_target: float,
    best_day_profit: float,
    consistency_threshold: float,
    consistency_buffer: float,
) -> float:
    if best_day_profit <= 0 or consistency_threshold <= 0:
        return base_profit_target
    consistency_implied = (best_day_profit / consistency_threshold) + consistency_buffer
    return max(base_profit_target, consistency_implied)


def _month_key(value: str) -> str | None:
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m")
    except ValueError:
        return None


def simulate_attempt(
    daily: pd.DataFrame,
    topstep: TopstepConfig,
    start_index: int = 0,
) -> CombineAttemptResult:
    cumulative_pnl = 0.0
    high_water_mark = 0.0
    best_day_profit = 0.0
    days_used = 0
    resolved = False
    passed = False
    failed = False
    start_day = None
    end_day = None
    path: list[dict[str, object]] = []
    months_seen: set[str] = set()
    monthly_fees_paid = 0.0
    ending_loss_limit = -topstep.max_loss_buffer
    violating_daily_loss = False
    for _, row in daily.iloc[start_index:].iterrows():
        session_date = str(row["session_date"])
        pnl = float(row["pnl"])
        if start_day is None:
            start_day = session_date
        month_key = _month_key(session_date)
        if month_key is not None and month_key not in months_seen:
            months_seen.add(month_key)
            monthly_fees_paid += topstep.monthly_subscription_fee
        if month_key is None and (days_used % max(topstep.trading_days_per_month, 1) == 0):
            monthly_fees_paid += topstep.monthly_subscription_fee
        days_used += 1
        cumulative_pnl += pnl
        high_water_mark = max(high_water_mark, cumulative_pnl)
        best_day_profit = max(best_day_profit, pnl)
        profit_target_required = _effective_profit_target(
            topstep.base_profit_target,
            best_day_profit,
            topstep.consistency_threshold,
            topstep.consistency_buffer,
        )
        if topstep.loss_limit_mode == "trailing_balance":
            ending_loss_limit = high_water_mark - topstep.max_loss_buffer
        elif topstep.loss_limit_mode == "static_from_start":
            ending_loss_limit = -topstep.max_loss_buffer
        else:
            raise ValueError(f"Unsupported loss limit mode: {topstep.loss_limit_mode}")
        if topstep.model_daily_loss_limit and topstep.daily_loss_limit is not None and pnl <= topstep.daily_loss_limit:
            violating_daily_loss = True
        passed = cumulative_pnl >= profit_target_required
        failed = cumulative_pnl <= ending_loss_limit or violating_daily_loss
        path.append(
            {
                "session_date": session_date,
                "pnl": pnl,
                "cum_pnl": cumulative_pnl,
                "best_day_profit": best_day_profit,
                "profit_target_required": profit_target_required,
                "loss_limit": ending_loss_limit,
            }
        )
        if passed or failed:
            resolved = True
            end_day = session_date
            break
        end_day = session_date
    profit_target_required = _effective_profit_target(
        topstep.base_profit_target,
        best_day_profit,
        topstep.consistency_threshold,
        topstep.consistency_buffer,
    )
    best_day_ratio = (best_day_profit / cumulative_pnl) if cumulative_pnl > 0 else None
    violated_consistency = bool(best_day_ratio is not None and best_day_ratio >= topstep.consistency_threshold)
    status = "passed" if passed else "failed" if failed else "active"
    return CombineAttemptResult(
        status=status,
        resolved=resolved,
        passed=passed,
        failed=failed,
        days_used=days_used,
        start_day=start_day,
        end_day=end_day,
        cumulative_pnl=float(cumulative_pnl),
        profit_target_required=float(profit_target_required),
        best_day_profit=float(best_day_profit),
        best_day_ratio=float(best_day_ratio) if best_day_ratio is not None else None,
        monthly_fees_paid=float(monthly_fees_paid),
        ending_loss_limit=float(ending_loss_limit),
        violated_consistency=violated_consistency,
        path=path,
    )


def simulate_rolling_attempts(daily: pd.DataFrame, topstep: TopstepConfig) -> list[CombineAttemptResult]:
    if daily.empty:
        return []
    return [simulate_attempt(daily, topstep, start_index=i) for i in range(len(daily))]


def attempts_to_frame(attempts: Iterable[CombineAttemptResult]) -> pd.DataFrame:
    return pd.DataFrame([attempt.to_dict() for attempt in attempts])
