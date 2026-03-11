from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TradeRecord:
    strategy_name: str
    config_id: str
    day_id: str
    entry_timestamp: str
    exit_timestamp: str
    direction: int
    entry_price: float
    exit_price: float
    stop_distance: float
    target_distance: float
    pnl: float
    mae: float
    mfe: float
    reason_for_exit: str
    contracts: int
    point_value: float
    trade_number: int
    regime: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    strategy_name: str
    config_id: str
    trades: pd.DataFrame
    daily: pd.DataFrame
    trade_metrics: dict[str, Any]


@dataclass
class CombineAttemptResult:
    status: str
    resolved: bool
    passed: bool
    failed: bool
    days_used: int
    start_day: str | None
    end_day: str | None
    cumulative_pnl: float
    profit_target_required: float
    best_day_profit: float
    best_day_ratio: float | None
    monthly_fees_paid: float
    ending_loss_limit: float
    violated_consistency: bool
    path: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonteCarloSummary:
    method: str
    iterations: int
    pass_probability: float
    fail_probability: float
    unresolved_probability: float
    avg_days_to_pass: float | None
    median_days_to_pass: float | None
    avg_days_to_fail: float | None
    median_days_to_fail: float | None
    largest_day_gt_half_target_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
