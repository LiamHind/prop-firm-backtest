from __future__ import annotations

from statistics import mean, median

import numpy as np
import pandas as pd

from topstep_research.config import MonteCarloConfig, TopstepConfig
from topstep_research.domain import CombineAttemptResult, MonteCarloSummary
from topstep_research.simulator import simulate_attempt


def _coerce_daily_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=["session_date", "pnl", "trade_count", "regime"])


def _trade_shuffle_daily(trades: pd.DataFrame, daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if trades.empty or daily.empty:
        return daily.copy()
    shuffled = trades.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    trade_counts = daily["trade_count"].tolist()
    regimes = daily["regime"].tolist()
    records: list[dict[str, object]] = []
    cursor = 0
    for idx, trade_count in enumerate(trade_counts):
        if cursor >= len(shuffled):
            break
        take = min(int(trade_count), len(shuffled) - cursor)
        slice_df = shuffled.iloc[cursor : cursor + take]
        records.append(
            {
                "session_date": f"shuffle_{idx:04d}",
                "pnl": float(slice_df["pnl"].sum()),
                "trade_count": int(take),
                "regime": str(regimes[idx]),
            }
        )
        cursor += take
    return _coerce_daily_frame(records)


def _day_bootstrap(daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    picks = rng.integers(0, len(daily), size=len(daily))
    sampled = daily.iloc[picks].reset_index(drop=True).copy()
    sampled["session_date"] = [f"day_{i:04d}" for i in range(len(sampled))]
    return sampled


def _week_bootstrap(daily: pd.DataFrame, rng: np.random.Generator, block_size: int) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    records: list[pd.DataFrame] = []
    while sum(len(block) for block in records) < len(daily):
        start = int(rng.integers(0, max(len(daily) - block_size + 1, 1)))
        block = daily.iloc[start : start + block_size].copy()
        records.append(block)
    sampled = pd.concat(records, ignore_index=True).iloc[: len(daily)].copy()
    sampled["session_date"] = [f"week_{i:04d}" for i in range(len(sampled))]
    return sampled


def _regime_bootstrap(daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if daily.empty or "regime" not in daily.columns:
        return daily.copy()
    by_regime = {name: frame.reset_index(drop=True) for name, frame in daily.groupby("regime")}
    rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(daily.iterrows()):
        regime = row["regime"]
        source = by_regime.get(regime)
        if source is None or source.empty:
            source_row = row
        else:
            source_row = source.iloc[int(rng.integers(0, len(source)))]
        rows.append(
            {
                "session_date": f"regime_{idx:04d}",
                "pnl": float(source_row["pnl"]),
                "trade_count": int(source_row["trade_count"]),
                "regime": str(regime),
            }
        )
    return _coerce_daily_frame(rows)


def _summarize(method: str, attempts: list[CombineAttemptResult], profit_target: float) -> MonteCarloSummary:
    passed = [a for a in attempts if a.passed]
    failed = [a for a in attempts if a.failed]
    unresolved = [a for a in attempts if not a.resolved]
    pass_days = [a.days_used for a in passed]
    fail_days = [a.days_used for a in failed]
    return MonteCarloSummary(
        method=method,
        iterations=len(attempts),
        pass_probability=float(len(passed) / len(attempts)) if attempts else 0.0,
        fail_probability=float(len(failed) / len(attempts)) if attempts else 0.0,
        unresolved_probability=float(len(unresolved) / len(attempts)) if attempts else 0.0,
        avg_days_to_pass=float(mean(pass_days)) if pass_days else None,
        median_days_to_pass=float(median(pass_days)) if pass_days else None,
        avg_days_to_fail=float(mean(fail_days)) if fail_days else None,
        median_days_to_fail=float(median(fail_days)) if fail_days else None,
        largest_day_gt_half_target_frequency=float(
            mean([(a.best_day_profit >= (profit_target * 0.5)) for a in attempts])
        )
        if attempts
        else 0.0,
    )


def run_monte_carlo(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    topstep: TopstepConfig,
    monte_carlo: MonteCarloConfig,
    seed: int,
) -> list[MonteCarloSummary]:
    if not monte_carlo.enabled or daily.empty:
        return []
    rng = np.random.default_rng(seed)
    outputs: list[MonteCarloSummary] = []
    methods: list[tuple[str, callable]] = []
    if monte_carlo.trade_shuffle:
        methods.append(("trade_shuffle", lambda: _trade_shuffle_daily(trades, daily, rng)))
    if monte_carlo.day_bootstrap:
        methods.append(("day_bootstrap", lambda: _day_bootstrap(daily, rng)))
    if monte_carlo.week_bootstrap:
        methods.append(("week_bootstrap", lambda: _week_bootstrap(daily, rng, monte_carlo.block_size_days)))
    if monte_carlo.regime_bootstrap:
        methods.append(("regime_bootstrap", lambda: _regime_bootstrap(daily, rng)))
    for method_name, sampler in methods:
        attempts = []
        for _ in range(monte_carlo.iterations):
            sampled_daily = sampler()
            attempts.append(simulate_attempt(sampled_daily, topstep, start_index=0))
        outputs.append(_summarize(method_name, attempts, topstep.base_profit_target))
    return outputs
