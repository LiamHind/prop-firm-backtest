from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from topstep_research.config import ExperimentConfig
from topstep_research.data import load_market_data
from topstep_research.engine import run_backtest
from topstep_research.metrics import regime_summary, summarize_attempts
from topstep_research.monte_carlo import run_monte_carlo
from topstep_research.reporting import (
    build_markdown_report,
    create_run_directory,
    save_manifest,
    save_optional_charts,
    save_strategy_frames,
    save_summary_table,
)
from topstep_research.simulator import simulate_rolling_attempts


def _subset_data(data: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    filtered = data
    if start is not None:
        filtered = filtered[filtered["session_date"] >= start]
    if end is not None:
        filtered = filtered[filtered["session_date"] <= end]
    return filtered.reset_index(drop=True)


def _segment_map(data: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    if not config.split.enabled:
        return {"full_sample": data}
    segments = {
        "in_sample": _subset_data(data, config.split.in_sample_start, config.split.in_sample_end),
        "out_of_sample": _subset_data(data, config.split.out_of_sample_start, config.split.out_of_sample_end),
    }
    return {name: frame for name, frame in segments.items() if not frame.empty}


def _enabled_strategies(config: ExperimentConfig) -> list[str]:
    names = [config.strategy.family]
    for benchmark_name, params in config.benchmarks.items():
        if benchmark_name == config.strategy.family:
            continue
        if params.get("enabled", False):
            names.append(benchmark_name)
    return names


def run_experiment(config: ExperimentConfig) -> Path:
    data = load_market_data(config.data)
    output_dir = create_run_directory(config)
    save_manifest(output_dir, config)
    strategies = _enabled_strategies(config)
    summary_rows: list[dict[str, Any]] = []
    for segment_name, segment_data in _segment_map(data, config).items():
        for offset, strategy_name in enumerate(strategies):
            result = run_backtest(segment_data, strategy_name=strategy_name, config=config, seed_offset=offset)
            attempts = simulate_rolling_attempts(result.daily, config.topstep)
            attempt_summary = summarize_attempts(
                attempts,
                profit_target=config.topstep.base_profit_target,
                assumed_pass_value=float(config.metadata.get("assumed_pass_value", 0.0)),
            )
            monte_carlo = run_monte_carlo(
                trades=result.trades,
                daily=result.daily,
                topstep=config.topstep,
                monte_carlo=config.monte_carlo,
                seed=config.experiment.random_seed + offset,
            )
            regime_table = regime_summary(result.daily)
            save_strategy_frames(output_dir, segment_name, result, attempts, monte_carlo)
            regime_table.to_csv(
                output_dir / "tables" / f"{segment_name}_{strategy_name}_regimes.csv",
                index=False,
            )
            summary_rows.append(
                {
                    "segment": segment_name,
                    "strategy_name": strategy_name,
                    "config_id": config.config_id,
                    **result.trade_metrics,
                    **attempt_summary,
                }
            )
    save_summary_table(output_dir, summary_rows)
    build_markdown_report(output_dir, config, summary_rows)
    if config.reporting.make_charts:
        save_optional_charts(output_dir, summary_rows)
    return output_dir
