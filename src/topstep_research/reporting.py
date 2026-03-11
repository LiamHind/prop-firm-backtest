from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from topstep_research.config import ExperimentConfig
from topstep_research.domain import BacktestResult, CombineAttemptResult, MonteCarloSummary


def create_run_directory(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{config.experiment.name}_{config.experiment.variant}_{config.config_id}_{timestamp}"
    output_dir = Path(config.experiment.output_dir) / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "trades").mkdir(exist_ok=True)
    (output_dir / "daily").mkdir(exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)
    return output_dir


def _safe_value(value: Any) -> Any:
    if isinstance(value, float) and (pd.isna(value) or not math.isfinite(value)):
        return None
    return value


def save_manifest(output_dir: Path, config: ExperimentConfig) -> None:
    payload = config.to_dict()
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_strategy_frames(
    output_dir: Path,
    segment: str,
    result: BacktestResult,
    attempts: list[CombineAttemptResult],
    monte_carlo: list[MonteCarloSummary],
) -> None:
    trade_slug = f"{segment}_{result.strategy_name}.csv"
    result.trades.to_csv(output_dir / "trades" / trade_slug, index=False)
    result.daily.to_csv(output_dir / "daily" / trade_slug, index=False)
    pd.DataFrame([attempt.to_dict() for attempt in attempts]).to_csv(
        output_dir / "tables" / f"{segment}_{result.strategy_name}_attempts.csv", index=False
    )
    pd.DataFrame([summary.to_dict() for summary in monte_carlo]).to_csv(
        output_dir / "tables" / f"{segment}_{result.strategy_name}_monte_carlo.csv", index=False
    )


def save_summary_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "tables" / "summary.csv", index=False)
    with (output_dir / "tables" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump([{k: _safe_value(v) for k, v in row.items()} for row in rows], handle, indent=2, default=str)


def build_markdown_report(
    output_dir: Path,
    config: ExperimentConfig,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {config.experiment.name}",
        "",
        f"- Variant: `{config.experiment.variant}`",
        f"- Config ID: `{config.config_id}`",
        f"- Strategy family: `{config.strategy.family}`",
        f"- Profit target: `${config.topstep.base_profit_target:,.0f}`",
        f"- Max loss buffer: `${config.topstep.max_loss_buffer:,.0f}`",
        f"- Loss mode: `{config.topstep.loss_limit_mode}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"### {row['segment']} / {row['strategy_name']}",
                "",
                f"- Pass probability: `{row['pass_probability']:.3f}`",
                f"- Fail probability: `{row['fail_probability']:.3f}`",
                f"- Unresolved probability: `{row['unresolved_probability']:.3f}`",
                f"- Avg days to pass: `{row['average_days_to_pass']}`",
                f"- Avg days to fail: `{row['average_days_to_fail']}`",
                f"- Largest-day >50% target frequency: `{row['largest_day_gt_half_target_frequency']:.3f}`",
                f"- Expectancy/trade: `{row['expectancy_per_trade']:.2f}`",
                f"- Profit factor: `{row['profit_factor']}`",
                f"- Total trades: `{row['total_trades']}`",
                "",
            ]
        )
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def save_optional_charts(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        note = "matplotlib is not installed. Install the 'charts' extra to generate PNG charts.\n"
        (output_dir / "charts" / "README.txt").write_text(note, encoding="utf-8")
        return
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return
    chart_df = summary[["strategy_name", "segment", "pass_probability", "fail_probability"]].copy()
    chart_df["label"] = chart_df["segment"] + " / " + chart_df["strategy_name"]
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_df.plot(
        x="label",
        y=["pass_probability", "fail_probability"],
        kind="bar",
        ax=ax,
        rot=30,
    )
    ax.set_title("Pass vs Fail Probability")
    ax.set_ylabel("Probability")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "charts" / "pass_fail_probabilities.png", dpi=150)
    plt.close(fig)
