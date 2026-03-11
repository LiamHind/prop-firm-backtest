from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import product
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any


def _default_column_map() -> dict[str, str]:
    return {
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "instrument": "instrument",
    }


@dataclass
class ExperimentSection:
    name: str = "mes_topstep_research"
    output_dir: str = "outputs"
    random_seed: int = 7
    variant: str = "base"


@dataclass
class DataConfig:
    path: str = "examples/mes_demo.csv"
    format: str = "csv"
    instrument: str = "MES"
    source_timezone: str = "UTC"
    exchange_timezone: str = "America/New_York"
    timestamp_column: str = "timestamp"
    session_start: str | None = None
    session_end: str | None = None
    column_map: dict[str, str] = field(default_factory=_default_column_map)
    event_calendar_path: str | None = None
    exclude_event_tags: list[str] = field(default_factory=list)
    exclude_event_dates: list[str] = field(default_factory=list)


@dataclass
class SplitConfig:
    enabled: bool = False
    in_sample_start: str | None = None
    in_sample_end: str | None = None
    out_of_sample_start: str | None = None
    out_of_sample_end: str | None = None


@dataclass
class StrategyConfig:
    family: str = "momentum_pullback"
    session_start: str = "09:30"
    session_end: str = "11:00"
    contracts: int = 1
    point_value: float = 5.0
    commission_per_contract: float = 0.62
    slippage_points_per_side: float = 0.125
    higher_timeframe_minutes: int = 30
    higher_timeframe_lookback: int = 4
    vwap_slope_lookback: int = 5
    pullback_lookback_bars: int = 3
    pullback_touch_tolerance_atr: float = 0.35
    confirmation_lookback_bars: int = 1
    atr_period: int = 14
    stop_atr: float = 1.0
    target_r: float = 2.0
    trailing: bool = False
    trailing_r: float = 1.0
    max_trades_per_day: int = 2
    daily_stop: float = -300.0
    daily_profit_cap: float = 500.0
    intrabar_fill_priority: str = "stop_first"
    benchmark_random_seed: int = 11
    no_trade_dates: list[str] = field(default_factory=list)


@dataclass
class TopstepConfig:
    account_label: str = "50K"
    base_profit_target: float = 3000.0
    max_loss_buffer: float = 2000.0
    max_position_size: int = 5
    loss_limit_mode: str = "trailing_balance"
    consistency_threshold: float = 0.50
    consistency_buffer: float = 1.0
    model_daily_loss_limit: bool = False
    daily_loss_limit: float | None = None
    monthly_subscription_fee: float = 49.0
    trading_days_per_month: int = 21


@dataclass
class MonteCarloConfig:
    enabled: bool = True
    iterations: int = 250
    trade_shuffle: bool = True
    day_bootstrap: bool = True
    week_bootstrap: bool = True
    regime_bootstrap: bool = True
    block_size_days: int = 5


@dataclass
class ReportingConfig:
    save_trade_csv: bool = True
    save_daily_csv: bool = True
    save_markdown: bool = True
    save_json: bool = True
    make_charts: bool = True


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    topstep: TopstepConfig = field(default_factory=TopstepConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    benchmarks: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "random_entry": {"enabled": True},
            "breakout": {"enabled": True, "lookback_bars": 20},
            "mean_reversion": {"enabled": True, "distance_from_vwap_atr": 1.0},
        }
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _dataclass_from_mapping(cls: type[Any], payload: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_name, field_info in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name not in payload:
            continue
        value = payload[field_name]
        default = getattr(cls, field_name, None)
        if is_dataclass(default):
            kwargs[field_name] = _dataclass_from_mapping(type(default), value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def _set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _slugify(value: Any) -> str:
    text = str(value).replace(" ", "_").replace("/", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-", "."})


def _expand_raw_configs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = raw.pop("sweep", {})
    if not sweep:
        return [raw]
    keys = list(sweep.keys())
    values = [list(v) for v in sweep.values()]
    expanded: list[dict[str, Any]] = []
    for combo in product(*values):
        variant = deepcopy(raw)
        applied = {}
        slug_parts = []
        for dotted_key, value in zip(keys, combo):
            _set_nested_value(variant, dotted_key, value)
            applied[dotted_key] = value
            slug_parts.append(f"{dotted_key.split('.')[-1]}-{_slugify(value)}")
        experiment = variant.setdefault("experiment", {})
        base_variant = experiment.get("variant", "base")
        experiment["variant"] = "_".join([base_variant, *slug_parts])
        metadata = variant.setdefault("metadata", {})
        metadata["applied_sweep"] = applied
        expanded.append(variant)
    return expanded


def load_configs(config_path: str | Path) -> list[ExperimentConfig]:
    path = Path(config_path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    configs = []
    config_root = path.parent.resolve()
    cwd_root = Path.cwd()
    for payload in _expand_raw_configs(raw):
        experiment_payload = payload.get("experiment", {})
        data_payload = payload.get("data", {})
        output_dir = experiment_payload.get("output_dir", "outputs")
        data_path = data_payload.get("path", DataConfig.path)
        event_path = data_payload.get("event_calendar_path")
        experiment_payload["output_dir"] = str(resolve_path(cwd_root, output_dir))
        data_payload["path"] = str(resolve_existing_path(config_root, cwd_root, data_path))
        if event_path is not None:
            data_payload["event_calendar_path"] = str(resolve_existing_path(config_root, cwd_root, event_path))
        config = ExperimentConfig(
            experiment=ExperimentSection(**experiment_payload),
            data=DataConfig(**data_payload),
            split=SplitConfig(**payload.get("split", {})),
            strategy=StrategyConfig(**payload.get("strategy", {})),
            topstep=TopstepConfig(**payload.get("topstep", {})),
            monte_carlo=MonteCarloConfig(**payload.get("monte_carlo", {})),
            reporting=ReportingConfig(**payload.get("reporting", {})),
            benchmarks=payload.get("benchmarks", {}),
            metadata=payload.get("metadata", {}),
        )
        configs.append(config)
    return configs


def resolve_path(root: str | Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path


def resolve_existing_path(primary_root: str | Path, fallback_root: str | Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    primary = Path(primary_root) / path
    if primary.exists():
        return primary
    return Path(fallback_root) / path
