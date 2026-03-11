# Topstep Research Framework

This repository contains a research and simulation framework built specifically for the Topstep $50K Trading Combine problem, not a generic "PnL backtester."

The core question is:

Can a config-driven intraday futures strategy reach the Topstep profit objective before breaching the loss boundary, while remaining robust to consistency, sequencing, and regime changes?

## What it does

- Loads intraday OHLCV futures data from CSV or Parquet
- Normalizes timestamps into exchange time
- Applies event-day and session filters
- Runs a rules-based intraday strategy engine for MES-first research
- Simulates Topstep-style pass/fail outcomes, including:
  - profit target
  - trailing or static loss boundary
  - best-day consistency concentration
  - optional subscription drag
- Benchmarks the strategy against random entry, naive breakout, and naive mean reversion baselines
- Runs Monte Carlo sequencing tests:
  - trade-order reshuffle
  - day bootstrap
  - week bootstrap
  - regime-conditioned bootstrap
- Saves versioned experiment outputs and markdown summaries

## Quickstart

1. Put 1-minute or other intraday OHLCV data somewhere on disk.
2. Update [`configs/mes_momentum.toml`](/Users/liamhind/Desktop/Prop Firm Backtest/configs/mes_momentum.toml) with your paths and settings.
3. Run:

```bash
PYTHONPATH=src python3 -m topstep_research.cli run --config configs/mes_momentum.toml
```

To generate synthetic data for a smoke test:

```bash
PYTHONPATH=src python3 -m topstep_research.cli generate-demo-data --output examples/mes_demo.csv
PYTHONPATH=src python3 -m topstep_research.cli run --config configs/demo_mes.toml
```

## Data schema

Expected columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional columns:

- `instrument`

See [`docs/data_format.md`](/Users/liamhind/Desktop/Prop Firm Backtest/docs/data_format.md).

## Topstep modeling notes

The simulator is built around the current 50K combine problem surface:

- base profit target of `$3,000`
- maximum loss buffer of `$2,000`
- 50% best-day consistency objective
- optional daily loss limit modeling

The implementation supports both:

- `trailing_balance` mode for current Topstep-style maximum loss behavior
- `static_from_start` mode if you want the simpler fixed `+3000 / -2000` barrier framing

## Repository layout

- [`src/topstep_research`](/Users/liamhind/Desktop/Prop Firm Backtest/src/topstep_research) core package
- [`configs`](/Users/liamhind/Desktop/Prop Firm Backtest/configs) experiment configs
- [`examples`](/Users/liamhind/Desktop/Prop Firm Backtest/examples) demo inputs
- [`tests`](/Users/liamhind/Desktop/Prop Firm Backtest/tests) smoke tests
