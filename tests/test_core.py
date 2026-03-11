from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from topstep_research.config import load_configs
from topstep_research.monte_carlo import run_monte_carlo
from topstep_research.sample_data import generate_demo_intraday_data
from topstep_research.simulator import simulate_attempt, simulate_rolling_attempts


class ConfigTests(unittest.TestCase):
    def test_sweep_expansion(self) -> None:
        config_text = """
[experiment]
name = "test"

[data]
path = "examples/mes_demo.csv"

[sweep]
"strategy.stop_atr" = [1.0, 1.2]
"strategy.target_r" = [1.5, 2.0]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.toml"
            path.write_text(config_text, encoding="utf-8")
            configs = load_configs(path)
        self.assertEqual(len(configs), 4)


class SimulatorTests(unittest.TestCase):
    def test_static_attempt_can_pass(self) -> None:
        daily = pd.DataFrame(
            [
                {"session_date": "2024-01-02", "pnl": 500.0, "trade_count": 1, "regime": "highvol_up"},
                {"session_date": "2024-01-03", "pnl": 700.0, "trade_count": 1, "regime": "highvol_up"},
                {"session_date": "2024-01-04", "pnl": 2000.0, "trade_count": 1, "regime": "highvol_up"},
            ]
        )
        from topstep_research.config import TopstepConfig

        result = simulate_attempt(
            daily,
            TopstepConfig(loss_limit_mode="static_from_start", consistency_threshold=1.0, consistency_buffer=0.0),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "passed")

    def test_rolling_attempt_count_matches_days(self) -> None:
        daily = pd.DataFrame(
            [
                {"session_date": "2024-01-02", "pnl": 100.0, "trade_count": 1, "regime": "highvol_up"},
                {"session_date": "2024-01-03", "pnl": -50.0, "trade_count": 1, "regime": "lowvol_down"},
                {"session_date": "2024-01-04", "pnl": 150.0, "trade_count": 1, "regime": "highvol_up"},
            ]
        )
        from topstep_research.config import TopstepConfig

        results = simulate_rolling_attempts(daily, TopstepConfig())
        self.assertEqual(len(results), 3)


class MonteCarloTests(unittest.TestCase):
    def test_day_bootstrap_runs(self) -> None:
        from topstep_research.config import MonteCarloConfig, TopstepConfig

        trades = pd.DataFrame(
            [
                {"pnl": 100.0, "day_id": "2024-01-02"},
                {"pnl": -50.0, "day_id": "2024-01-03"},
                {"pnl": 75.0, "day_id": "2024-01-04"},
            ]
        )
        daily = pd.DataFrame(
            [
                {"session_date": "2024-01-02", "pnl": 100.0, "trade_count": 1, "regime": "highvol_up"},
                {"session_date": "2024-01-03", "pnl": -50.0, "trade_count": 1, "regime": "lowvol_down"},
                {"session_date": "2024-01-04", "pnl": 75.0, "trade_count": 1, "regime": "highvol_up"},
            ]
        )
        results = run_monte_carlo(
            trades=trades,
            daily=daily,
            topstep=TopstepConfig(loss_limit_mode="static_from_start"),
            monte_carlo=MonteCarloConfig(iterations=5, trade_shuffle=False, week_bootstrap=False, regime_bootstrap=False),
            seed=7,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].method, "day_bootstrap")


class DemoDataTests(unittest.TestCase):
    def test_demo_generator_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.csv"
            generated = generate_demo_intraday_data(path, sessions=3, bars_per_session=10, seed=7)
            frame = pd.read_csv(generated)
        self.assertFalse(frame.empty)
        self.assertIn("timestamp", frame.columns)


if __name__ == "__main__":
    unittest.main()
