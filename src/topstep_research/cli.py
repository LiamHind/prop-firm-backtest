from __future__ import annotations

import argparse
from pathlib import Path

from topstep_research.config import load_configs
from topstep_research.runner import run_experiment
from topstep_research.sample_data import generate_demo_intraday_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topstep combine research framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more experiment configs")
    run_parser.add_argument("--config", required=True, help="Path to TOML config")

    demo_parser = subparsers.add_parser("generate-demo-data", help="Generate synthetic MES intraday data")
    demo_parser.add_argument("--output", required=True, help="Where to write the CSV")
    demo_parser.add_argument("--sessions", type=int, default=80, help="Number of trading sessions")
    demo_parser.add_argument("--bars-per-session", type=int, default=180, help="Bars per session")
    demo_parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "generate-demo-data":
        path = generate_demo_intraday_data(
            output_path=args.output,
            sessions=args.sessions,
            bars_per_session=args.bars_per_session,
            seed=args.seed,
        )
        print(path)
        return
    if args.command == "run":
        for config in load_configs(args.config):
            output_dir = run_experiment(config)
            print(output_dir)


if __name__ == "__main__":
    main()
