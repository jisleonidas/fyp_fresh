from __future__ import annotations

import argparse

from buggy_ai.config import load_config
from buggy_ai.runtime.pipeline import AutonomyPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo self-driving buggy stack")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = AutonomyPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
