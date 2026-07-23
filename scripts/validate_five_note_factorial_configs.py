#!/usr/bin/env python
"""Validate five-note factorial configuration invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ala_pianist.experiments import validate_factorial_configs


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_CONFIG_DIR = ROOT / "configs" / "five_note_factorial_1m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    validate_factorial_configs(config_dir)
    manifest = json.loads((config_dir / "factorial_manifest_seed13.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "ok",
                "config_dir": str(config_dir),
                "conditions": [condition["condition_id"] for condition in manifest["conditions"]],
                "required_code_commit": manifest["required_code_commit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
