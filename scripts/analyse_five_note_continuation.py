#!/usr/bin/env python
"""Summarise whether five-note factorial runs merit continuation beyond 1M."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", required=True)
    args = parser.parse_args()
    aggregate = Path(args.aggregate_dir)
    learning = read_csv(aggregate / "learning_curve_classification.csv")
    recommendations = []
    for row in learning:
        if row["classification"] == "still_improving":
            recommendations.append(
                {
                    "condition_id": row["condition_id"],
                    "sequence_group": row["sequence_group"],
                    "recommendation": "consider_continuation",
                    "reason": "timestep F1 still improved over observed checkpoints",
                }
            )
    payload = {
        "analysis_regions": ["100k-300k", "300k-500k", "500k-700k", "700k-1M"],
        "recommendations": recommendations,
        "caution": "Do not infer stopped exploration without exploration metrics.",
    }
    path = aggregate / "continuation_analysis.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
