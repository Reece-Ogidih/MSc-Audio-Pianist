#!/usr/bin/env python
"""Select Pareto-leading five-note factorial candidates from aggregate metrics."""

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
    pareto = read_csv(aggregate / "final_pareto_candidates.csv")
    selected = sorted(
        pareto,
        key=lambda row: (
            float(row.get("mean_timestep_f1", 0.0)),
            float(row.get("mean_pressed_key_f1", 0.0)),
            -float(row.get("mean_integrated_unintended_travel", 0.0)),
        ),
        reverse=True,
    )[:2]
    payload = {
        "selection_method": "accuracy gate plus Pareto; shaped return intentionally excluded",
        "top_candidates_for_additional_seed": selected,
        "interpretation_note": "Seed-13 only; replicate finalists before algorithm-superiority claims.",
    }
    path = aggregate / "selection_candidates.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
