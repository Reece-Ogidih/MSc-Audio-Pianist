#!/usr/bin/env python
"""Aggregate completed five-note factorial run evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITION_DIRS = ("droq_original", "droq_sensitive_v1", "sac_original", "sac_sensitive_v1")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_condition_runs(root: Path) -> dict[str, Path]:
    runs = {}
    for condition in CONDITION_DIRS:
        condition_root = root / condition
        candidates = []
        if (condition_root / "evaluation/per_checkpoint_summary.csv").exists():
            candidates.append(condition_root)
        candidates.extend(
            sorted(
                [path for path in condition_root.glob("*") if (path / "evaluation/per_checkpoint_summary.csv").exists()]
            )
        )
        if candidates:
            runs[condition] = candidates[-1]
    return runs


def condition_metadata(condition: str) -> tuple[str, str]:
    algorithm = "droq" if condition.startswith("droq") else "sac"
    reward = "transition_cleanup_sensitive_v1" if "sensitive" in condition else "transition_cleanup"
    return algorithm, reward


def aggregate(root: Path, out_dir: Path) -> dict[str, Any]:
    condition_runs = discover_condition_runs(root)
    all_summary = []
    all_sequence = []
    for condition, run_dir in condition_runs.items():
        algorithm, reward = condition_metadata(condition)
        for row in read_csv(run_dir / "evaluation/per_checkpoint_summary.csv"):
            row.update({"condition_id": condition, "algorithm": algorithm, "reward_profile": reward, "run_dir": str(run_dir)})
            all_summary.append(row)
        for row in read_csv(run_dir / "evaluation/per_checkpoint_per_sequence_metrics.csv"):
            row.update({"condition_id": condition, "algorithm": algorithm, "reward_profile": reward, "run_dir": str(run_dir)})
            all_sequence.append(row)
    write_csv(out_dir / "all_conditions_per_checkpoint.csv", all_summary)
    write_csv(out_dir / "all_conditions_per_sequence.csv", all_sequence)
    final_rows = [row for row in all_summary if int(float(row["checkpoint_step"])) == 1000000]
    write_csv(out_dir / "all_conditions_1m_summary.csv", final_rows)
    _write_effect_tables(out_dir, final_rows)
    pareto = pareto_candidates(all_summary)
    write_csv(out_dir / "pareto_candidates_by_checkpoint.csv", pareto)
    write_csv(out_dir / "final_pareto_candidates.csv", [row for row in pareto if int(float(row["checkpoint_step"])) == 1000000])
    learning = classify_learning_curves(all_summary)
    write_csv(out_dir / "learning_curve_classification.csv", learning)
    report = {
        "condition_runs": {key: str(value) for key, value in condition_runs.items()},
        "conditions_found": sorted(condition_runs),
        "selection_note": "Seed-13 descriptive results only; do not claim statistical algorithm superiority.",
        "pareto_count": len(pareto),
    }
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "selection_report.md").write_text(markdown_report(report, final_rows, learning), encoding="utf-8")
    return report


def _write_effect_tables(out_dir: Path, final_rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "algorithm_effect_original_reward.csv", [r for r in final_rows if r["reward_profile"] == "transition_cleanup"])
    write_csv(out_dir / "algorithm_effect_sensitive_reward.csv", [r for r in final_rows if r["reward_profile"] == "transition_cleanup_sensitive_v1"])
    write_csv(out_dir / "reward_effect_droq.csv", [r for r in final_rows if r["algorithm"] == "droq"])
    write_csv(out_dir / "reward_effect_sac.csv", [r for r in final_rows if r["algorithm"] == "sac"])


def pareto_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint_step"], row["sequence_group"])].append(row)
    selected = []
    for _, items in grouped.items():
        for row in items:
            if not any(_dominates(other, row) for other in items if other is not row):
                selected.append(row)
    return selected


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    high = ["mean_pressed_key_f1", "mean_timestep_f1"]
    low = ["mean_integrated_unintended_travel", "mean_wrong_key_crossings", "affected_episode_percentage"]
    better_or_equal = all(float(a[k]) >= float(b[k]) for k in high) and all(float(a[k]) <= float(b[k]) for k in low)
    strictly_better = any(float(a[k]) > float(b[k]) for k in high) or any(float(a[k]) < float(b[k]) for k in low)
    return better_or_equal and strictly_better


def classify_learning_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition_id"], row["sequence_group"])].append(row)
    output = []
    for (condition, group), items in grouped.items():
        items = sorted(items, key=lambda row: int(float(row["checkpoint_step"])))
        f1 = [float(row["mean_timestep_f1"]) for row in items]
        clean = [float(row["mean_integrated_unintended_travel"]) for row in items]
        if len(f1) < 2:
            label = "insufficient_data"
        elif f1[-1] < 0.1:
            label = "failed"
        elif f1[-1] + 0.05 < f1[0]:
            label = "regressing"
        elif f1[-1] > f1[0] + 0.05:
            label = "still_improving"
        elif np.std(f1) > 0.15 or np.std(clean) > 5.0:
            label = "unstable"
        else:
            label = "approximately_plateaued"
        output.append(
            {
                "condition_id": condition,
                "sequence_group": group,
                "classification": label,
                "first_timestep_f1": f1[0] if f1 else None,
                "last_timestep_f1": f1[-1] if f1 else None,
                "first_integrated_unintended": clean[0] if clean else None,
                "last_integrated_unintended": clean[-1] if clean else None,
            }
        )
    return output


def markdown_report(report: dict[str, Any], final_rows: list[dict[str, Any]], learning: list[dict[str, Any]]) -> str:
    lines = ["# Five-Note Factorial Selection Report", "", "Seed-13 findings are descriptive, not statistically conclusive.", ""]
    lines.append("## Conditions Found")
    for condition, path in sorted(report["condition_runs"].items()):
        lines.append(f"- `{condition}`: `{path}`")
    lines.extend(["", "## Final 1M Rows"])
    for row in final_rows:
        lines.append(
            f"- `{row['condition_id']}` `{row['sequence_group']}`: pressed F1={row['mean_pressed_key_f1']}, "
            f"timestep F1={row['mean_timestep_f1']}, integrated unintended={row['mean_integrated_unintended_travel']}"
        )
    lines.extend(["", "## Learning-Curve Labels"])
    for row in learning:
        lines.append(f"- `{row['condition_id']}` `{row['sequence_group']}`: {row['classification']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(aggregate(Path(args.root), out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
