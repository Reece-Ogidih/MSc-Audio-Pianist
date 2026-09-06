#!/usr/bin/env python3
"""Aggregate real-audio distribution-shift evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring optional tabulate."""

    if frame.empty:
        return "_No rows._\n"
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([_format_cell(row[column]) for column in frame.columns])
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/real_audio_distribution_shift/evaluation_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/real_audio_distribution_shift/analysis_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.evaluation_root.rglob("long_horizon_sequence_metrics.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frame["condition_group"] = path.parent.name
        rows.append(frame)
    if not rows:
        raise SystemExit(f"No evaluation CSVs found under {args.evaluation_root}")
    master = pd.concat(rows, ignore_index=True)
    master.to_csv(args.output_dir / "sequence_level_results.csv", index=False)
    group_cols = [col for col in ["pipeline", "model_label", "condition_group", "audio_mode"] if col in master.columns]
    summary = master.groupby(group_cols).agg(
        sequence_count=("sequence_name", "count"),
        pressed_key_f1=("pressed_key_f1", "mean"),
        timestep_f1=("timestep_f1", "mean"),
        event_hit=("target_event_hit_rate", "mean"),
        ordered_event=("ordered_event_accuracy", "mean"),
        max_unintended=("max_unintended_key_state", "mean"),
        integrated_unintended=("integrated_unintended_key_state", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "headline_metrics.csv", index=False)
    (args.output_dir / "real_audio_evidence.md").write_text(
        "# Real-Audio Distribution-Shift Evidence\n\n"
        "This report was generated from existing evaluation CSVs. Interpret paired synthetic-real deltas "
        "only after confirming that `condition_group` labels correspond to matched target sequences.\n\n"
        + _markdown_table(summary),
        encoding="utf-8",
    )
    print(f"output_dir={args.output_dir}")
    print("REAL_AUDIO_ANALYSIS_COMPLETE=true")


if __name__ == "__main__":
    main()
