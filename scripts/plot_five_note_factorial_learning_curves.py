#!/usr/bin/env python
"""Create compact SVG learning-curve plots from factorial aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def svg_plot(path: Path, title: str, series: dict[str, list[tuple[float, float]]], ylabel: str) -> None:
    width, height = 760, 460
    left, right, top, bottom = 70, 190, 50, 60
    points = [point for values in series.values() for point in values]
    if not points:
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymax += 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def sx(x):
        return left + (x - xmin) / max(1e-6, xmax - xmin) * (width - left - right)

    def sy(y):
        return top + (ymax - y) / max(1e-6, ymax - ymin) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
    ]
    for index, (name, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(values))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{coords}"/>')
        ly = top + 22 + index * 22
        lines.append(f'<text x="{width-right+18}" y="{ly}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    lines.extend(
        [
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">Environment timestep</text>',
            f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict], out: Path, group: str, metric: str, title: str) -> None:
    series = defaultdict(list)
    for row in rows:
        if row.get("sequence_group") == group:
            series[row["condition_id"]].append((float(row["checkpoint_step"]), float(row[metric])))
    svg_plot(out / f"{group}_{metric}.svg", title, dict(series), metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", required=True)
    args = parser.parse_args()
    aggregate = Path(args.aggregate_dir)
    rows = read_rows(aggregate / "all_conditions_per_checkpoint.csv")
    out = aggregate / "plots"
    for group in ["trained_single", "trained_transition", "heldout_transition", "composition_probe"]:
        plot_metric(rows, out, group, "mean_pressed_key_f1", f"{group} pressed-key F1")
        plot_metric(rows, out, group, "mean_timestep_f1", f"{group} timestep F1")
        plot_metric(rows, out, group, "mean_integrated_unintended_travel", f"{group} integrated unintended")
        plot_metric(rows, out, group, "mean_wrong_key_crossings", f"{group} wrong-key threshold crossings")
    print(f"plots_dir={out}")


if __name__ == "__main__":
    main()
