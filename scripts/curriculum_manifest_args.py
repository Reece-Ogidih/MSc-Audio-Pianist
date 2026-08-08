#!/usr/bin/env python3
"""Print shell-safe sequence arguments from a curriculum manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format", choices=["shell", "plain"], default="plain")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    sequences = payload["sequences"]
    sequence_pitches = ";".join(",".join(str(pitch) for pitch in item["pitches"]) for item in sequences)
    weights = ",".join(f"{float(item['sampling_weight']):.12g}" for item in sequences)
    horizon_steps = _horizon_steps(sequences, note_duration=float(payload["note_duration"]), note_gap=float(payload["note_gap"]))
    if args.format == "shell":
        print(f"SEQUENCE_PITCHES={shlex.quote(sequence_pitches)}")
        print(f"SEQUENCE_SAMPLING_WEIGHTS={shlex.quote(weights)}")
        print(f"HORIZON_STEPS={horizon_steps}")
    else:
        print(sequence_pitches)
        print(weights)
        print(horizon_steps)


def _horizon_steps(sequences, *, note_duration: float, note_gap: float) -> int:
    max_len = max(len(item["pitches"]) for item in sequences)
    last_offset = (max_len - 1) * (note_duration + note_gap) + note_duration
    return int(__import__("math").ceil((last_offset + 0.80) / 0.05))


if __name__ == "__main__":
    main()
