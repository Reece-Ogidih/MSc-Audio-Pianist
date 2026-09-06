#!/usr/bin/env python
"""Create an annotated motion-quality demo from an existing rollout video/trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
import pandas as pd

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "src"))

from ala_pianist.evaluation.motion_quality import annotate_rollout_frame, phase_labels  # noqa: E402


AUDIT = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/audit"
DEFAULT_SRC = AUDIT / "reference_rollouts/droq_sensitive_v1_800k_frozen/trained_73_72"
DEFAULT_OUT = AUDIT / "motion_quality/annotated_demo/trained_73_72"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = create_demo(args.source_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def create_demo(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(source_dir / "trace.csv")
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    phases = phase_labels(trace)
    failure_labels = json.loads(summary.get("failure_labels", "[]"))
    reader = imageio.get_reader(source_dir / "rollout.mp4")
    frames = []
    try:
        for index, frame in enumerate(reader):
            if index >= len(trace):
                break
            row = trace.iloc[index]
            intended_midi = _loads(row["intended_midi_pitches"])
            intended_keys = _loads(row["intended_key_indices"])
            pressed_midi = _loads(row["pressed_midi_pitches"])
            target_keys = {int(key) for key in intended_keys}
            wrong_pressed = [midi for midi in pressed_midi if int(midi) - 21 not in target_keys]
            frames.append(
                annotate_rollout_frame(
                    np.asarray(frame),
                    sequence_id=str(summary.get("sequence_id", source_dir.name)),
                    simulation_time=float(row["simulation_time"]),
                    intended_midi=intended_midi,
                    intended_keys=intended_keys,
                    pressed_midi=pressed_midi,
                    wrong_pressed_midi=wrong_pressed,
                    phase=str(phases.iloc[index]),
                    failure_labels=failure_labels,
                    target_activation=float(row["target_key_activation"]),
                    max_unintended=float(row["max_unintended_key_state"]),
                )
            )
    finally:
        reader.close()
    raw_path = output_dir / "rollout_annotated_raw.mp4"
    compat_path = output_dir / "rollout_annotated.mp4"
    fps = int(summary.get("fps", 20))
    imageio.mimsave(raw_path, frames, fps=fps, macro_block_size=1)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-vf",
            f"fps={fps},format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-vsync",
            "cfr",
            "-bf",
            "0",
            "-movflags",
            "+faststart",
            str(compat_path),
        ],
        check=True,
    )
    for filename in ["trace.csv", "diagnostic_plot.png", "summary.json"]:
        target = output_dir / filename
        target.write_bytes((source_dir / filename).read_bytes())
    probe = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,r_frame_rate,has_b_frames,nb_frames",
            "-of",
            "json",
            str(compat_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    (output_dir / "rollout_annotated_ffprobe.json").write_text(probe.stdout, encoding="utf-8")
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(compat_path), "-f", "null", "-"], check=True)
    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "annotated_video": str(compat_path),
        "frame_count": len(frames),
        "fps": fps,
    }


def _loads(value) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    return []


if __name__ == "__main__":
    main()
