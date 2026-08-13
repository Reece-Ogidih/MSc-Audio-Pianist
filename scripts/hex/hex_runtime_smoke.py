#!/usr/bin/env python3
"""End-to-end runtime smoke used by the Hex container launcher."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cuda", action="store_true", help="Local validation only; never used by Hex.")
    args = parser.parse_args()

    numba_cache = _require_writable_directory("NUMBA_CACHE_DIR")
    xdg_cache = _require_writable_directory("XDG_CACHE_HOME")

    import basic_pitch
    import dm_control
    import mujoco
    import onnxruntime
    import robopianist
    import torch
    from basic_pitch import inference as basic_pitch_inference

    from ala_pianist.audio import BasicPitchTranscriber
    from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi
    from ala_pianist.pipelines.indirect import render_midi_with_fluidsynth
    from ala_pianist.rl import GeneralOneHandGoalEnv

    print("python_executable", __import__("sys").executable)
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("torch_cuda_available", torch.cuda.is_available())
    print("basic_pitch", getattr(basic_pitch, "__version__", "0.4.0"), basic_pitch.__file__)
    print("onnxruntime", onnxruntime.__version__)
    print("mujoco", mujoco.__version__)
    print("dm_control", dm_control.__file__)
    print("robopianist", robopianist.__file__)
    print("numba_cache_dir", numba_cache)
    print("xdg_cache_home", xdg_cache)

    if not str(getattr(basic_pitch, "__version__", "0.4.0")).startswith("0.4.0"):
        raise SystemExit("Expected Basic Pitch 0.4.0")
    if onnxruntime.__version__ != "1.23.2":
        raise SystemExit(f"Expected ONNX Runtime 1.23.2, got {onnxruntime.__version__}")
    basic_pitch_model = Path(basic_pitch_inference.ICASSP_2022_MODEL_PATH).with_suffix(".onnx")
    print("basic_pitch_model", basic_pitch_model)
    if not basic_pitch_model.is_file():
        raise SystemExit(f"Basic Pitch ONNX model missing: {basic_pitch_model}")

    if args.skip_cuda:
        print("cuda_operation", "skipped_for_local_validation")
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available inside the Hare container")
        value = torch.ones((8, 8), device="cuda")
        print("cuda_sum", float((value @ value).sum().detach().cpu()))

    evaluator_path = Path("/app/scripts/evaluate_long_horizon_compositional.py")
    if not evaluator_path.is_file():
        evaluator_path = Path(__file__).resolve().parents[1] / "evaluate_long_horizon_compositional.py"
    spec = importlib.util.spec_from_file_location("hex_smoke_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load project evaluator: {evaluator_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    print("project_evaluator", evaluator_path)

    soundfont = Path(robopianist.SF2_PATH)
    print("soundfont", soundfont, soundfont.exists())
    if not soundfont.is_file():
        raise SystemExit(f"Soundfont not found: {soundfont}")

    smoke_dir = Path("/tmp/ala-hex-runtime-smoke")
    midi_path = write_monophonic_midi([NoteEvent(74, 0.0, 0.28, 90)], smoke_dir / "note.mid")
    wav_path = render_midi_with_fluidsynth(midi_path, smoke_dir / "note.wav", soundfont_path=soundfont)
    transcription = BasicPitchTranscriber().transcribe(wav_path)
    print("basic_pitch_note_count", len(transcription.notes))
    if not transcription.notes:
        raise SystemExit("Basic Pitch smoke produced no notes")

    env = GeneralOneHandGoalEnv(
        generated_midi_dir=smoke_dir / "midi",
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=3,
        action_mode="direct",
        action_repeat=1,
    )
    observation, _info = env.reset(seed=1)
    for _ in range(3):
        observation, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    marker_name = os.environ.get("SMOKE_MARKER")
    if marker_name:
        marker = Path("/workspace") / marker_name
        marker.write_text("hex smoke ok\n", encoding="utf-8")
        print("smoke_marker", marker)
    print("HEX_RUNTIME_SMOKE_COMPLETE=true")


def _require_writable_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"{variable} is not set")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".ala-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(f"{variable} is not writable: {path}: {exc}") from exc
    return path


if __name__ == "__main__":
    main()
