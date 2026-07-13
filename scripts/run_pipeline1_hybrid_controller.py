import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ala_pianist.controllers import HybridPipeline1Controller
from ala_pianist.controllers.action_library import KEYSET_MIDI
from ala_pianist.pipelines import pipeline1_events_from_pitches, run_pipeline1


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "pipeline1_hybrid"
DEFAULT_MODEL = ROOT / "experiments" / "residual_single_note" / "residual_sac_cleanliness_scale_0.1"
DEFAULT_DSHARP5_MODEL = DEFAULT_MODEL


def _print_result(label, result) -> None:
    print(f"{label}:")
    print(f"  transcription={result.transcription}")
    print(f"  target_recall={result.target_recall:.6f}")
    print(f"  wrong_key_rate={result.wrong_key_rate:.6f}")
    print(f"  max_unintended_key_state={result.max_unintended_key_state:.6f}")
    print(f"  clean_hit_count={result.clean_hit_count}")
    print(f"  dirty_hit_count={result.dirty_hit_count}")
    print(f"  miss_count={result.miss_count}")
    for metric in result.note_metrics:
        print(
            f"  midi={metric.midi_pitch} outcome={metric.outcome} "
            f"strict={metric.strict_outcome} "
            f"quality={metric.trajectory_quality} "
            f"target={metric.max_target_key_state:.6f} "
            f"unintended={metric.max_unintended_key_state:.6f} "
            f"nearby={metric.nearby_key_states} "
            f"pressed={list(metric.pressed_keys)} "
            f"closest_finger={metric.closest_finger_to_target} "
            f"contact_finger={metric.target_contact_finger} "
            f"distance={metric.finger_target_distance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-model-path", type=Path, default=DEFAULT_DSHARP5_MODEL)
    parser.add_argument("--d5-residual-model-path", type=Path, default=None)
    parser.add_argument("--csharp5-residual-model-path", type=Path, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    residual_paths = {}
    if args.residual_model_path is not None and args.residual_model_path.exists():
        residual_paths[75] = args.residual_model_path
    if args.d5_residual_model_path is not None and args.d5_residual_model_path.exists():
        residual_paths[74] = args.d5_residual_model_path
    if args.csharp5_residual_model_path is not None and args.csharp5_residual_model_path.exists():
        residual_paths[73] = args.csharp5_residual_model_path

    if not residual_paths:
        v0 = run_pipeline1(
            audio_path=OUT_DIR / "v0_audio.wav",
            library_path=OUT_DIR / "keyset_action_library.json",
            rollout_midi_path=OUT_DIR / "v0_rollout.mid",
            summary_path=OUT_DIR / "v0_summary.json",
            build_library=True,
        )
        print(
            "residual_models_missing="
            f"D#5:{args.residual_model_path}, D5:{args.d5_residual_model_path}, "
            f"C#5:{args.csharp5_residual_model_path}; ran v0 only"
        )
        _print_result("v0_action_library", v0)
        return

    controller = HybridPipeline1Controller(residual_model_paths=residual_paths)
    sequence_specs = {
        "default": None,
        "known_anchors": [74, 75, 74, 75],
        "mixed_phrase": [69, 74, 75, 71],
    }
    combined = {
        "keyset": list(KEYSET_MIDI),
        "residual_model_paths": {str(key): str(value) for key, value in residual_paths.items()},
        "sequences": {},
    }
    printed_header = False
    for name, pitches in sequence_specs.items():
        events = None if pitches is None else pipeline1_events_from_pitches(pitches)
        v0 = run_pipeline1(
            audio_path=OUT_DIR / f"{name}_v0_audio.wav",
            library_path=OUT_DIR / "keyset_action_library.json",
            rollout_midi_path=OUT_DIR / f"{name}_v0_rollout.mid",
            summary_path=OUT_DIR / f"{name}_v0_summary.json",
            note_events=events,
            build_library=not printed_header,
        )
        printed_header = True
        v1 = run_pipeline1(
            audio_path=OUT_DIR / f"{name}_v1_audio.wav",
            library_path=OUT_DIR / "keyset_action_library.json",
            rollout_midi_path=OUT_DIR / f"{name}_v1_rollout.mid",
            summary_path=OUT_DIR / f"{name}_v1_summary.json",
            note_events=events,
            build_library=False,
            controller=controller,
        )
        combined["sequences"][name] = {
            "pitches": list(pitch for pitch in (pitches or [69, 73, 75, 71])),
            "v0": asdict(v0),
            "v1": asdict(v1),
        }
    combined_path = OUT_DIR / "hybrid_comparison_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={combined_path}")
    print(f"residual_model_paths={residual_paths}")
    for name, payload in combined["sequences"].items():
        print(f"sequence={name} pitches={payload['pitches']}")
        _print_result(f"{name}_v0_action_library", run_pipeline1_result_from_dict(payload["v0"]))
        _print_result(f"{name}_v1_hybrid", run_pipeline1_result_from_dict(payload["v1"]))


def run_pipeline1_result_from_dict(payload):
    from ala_pianist.pipelines.pipeline1 import NoteRolloutMetric, Pipeline1Result

    metrics = tuple(NoteRolloutMetric(**metric) for metric in payload["note_metrics"])
    return Pipeline1Result(
        expected_pitches=tuple(payload["expected_pitches"]),
        transcribed_pitches=tuple(payload["transcribed_pitches"]),
        transcription=payload["transcription"],
        note_metrics=metrics,
        target_recall=float(payload["target_recall"]),
        wrong_key_rate=float(payload["wrong_key_rate"]),
        max_unintended_key_state=float(payload["max_unintended_key_state"]),
        clean_hit_count=int(payload["clean_hit_count"]),
        dirty_hit_count=int(payload["dirty_hit_count"]),
        miss_count=int(payload["miss_count"]),
    )


if __name__ == "__main__":
    main()
