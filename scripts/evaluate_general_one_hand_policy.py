import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.controllers import HybridPipeline1Controller
from ala_pianist.music import NoteEvent, assign_right_hand_fingering, write_monophonic_midi
from ala_pianist.pipelines.pipeline1 import pipeline1_events_from_pitches, run_pipeline1
from ala_pianist.rl import GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand"
D_SHARP_MODEL = ROOT / "experiments" / "residual_single_note" / "residual_sac_cleanliness_scale_0.1"
D5_MODEL = ROOT / "experiments" / "residual_single_note" / "residual_sac_midi74_constrained_cleanliness_scale_0p03_penalty_0p1.zip"


SEQUENCES = {
    "single_csharp5": [73],
    "single_dsharp5": [75],
    "stage2_pair": [73, 75],
    "debug_keyset": [73, 74, 75],
    "current_phrase": [69, 73, 75, 71],
    "d5_dsharp5": [74, 75, 74, 75],
    "local_range_phrase": [69, 71, 73, 74, 75],
}


def evaluate_general_model(model_path: Path, pitches: list[int], *, seed: int, horizon_steps: int) -> dict:
    model = SAC.load(model_path)
    midi_path = _write_sequence_midi(pitches, OUT_DIR / "eval_midi" / f"general_{'_'.join(map(str, pitches))}.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
    )
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    native_reward_sum = 0.0
    max_target = 0.0
    max_unintended = 0.0
    pressed_keys = set()
    for _ in range(horizon_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        native_reward_sum += float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed_keys.update(info["pressed_keys"])
        if terminated or truncated:
            break
    target_keys = {pitch - 21 for pitch in pitches}
    target_hits = len([key for key in target_keys if key in pressed_keys])
    wrong_pressed = [key for key in sorted(pressed_keys) if key not in target_keys]
    return {
        "clip_pitches": pitches,
        "midi_path": str(midi_path),
        "target_recall": target_hits / max(1, len(target_keys)),
        "strict_outcome": _strict_outcome(target_keys, pressed_keys, max_target, max_unintended),
        "wrong_key_count": len(wrong_pressed),
        "wrong_pressed_keys": wrong_pressed,
        "max_target_key_state": max_target,
        "max_unintended_key_state": max_unintended,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward_sum,
        "pressed_keys": sorted(pressed_keys),
        "final_info": dict(info),
    }


def evaluate_zero(pitches: list[int], *, seed: int, horizon_steps: int) -> dict:
    midi_path = _write_sequence_midi(pitches, OUT_DIR / "eval_midi" / f"zero_{'_'.join(map(str, pitches))}.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
    )
    obs, info = env.reset(seed=seed)
    del obs
    total_reward = 0.0
    max_target = 0.0
    max_unintended = 0.0
    pressed_keys = set()
    for _ in range(horizon_steps):
        _, reward, terminated, truncated, info = env.step(np.zeros(22, dtype=np.float32))
        total_reward += float(reward)
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed_keys.update(info["pressed_keys"])
        if terminated or truncated:
            break
    return {
        "midi_path": str(midi_path),
        "max_target_key_state": max_target,
        "max_unintended_key_state": max_unintended,
        "shaped_return": total_reward,
        "pressed_keys": sorted(pressed_keys),
        "trajectory_quality": info["trajectory_quality"],
    }


def _write_sequence_midi(pitches: list[int], path: Path) -> Path:
    midi_min = min(pitches)
    midi_max = max(pitches)
    events = [
        NoteEvent(
            pitch=int(pitch),
            start=0.40 * index,
            duration=0.28,
            velocity=90,
            fingering=assign_right_hand_fingering(int(pitch), midi_min, midi_max),
        )
        for index, pitch in enumerate(pitches)
    ]
    return write_monophonic_midi(events, path, title="general one hand evaluation sequence")


def evaluate_pipeline_baseline(
    name: str,
    pitches: list[int],
    *,
    controller=None,
    build_library: bool = True,
) -> dict:
    result = run_pipeline1(
        audio_path=OUT_DIR / f"{name}.wav",
        library_path=OUT_DIR / "pipeline_action_library.json",
        rollout_midi_path=OUT_DIR / f"{name}_rollout.mid",
        note_events=pipeline1_events_from_pitches(pitches),
        build_library=build_library,
        controller=controller,
    )
    return asdict(result)


def maybe_hybrid_controller(dsharp_path: Path | None, d5_path: Path | None) -> HybridPipeline1Controller | None:
    policies = {}
    if dsharp_path is not None and dsharp_path.exists():
        policies[75] = dsharp_path
    if d5_path is not None and d5_path.exists():
        policies[74] = d5_path
    return HybridPipeline1Controller(residual_model_paths=policies) if policies else None


def parse_stage_model(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    path = Path(raw)
    return path.stem, path


def _strict_outcome(
    target_keys: set[int],
    pressed_keys: set[int],
    max_target: float,
    max_unintended: float,
) -> str:
    if target_keys and target_keys.issubset(pressed_keys) and not (pressed_keys - target_keys):
        return "clean_low_unintended" if max_unintended < 0.25 else "clean_high_unintended"
    if target_keys and target_keys.intersection(pressed_keys):
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compare-model-path", action="append", default=[])
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--dsharp-residual-model-path", default=str(D_SHARP_MODEL))
    parser.add_argument("--d5-residual-model-path", default=str(D5_MODEL))
    args = parser.parse_args()

    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    stage_models = [("general_policy", model_path)]
    stage_models.extend(parse_stage_model(raw) for raw in args.compare_model_path)
    hybrid = maybe_hybrid_controller(
        Path(args.dsharp_residual_model_path) if args.dsharp_residual_model_path else None,
        Path(args.d5_residual_model_path) if args.d5_residual_model_path else None,
    )

    summary = {}
    for name, pitches in SEQUENCES.items():
        print(f"evaluating_sequence={name} pitches={pitches}")
        sequence_summary = {
            "zero": evaluate_zero(pitches, seed=args.seed, horizon_steps=args.horizon_steps),
            "action_library_v0": evaluate_pipeline_baseline(
                f"{name}_v0",
                pitches,
                controller=None,
                build_library=True,
            ),
        }
        for label, path in stage_models:
            sequence_summary[label] = evaluate_general_model(
                path,
                pitches,
                seed=args.seed,
                horizon_steps=args.horizon_steps,
            )
        if hybrid is not None:
            sequence_summary["hybrid_residual"] = evaluate_pipeline_baseline(
                f"{name}_hybrid",
                pitches,
                controller=hybrid,
                build_library=False,
            )
        summary[name] = sequence_summary

    summary_path = OUT_DIR / "general_one_hand_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_path={summary_path}")
    for name, sequence in summary.items():
        general = sequence["general_policy"]
        zero = sequence["zero"]
        print(
            f"{name}: zero_target={zero['max_target_key_state']:.6f} "
            f"general_target={general['max_target_key_state']:.6f} "
            f"general_unintended={general['max_unintended_key_state']:.6f} "
            f"general_recall={general['target_recall']:.3f} "
            f"general_pressed={general['pressed_keys']} "
            f"general_strict={general['strict_outcome']}"
        )
        for label, result in sequence.items():
            if label in {"zero", "action_library_v0", "hybrid_residual", "general_policy"}:
                continue
            print(
                f"{name} {label}: target={result['max_target_key_state']:.6f} "
                f"unintended={result['max_unintended_key_state']:.6f} "
                f"recall={result['target_recall']:.3f} "
                f"pressed={result['pressed_keys']} "
                f"strict={result['strict_outcome']}"
            )


if __name__ == "__main__":
    main()
