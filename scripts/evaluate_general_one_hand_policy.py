import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.controllers import HybridPipeline1Controller
from ala_pianist.music import (
    assign_right_hand_fingering,
    sequence_timing_from_profile,
    write_sequence_midi,
)
from ala_pianist.pipelines.pipeline1 import pipeline1_events_from_pitches, run_pipeline1
from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand"
D_SHARP_MODEL = ROOT / "experiments" / "residual_single_note" / "residual_sac_cleanliness_scale_0.1"
D5_MODEL = ROOT / "experiments" / "residual_single_note" / "residual_sac_midi74_constrained_cleanliness_scale_0p03_penalty_0p1.zip"


SEQUENCES = {
    "single_csharp5": [73],
    "single_d5": [74],
    "single_dsharp5": [75],
    "csharp5_d5": [73, 74],
    "d5_dsharp5_once": [74, 75],
    "dsharp5_d5": [75, 74],
    "d5_csharp5": [74, 73],
    "stage2_pair": [73, 75],
    "stage2_reverse_pair": [75, 73],
    "stage2_return_pair": [73, 75, 73],
    "debug_keyset": [73, 74, 75],
    "debug_keyset_reverse": [75, 74, 73],
    "current_phrase": [69, 73, 75, 71],
    "d5_dsharp5": [74, 75, 74, 75],
    "local_range_phrase": [69, 71, 73, 74, 75],
}


def evaluate_general_model(
    model_path: Path,
    pitches: list[int],
    *,
    seed: int,
    horizon_steps: int,
    action_mode: str,
    action_repeat: int,
    ramp_steps: int,
    reward_config: GeneralRewardConfig,
    timing_profile: str,
) -> dict:
    model = SAC.load(model_path)
    midi_path = _write_sequence_midi(
        pitches,
        OUT_DIR / "eval_midi" / f"general_{timing_profile}_{'_'.join(map(str, pitches))}.mid",
        timing_profile=timing_profile,
    )
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
        action_mode=action_mode,
        action_repeat=action_repeat,
        ramp_steps=ramp_steps,
        reward_config=reward_config,
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
        "action_mode": action_mode,
        "action_repeat": action_repeat,
        "ramp_steps": ramp_steps,
        "wrong_key_count": len(wrong_pressed),
        "wrong_pressed_keys": wrong_pressed,
        "max_target_key_state": max_target,
        "max_unintended_key_state": max_unintended,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward_sum,
        "pressed_keys": sorted(pressed_keys),
        "final_info": dict(info),
    }


def evaluate_zero(
    pitches: list[int],
    *,
    seed: int,
    horizon_steps: int,
    timing_profile: str,
) -> dict:
    midi_path = _write_sequence_midi(
        pitches,
        OUT_DIR / "eval_midi" / f"zero_{'_'.join(map(str, pitches))}.mid",
        timing_profile=timing_profile,
    )
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


def _write_sequence_midi(pitches: list[int], path: Path, *, timing_profile: str) -> Path:
    midi_min = min(pitches)
    midi_max = max(pitches)
    return write_sequence_midi(
        pitches,
        path,
        midi_min=midi_min,
        midi_max=midi_max,
        timing=sequence_timing_from_profile(timing_profile),
        fingering_fn=assign_right_hand_fingering,
        title=f"general one hand {timing_profile} evaluation sequence",
    )


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


def reward_config_from_profile(profile: str) -> GeneralRewardConfig:
    if profile == "default":
        return GeneralRewardConfig()
    if profile == "press_bonus":
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=1.5,
            wrong_pressed_weight=1.0,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=5.0,
            target_activation_threshold=0.9,
            high_unintended_weight=1.0,
            high_unintended_threshold=0.75,
        )
    if profile in {"cleanup", "gated_cleanliness"}:
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=0.25,
            wrong_pressed_weight=0.25,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=3.0,
            target_activation_threshold=0.9,
            high_unintended_weight=0.5,
            high_unintended_threshold=0.75,
            cleanup_gate_threshold=0.75,
            gated_unintended_weight=3.0,
            gated_wrong_pressed_weight=2.0,
            nearby_wrong_key_weight=2.0,
        )
    if profile == "anti_coupling":
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=0.20,
            wrong_pressed_weight=0.25,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=3.0,
            target_activation_threshold=0.9,
            high_unintended_weight=0.5,
            high_unintended_threshold=0.75,
            cleanup_gate_threshold=0.75,
            gated_unintended_weight=2.0,
            gated_wrong_pressed_weight=1.5,
            nearby_wrong_key_weight=1.0,
            csharp_dsharp_key54_weight=5.0,
            csharp_dsharp_pressed_weight=3.0,
            dsharp_csharp_key52_weight=2.0,
            dsharp_csharp_pressed_weight=1.0,
        )
    if profile == "transition_cleanup":
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=0.25,
            wrong_pressed_weight=0.30,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=3.0,
            target_activation_threshold=0.9,
            high_unintended_weight=0.5,
            high_unintended_threshold=0.75,
            cleanup_gate_threshold=0.75,
            gated_unintended_weight=1.5,
            gated_wrong_pressed_weight=1.0,
            nearby_wrong_key_weight=1.0,
            csharp_dsharp_key54_weight=4.0,
            csharp_dsharp_pressed_weight=2.5,
            dsharp_csharp_key52_weight=1.5,
            dsharp_csharp_pressed_weight=0.75,
            release_previous_key_weight=1.5,
            transition_stray_key_weight=3.0,
            transition_stray_pressed_weight=1.5,
        )
    raise ValueError(f"Unknown reward profile {profile!r}.")


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
    parser.add_argument("--action-mode", default="direct", choices=["direct", "hold", "ramp_hold"])
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--ramp-steps", type=int, default=1)
    parser.add_argument(
        "--sequence-timing-profile",
        default="aligned",
        choices=["aligned", "legacy_curriculum"],
    )
    parser.add_argument(
        "--reward-profile",
        default="default",
        choices=[
            "default",
            "press_bonus",
            "cleanup",
            "gated_cleanliness",
            "anti_coupling",
            "transition_cleanup",
        ],
    )
    parser.add_argument("--sequence", action="append", choices=sorted(SEQUENCES), default=None)
    parser.add_argument("--dsharp-residual-model-path", default=str(D_SHARP_MODEL))
    parser.add_argument("--d5-residual-model-path", default=str(D5_MODEL))
    args = parser.parse_args()

    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    reward_config = reward_config_from_profile(args.reward_profile)
    stage_models = [("general_policy", model_path)]
    stage_models.extend(parse_stage_model(raw) for raw in args.compare_model_path)
    hybrid = maybe_hybrid_controller(
        Path(args.dsharp_residual_model_path) if args.dsharp_residual_model_path else None,
        Path(args.d5_residual_model_path) if args.d5_residual_model_path else None,
    )

    selected_sequences = {
        name: pitches
        for name, pitches in SEQUENCES.items()
        if args.sequence is None or name in set(args.sequence)
    }

    summary = {}
    for name, pitches in selected_sequences.items():
        print(f"evaluating_sequence={name} pitches={pitches}")
        sequence_summary = {
            "zero": evaluate_zero(
                pitches,
                seed=args.seed,
                horizon_steps=args.horizon_steps,
                timing_profile=args.sequence_timing_profile,
            ),
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
                action_mode=args.action_mode,
                action_repeat=args.action_repeat,
                ramp_steps=args.ramp_steps,
                reward_config=reward_config,
                timing_profile=args.sequence_timing_profile,
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
    payload = {
        "action_mode": args.action_mode,
        "action_repeat": args.action_repeat,
        "ramp_steps": args.ramp_steps,
        "reward_profile": args.reward_profile,
        "sequence_timing_profile": args.sequence_timing_profile,
        "sequences": summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
