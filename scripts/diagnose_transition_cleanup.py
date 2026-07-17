"""Diagnose transition/release failures for the C#5-D#5 direct policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.music import NoteEvent, assign_right_hand_fingering, write_monophonic_midi
from ala_pianist.rl import GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand"
DEFAULT_MODEL = (
    OUT_DIR
    / "stage2e_direct_anti_coupling_73_75_general_one_hand_sac_single_notes_pitches73-75_lookahead1_directx1_anti_coupling_seed13_1000000.zip"
)
SEQUENCES = {
    "single_csharp5": [73],
    "single_dsharp5": [75],
    "csharp_dsharp": [73, 75],
    "dsharp_csharp": [75, 73],
    "csharp_dsharp_csharp": [73, 75, 73],
}
TRACKED_KEYS = (52, 54, 55, 56, 57)
WRONG_TRANSITION_KEYS = (55, 56, 57)


def write_sequence_midi(pitches: list[int], path: Path) -> Path:
    midi_min = min(pitches)
    midi_max = max(pitches)
    events = []
    for index, pitch in enumerate(pitches):
        events.append(
            NoteEvent(
                pitch=int(pitch),
                start=0.40 * index,
                duration=0.28,
                velocity=90,
                fingering=assign_right_hand_fingering(int(pitch), midi_min, midi_max),
            )
        )
    return write_monophonic_midi(events, path, title="Transition cleanup diagnostic")


def rollout(model: SAC, pitches: list[int], *, seed: int, horizon_steps: int) -> dict[str, Any]:
    midi_path = write_sequence_midi(
        pitches,
        OUT_DIR / "diagnostic_midi" / f"transition_{'_'.join(map(str, pitches))}.mid",
    )
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
        action_mode="direct",
        action_repeat=1,
    )
    obs, _ = env.reset(seed=seed)
    records = []
    actions = []
    pressed_union: set[int] = set()
    first_wrong_activation: dict[int, dict[str, Any]] = {}
    previous_nonempty_target: tuple[int, ...] = ()
    for step in range(horizon_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        states = env.piano_key_states()
        active_target = tuple(int(key) for key in info["target_keys"])
        if active_target:
            previous_nonempty_target = active_target
        phase = classify_phase(active_target, previous_nonempty_target)
        pressed = tuple(int(key) for key in info["pressed_keys"])
        pressed_union.update(pressed)
        for key in WRONG_TRANSITION_KEYS:
            if key not in first_wrong_activation and (states[key] >= 0.25 or key in pressed):
                first_wrong_activation[key] = {
                    "step": step,
                    "state": float(states[key]),
                    "phase": phase,
                    "active_target": active_target,
                    "pressed": key in pressed,
                }
        records.append(
            {
                "step": step,
                "reward": float(reward),
                "active_target_keys": active_target,
                "phase": phase,
                "pressed_keys": pressed,
                "target_key_state": float(info["target_key_state"]),
                "max_unintended_key_state": float(info["max_unintended_key_state"]),
                "key_states": {str(key): float(states[key]) for key in TRACKED_KEYS},
                "reward_components": dict(info["reward_components"]),
            }
        )
        if terminated or truncated:
            break
    target_keys = {pitch - 21 for pitch in pitches}
    strict = strict_outcome(target_keys, pressed_union, records)
    return {
        "pitches": pitches,
        "note_windows": note_windows(pitches),
        "pressed_keys": sorted(pressed_union),
        "strict_outcome": strict,
        "first_wrong_activation": first_wrong_activation,
        "max_states": {
            str(key): max((record["key_states"][str(key)] for record in records), default=0.0)
            for key in TRACKED_KEYS
        },
        "max_target_key_state": max((record["target_key_state"] for record in records), default=0.0),
        "max_unintended_key_state": max(
            (record["max_unintended_key_state"] for record in records),
            default=0.0,
        ),
        "actions": np.asarray(actions, dtype=float).tolist(),
        "records": records,
    }


def classify_phase(active_target: tuple[int, ...], previous_target: tuple[int, ...]) -> str:
    if 52 in active_target:
        return "csharp5_target_window"
    if 54 in active_target:
        return "dsharp5_target_window"
    if 52 in previous_target:
        return "release_after_csharp5"
    if 54 in previous_target:
        return "release_after_dsharp5"
    return "pre_target_or_silence"


def note_windows(pitches: list[int]) -> list[dict[str, float | int]]:
    return [
        {
            "pitch": int(pitch),
            "key_index": int(pitch) - 21,
            "start_seconds": 0.40 * index,
            "end_seconds": 0.40 * index + 0.28,
        }
        for index, pitch in enumerate(pitches)
    ]


def strict_outcome(target_keys: set[int], pressed: set[int], records: list[dict[str, Any]]) -> str:
    max_target = max((record["target_key_state"] for record in records), default=0.0)
    max_unintended = max((record["max_unintended_key_state"] for record in records), default=0.0)
    if target_keys and target_keys.issubset(pressed) and not (pressed - target_keys):
        return "clean_low_unintended" if max_unintended < 0.25 else "clean_high_unintended"
    if target_keys and target_keys.intersection(pressed):
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--horizon-steps", type=int, default=112)
    parser.add_argument(
        "--output-path",
        default=str(OUT_DIR / "transition_diagnostics_stage2e.json"),
    )
    args = parser.parse_args()

    model = SAC.load(args.model_path)
    results = {
        "model_path": str(args.model_path),
        "rollouts": {
            name: rollout(model, pitches, seed=args.seed, horizon_steps=args.horizon_steps)
            for name, pitches in SEQUENCES.items()
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_path={output_path}")
    for name, item in results["rollouts"].items():
        print(
            f"{name}: pressed={item['pressed_keys']} strict={item['strict_outcome']} "
            f"target={item['max_target_key_state']:.3f} unintended={item['max_unintended_key_state']:.3f} "
            f"key52={item['max_states']['52']:.3f} key54={item['max_states']['54']:.3f} "
            f"key55={item['max_states']['55']:.3f} key56={item['max_states']['56']:.3f} "
            f"key57={item['max_states']['57']:.3f}"
        )
        for key in WRONG_TRANSITION_KEYS:
            activation = item["first_wrong_activation"].get(key)
            if activation is not None:
                print(
                    f"  key{key}_first_activation: step={activation['step']} "
                    f"phase={activation['phase']} state={activation['state']:.3f} "
                    f"pressed={activation['pressed']}"
                )


if __name__ == "__main__":
    main()
