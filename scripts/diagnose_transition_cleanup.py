"""Diagnose transition/release failures for the C#5-D#5 direct policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.music import (
    assign_right_hand_fingering,
    generate_sequence_events,
    note_windows as shared_note_windows,
    sequence_timing_from_profile,
    write_sequence_midi as shared_write_sequence_midi,
)
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


def write_sequence_midi(pitches: list[int], path: Path, *, timing_profile: str = "aligned") -> Path:
    midi_min = min(pitches)
    midi_max = max(pitches)
    return shared_write_sequence_midi(
        pitches,
        path,
        midi_min=midi_min,
        midi_max=midi_max,
        timing=sequence_timing_from_profile(timing_profile),
        fingering_fn=assign_right_hand_fingering,
        title=f"Transition cleanup {timing_profile} diagnostic",
    )


def rollout(
    model: SAC,
    pitches: list[int],
    *,
    seed: int,
    horizon_steps: int,
    timing_profile: str = "aligned",
) -> dict[str, Any]:
    midi_path = write_sequence_midi(
        pitches,
        OUT_DIR / "diagnostic_midi" / f"transition_{timing_profile}_{'_'.join(map(str, pitches))}.mid",
        timing_profile=timing_profile,
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
        "note_windows": note_windows(pitches, timing_profile=timing_profile),
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


def note_windows(pitches: list[int], *, timing_profile: str = "aligned") -> list[dict[str, float | int]]:
    return list(shared_note_windows(pitches, timing=sequence_timing_from_profile(timing_profile)))


def compare_timing(pitches: list[int], *, horizon_steps: int, seed: int) -> dict[str, Any]:
    aligned_events = generate_sequence_events(
        pitches,
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
    )
    legacy_events = generate_sequence_events(
        pitches,
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("legacy_curriculum"),
        fingering_fn=assign_right_hand_fingering,
    )
    aligned_midi = write_sequence_midi(
        pitches,
        OUT_DIR / "diagnostic_midi" / "timing_aligned_73_75.mid",
        timing_profile="aligned",
    )
    legacy_midi = write_sequence_midi(
        pitches,
        OUT_DIR / "diagnostic_midi" / "timing_legacy_curriculum_73_75.mid",
        timing_profile="legacy_curriculum",
    )
    return {
        "pitches": pitches,
        "aligned_events": event_table(aligned_events),
        "legacy_curriculum_events": event_table(legacy_events),
        "events_identical": event_table(aligned_events) == event_table(legacy_events),
        "aligned_schedule": active_target_schedule(
            aligned_midi,
            horizon_steps=horizon_steps,
            seed=seed,
        ),
        "legacy_curriculum_schedule": active_target_schedule(
            legacy_midi,
            horizon_steps=horizon_steps,
            seed=seed,
        ),
    }


def event_table(events) -> list[dict[str, float | int | None]]:
    return [
        {
            "pitch": int(event.pitch),
            "key_index": int(event.pitch) - 21,
            "start": float(event.start),
            "end": float(event.start + event.duration),
            "duration": float(event.duration),
            "velocity": int(event.velocity),
            "fingering": event.fingering,
        }
        for event in events
    ]


def active_target_schedule(midi_path: Path, *, horizon_steps: int, seed: int) -> list[dict[str, Any]]:
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
        action_mode="direct",
        action_repeat=1,
    )
    env.reset(seed=seed)
    schedule = []
    for step in range(horizon_steps):
        schedule.append({"step": step, "native_target_keys": tuple(env.current_target_keys())})
        _, _, terminated, truncated, _ = env.step(np.zeros(22, dtype=np.float32))
        if terminated or truncated:
            break
    return schedule


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
        "--sequence-timing-profile",
        default="aligned",
        choices=["aligned", "legacy_curriculum"],
    )
    parser.add_argument("--compare-timing", action="store_true")
    parser.add_argument(
        "--output-path",
        default=str(OUT_DIR / "transition_diagnostics_stage2e.json"),
    )
    args = parser.parse_args()

    if args.compare_timing:
        comparison = compare_timing([73, 75], horizon_steps=args.horizon_steps, seed=args.seed)
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        print(f"summary_path={output_path}")
        print(f"events_identical={comparison['events_identical']}")
        print(f"aligned_events={comparison['aligned_events']}")
        print(f"legacy_curriculum_events={comparison['legacy_curriculum_events']}")
        print(f"aligned_schedule_first12={comparison['aligned_schedule'][:12]}")
        print(f"legacy_schedule_first12={comparison['legacy_curriculum_schedule'][:12]}")
        return

    model = SAC.load(args.model_path)
    results = {
        "model_path": str(args.model_path),
        "rollouts": {
            name: rollout(
                model,
                pitches,
                seed=args.seed,
                horizon_steps=args.horizon_steps,
                timing_profile=args.sequence_timing_profile,
            )
            for name, pitches in SEQUENCES.items()
        },
        "sequence_timing_profile": args.sequence_timing_profile,
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
