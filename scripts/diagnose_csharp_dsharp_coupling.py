"""Diagnose C#5/D#5 coupling in a trained general one-hand policy."""

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
    / "stage2c_plus1m_73_75_hold_cleanup_general_one_hand_sac_single_notes_pitches73-75_lookahead1_holdx2_cleanup_seed13_1000000.zip"
)
SEQUENCES = {
    "single_csharp5": [73],
    "single_dsharp5": [75],
    "csharp_dsharp_pair": [73, 75],
}


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
    return write_monophonic_midi(events, path, title="C#5/D#5 coupling diagnostic")


def rollout(
    model: SAC,
    pitches: list[int],
    *,
    action_mode: str,
    action_repeat: int,
    horizon_steps: int,
    seed: int,
) -> dict[str, Any]:
    midi_path = write_sequence_midi(
        pitches,
        OUT_DIR / "diagnostic_midi" / f"coupling_{'_'.join(map(str, pitches))}.mid",
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
    )
    obs, _ = env.reset(seed=seed)
    records = []
    actions = []
    pressed_union: set[int] = set()
    first_key52_pressed = None
    first_key54_pressed = None
    for step in range(horizon_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        states = env.piano_key_states()
        pressed = tuple(int(key) for key in info["pressed_keys"])
        pressed_union.update(pressed)
        if first_key52_pressed is None and 52 in pressed:
            first_key52_pressed = step
        if first_key54_pressed is None and 54 in pressed:
            first_key54_pressed = step
        active_targets = tuple(int(key) for key in info["target_keys"])
        target_key = active_targets[0] if active_targets else None
        nearest = env.nearest_fingertip_to_key(target_key) if target_key is not None else None
        records.append(
            {
                "step": step,
                "reward": float(reward),
                "target_keys": active_targets,
                "pressed_keys": pressed,
                "key52_state": float(states[52]),
                "key54_state": float(states[54]),
                "target_key_state": float(info["target_key_state"]),
                "max_unintended_key_state": float(info["max_unintended_key_state"]),
                "closest_fingertip": None if nearest is None else nearest["fingertip"],
                "closest_fingertip_distance": None if nearest is None else float(nearest["distance"]),
                "contacts": contact_pairs(env),
            }
        )
        if terminated or truncated:
            break
    timing = activation_timing(first_key52_pressed, first_key54_pressed)
    action_array = np.asarray(actions, dtype=float) if actions else np.zeros((0, 22), dtype=float)
    return {
        "pitches": pitches,
        "action_mode": action_mode,
        "action_repeat": action_repeat,
        "pressed_keys": sorted(pressed_union),
        "first_key52_pressed_step": first_key52_pressed,
        "first_key54_pressed_step": first_key54_pressed,
        "key54_vs_key52_timing": timing,
        "max_key52_state": max((item["key52_state"] for item in records), default=0.0),
        "max_key54_state": max((item["key54_state"] for item in records), default=0.0),
        "max_target_key_state": max((item["target_key_state"] for item in records), default=0.0),
        "max_unintended_key_state": max(
            (item["max_unintended_key_state"] for item in records),
            default=0.0,
        ),
        "action_stats": action_stats(action_array),
        "actions": action_array.tolist(),
        "records": records,
    }


def contact_pairs(env: GeneralOneHandGoalEnv) -> list[dict[str, str]]:
    physics = env.env.physics
    pairs = []
    for idx in range(int(physics.data.ncon)):
        contact = physics.data.contact[idx]
        geom_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            try:
                geom_names.append(physics.model.id2name(geom_id, "geom") or str(geom_id))
            except Exception:
                geom_names.append(str(geom_id))
        if any("fftip" in name or "mftip" in name or "rftip" in name or "lftip" in name or "thtip" in name for name in geom_names) or any("key" in name for name in geom_names):
            pairs.append({"geom1": geom_names[0], "geom2": geom_names[1]})
    return pairs


def activation_timing(first52: int | None, first54: int | None) -> str:
    if first52 is None and first54 is None:
        return "neither_key_pressed"
    if first52 is None:
        return "key54_without_key52"
    if first54 is None:
        return "key52_without_key54"
    if first54 < first52:
        return "key54_before_key52"
    if first54 == first52:
        return "key54_during_key52"
    return "key54_after_key52"


def action_stats(actions: np.ndarray) -> dict[str, float]:
    if actions.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "saturation_fraction": 0.0}
    return {
        "mean": float(np.mean(actions)),
        "std": float(np.std(actions)),
        "min": float(np.min(actions)),
        "max": float(np.max(actions)),
        "saturation_fraction": float(np.mean(np.abs(actions) >= 0.95)),
    }


def compare_actions(
    action_names: tuple[str, ...],
    csharp_actions: list[list[float]],
    dsharp_actions: list[list[float]],
) -> dict[str, Any]:
    csharp = np.asarray(csharp_actions, dtype=float)
    dsharp = np.asarray(dsharp_actions, dtype=float)
    count = min(len(csharp), len(dsharp))
    if count == 0:
        return {}
    csharp = csharp[:count]
    dsharp = dsharp[:count]
    diff = csharp - dsharp
    l2_by_step = np.linalg.norm(diff, axis=1)
    cosine_by_step = []
    for a, b in zip(csharp, dsharp, strict=True):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine_by_step.append(float(np.dot(a, b) / denom) if denom else 0.0)
    mean_abs_diff = np.mean(np.abs(diff), axis=0)
    top_indices = np.argsort(mean_abs_diff)[::-1][:8]
    return {
        "aligned_steps": int(count),
        "mean_l2_distance": float(np.mean(l2_by_step)),
        "max_l2_distance": float(np.max(l2_by_step)),
        "mean_cosine_similarity": float(np.mean(cosine_by_step)),
        "csharp_saturation_fraction": float(np.mean(np.abs(csharp) >= 0.95)),
        "dsharp_saturation_fraction": float(np.mean(np.abs(dsharp) >= 0.95)),
        "top_differing_dimensions": [
            {
                "index": int(index),
                "name": action_names[int(index)] if int(index) < len(action_names) else str(index),
                "mean_abs_diff": float(mean_abs_diff[int(index)]),
                "csharp_mean": float(np.mean(csharp[:, int(index)])),
                "dsharp_mean": float(np.mean(dsharp[:, int(index)])),
            }
            for index in top_indices
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-path", default=str(OUT_DIR / "coupling_diagnostics_stage2c_plus1m.json"))
    args = parser.parse_args()

    model = SAC.load(args.model_path)
    results: dict[str, Any] = {
        "model_path": str(args.model_path),
        "rollouts": {},
    }
    modes = [("direct", "direct", 1), ("holdx1", "hold", 1), ("holdx2", "hold", 2), ("holdx4", "hold", 4)]
    action_names: tuple[str, ...] | None = None
    for mode_label, action_mode, action_repeat in modes:
        results["rollouts"][mode_label] = {}
        for seq_name, pitches in SEQUENCES.items():
            item = rollout(
                model,
                pitches,
                action_mode=action_mode,
                action_repeat=action_repeat,
                horizon_steps=args.horizon_steps,
                seed=args.seed,
            )
            results["rollouts"][mode_label][seq_name] = item
            if action_names is None:
                env = GeneralOneHandGoalEnv(
                    midi_path=write_sequence_midi([73], OUT_DIR / "diagnostic_midi" / "action_names.mid"),
                    midi_min=73,
                    midi_max=73,
                    lookahead=1,
                )
                action_names = env.action_names

    action_names = action_names or tuple(str(index) for index in range(22))
    results["action_comparisons"] = {}
    for mode_label in results["rollouts"]:
        csharp = results["rollouts"][mode_label]["single_csharp5"]["actions"]
        dsharp = results["rollouts"][mode_label]["single_dsharp5"]["actions"]
        results["action_comparisons"][mode_label] = compare_actions(action_names, csharp, dsharp)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={output_path}")
    for mode_label, sequences in results["rollouts"].items():
        print(f"mode={mode_label}")
        for seq_name, item in sequences.items():
            print(
                f"  {seq_name}: pressed={item['pressed_keys']} "
                f"key52={item['max_key52_state']:.3f} key54={item['max_key54_state']:.3f} "
                f"target={item['max_target_key_state']:.3f} unintended={item['max_unintended_key_state']:.3f} "
                f"timing={item['key54_vs_key52_timing']}"
            )
        comparison = results["action_comparisons"][mode_label]
        print(
            f"  action_compare: l2={comparison['mean_l2_distance']:.3f} "
            f"cosine={comparison['mean_cosine_similarity']:.3f} "
            f"sat73={comparison['csharp_saturation_fraction']:.3f} "
            f"sat75={comparison['dsharp_saturation_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
