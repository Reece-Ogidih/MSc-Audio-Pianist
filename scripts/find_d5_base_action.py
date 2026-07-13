import json
from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.evaluation import record_action_rollout, save_trajectory_json
from ala_pianist.learning.random_search import generate_random_candidates
from ala_pianist.music import NoteEvent, write_monophonic_midi
from ala_pianist.rl.residual_env import (
    get_dirty_csharp5_base_action,
    get_dirty_dsharp5_base_action,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "d5_base_search"
MIDI_PATH = OUT_DIR / "d5_target.mid"
SUMMARY_PATH = OUT_DIR / "d5_base_search_summary.json"
TRAJECTORY_PATH = OUT_DIR / "best_d5_trajectory.json"
TARGET_MIDI = 74
TARGET_KEY = TARGET_MIDI - 21
NEARBY_KEYS = (52, 53, 54, 55, 56)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_monophonic_midi([NoteEvent(TARGET_MIDI, 0.0, 1.0, 90)], MIDI_PATH)
    env = ALAOneHandEnv(MIDI_PATH)
    env.reset()
    candidates = []
    for candidate in generate_random_candidates(env, count=40, seed=74):
        candidates.append(("random", candidate.index, np.asarray(candidate.action, dtype=float)))

    dsharp5_base = get_dirty_dsharp5_base_action(OUT_DIR / "dsharp5_base.mid")
    candidates.extend(_local_perturbations(env, dsharp5_base, source="local_dsharp5", seed=174))
    candidates.append(("dsharp5_base", 0, np.asarray(dsharp5_base, dtype=float)))

    try:
        csharp5_base = get_dirty_csharp5_base_action()
    except FileNotFoundError:
        csharp5_base = None
    if csharp5_base is not None:
        candidates.extend(_local_perturbations(env, csharp5_base, source="local_csharp5", seed=274))
        candidates.append(("csharp5_base", 0, np.asarray(csharp5_base, dtype=float)))

    results = []
    for source, index, action in candidates:
        result = evaluate_candidate(action)
        result.update({"source": source, "candidate_index": index, "action": [float(v) for v in action]})
        results.append(result)
    best = max(results, key=score_result)

    env = ALAOneHandEnv(MIDI_PATH)
    env.reset()
    records = record_action_rollout(
        env,
        target_midi=TARGET_MIDI,
        action=np.asarray(best["action"], dtype=env.action_spec().dtype),
        horizon_steps=24,
        ramp=True,
    )
    save_trajectory_json(records, TRAJECTORY_PATH)

    summary = {
        "target_midi": TARGET_MIDI,
        "target_key": TARGET_KEY,
        "candidate_count": len(candidates),
        "search_budget": {
            "random_candidates": 40,
            "local_dsharp5_perturbations": 40,
            "local_csharp5_perturbations": 40 if csharp5_base is not None else 0,
            "horizon_steps": 24,
        },
        "best": best,
        "usable_for_residual_rl": bool(best["max_target_key_state"] > 0.2 or TARGET_KEY in best["pressed_keys"]),
        "trajectory_path": str(TRAJECTORY_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={SUMMARY_PATH}")
    print(f"trajectory_path={TRAJECTORY_PATH}")
    print(f"target_midi={TARGET_MIDI}")
    print(f"target_key={TARGET_KEY}")
    print(f"candidate_count={len(candidates)}")
    print(f"best_source={best['source']}")
    print(f"best_candidate_index={best['candidate_index']}")
    print(f"best_outcome={best['outcome']}")
    print(f"best_target_state={best['max_target_key_state']:.6f}")
    print(f"best_unintended_state={best['max_unintended_key_state']:.6f}")
    print(f"best_nearby_key_states={best['nearby_key_states']}")
    print(f"best_pressed_keys={best['pressed_keys']}")
    print(f"best_closest_finger={best['closest_finger_to_target']}")
    print(f"best_finger_distance={best['finger_target_distance']}")
    print(f"usable_for_residual_rl={summary['usable_for_residual_rl']}")


def _local_perturbations(env: ALAOneHandEnv, base_action, *, source: str, seed: int):
    rng = np.random.default_rng(seed)
    spec = env.action_spec()
    low = np.asarray(spec.minimum, dtype=float)
    high = np.asarray(spec.maximum, dtype=float)
    span = high - low
    candidates = []
    for idx in range(40):
        scale = 0.08 if idx < 20 else 0.18
        action = np.clip(np.asarray(base_action, dtype=float) + rng.normal(0.0, scale * span), low, high)
        candidates.append((source, idx, action))
    return candidates


def evaluate_candidate(action) -> dict:
    env = ALAOneHandEnv(MIDI_PATH)
    env.reset()
    action = np.asarray(action, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_unintended = 0.0
    max_nearby = {key: 0.0 for key in NEARBY_KEYS}
    pressed = set()
    closest_name = "unknown"
    closest_distance = None
    contacts = set()
    native_reward_sum = 0.0
    for step in range(24):
        timestep = env.step(action * min(1.0, (step + 1) / 6.0))
        target = env.target_key_state(TARGET_KEY) or 0.0
        unintended = env.max_unintended_key_state(TARGET_KEY)
        max_target = max(max_target, target)
        max_unintended = max(max_unintended, unintended)
        for key in max_nearby:
            max_nearby[key] = max(max_nearby[key], env.target_key_state(key) or 0.0)
        pressed.update(env.current_pressed_keys())
        closest = env.nearest_fingertip_to_key(TARGET_KEY)
        if closest is not None:
            closest_name = str(closest["fingertip"])
            distance = float(closest["distance"])
            closest_distance = distance if closest_distance is None else min(closest_distance, distance)
        contacts.update(f"{a} <-> {b}" for a, b in env.key_contact_pairs(TARGET_KEY))
        if env.current_reward() is not None:
            native_reward_sum += float(env.current_reward())
        if timestep.last():
            break
    return {
        "max_target_key_state": float(max_target),
        "max_unintended_key_state": float(max_unintended),
        "nearby_key_states": {str(key): float(value) for key, value in max_nearby.items()},
        "pressed_keys": sorted(pressed),
        "key_pressed": TARGET_KEY in pressed,
        "wrong_keys_pressed": sorted(key for key in pressed if key != TARGET_KEY),
        "closest_finger_to_target": closest_name,
        "finger_target_distance": closest_distance,
        "contact_pairs": sorted(contacts),
        "native_reward_sum": native_reward_sum,
        "outcome": outcome(pressed, max_target, max_unintended),
    }


def outcome(pressed: set[int], max_target: float, max_unintended: float) -> str:
    if pressed == {TARGET_KEY}:
        return "clean"
    if TARGET_KEY in pressed:
        return "dirty"
    if max_target >= 0.25 and max_unintended <= max_target + 0.02:
        return "near_clean"
    if max_target > 0.02:
        return "partial"
    return "missed"


def score_result(result: dict) -> tuple[float, float, float]:
    target = float(result["max_target_key_state"])
    unintended = float(result["max_unintended_key_state"])
    wrong = len(result["wrong_keys_pressed"])
    pressed_bonus = 3.0 if result["key_pressed"] else 0.0
    return (8.0 * target - 2.0 * unintended - wrong + pressed_bonus, target, -unintended)


if __name__ == "__main__":
    main()
