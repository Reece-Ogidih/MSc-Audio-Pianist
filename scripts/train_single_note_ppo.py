import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO

from ala_pianist.learning.random_search import run_single_note_random_search
from ala_pianist.rl import SingleNotePianoGymEnv, write_single_note_rl_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "single_note_ppo_dsharp5.mid"
OUT_DIR = ROOT / "experiments" / "sb3_single_note"


def evaluate_policy(model, midi_path: Path, *, episodes: int = 3, horizon_steps: int = 32):
    episode_results = []
    for _ in range(episodes):
        env = SingleNotePianoGymEnv(midi_path, horizon_steps=horizon_steps)
        obs, info = env.reset()
        total_reward = 0.0
        native_reward_sum = 0.0
        max_target = 0.0
        max_unintended = 0.0
        clean_count = 0
        near_clean_count = 0
        wrong_pressed_keys = set()
        for _step in range(horizon_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info["native_reward"] is not None:
                native_reward_sum += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            clean_count += int(bool(info["clean_target_press"]))
            near_clean_count += int(bool(info["near_clean_partial_press"]))
            wrong_pressed_keys.update(
                key for key in info["pressed_keys"] if key != info["target_key"]
            )
            if terminated or truncated:
                break
        episode_results.append(
            {
                "debug_return": total_reward,
                "native_reward_sum": native_reward_sum,
                "max_target_key_state": max_target,
                "max_unintended_key_state": max_unintended,
                "clean_target_press_count": clean_count,
                "near_clean_partial_press_count": near_clean_count,
                "wrong_pressed_keys": sorted(wrong_pressed_keys),
                "final_info": dict(info),
            }
        )
    return {
        "episodes": episode_results,
        "mean_debug_return": float(np.mean([r["debug_return"] for r in episode_results])),
        "mean_max_target_key_state": float(
            np.mean([r["max_target_key_state"] for r in episode_results])
        ),
        "mean_max_unintended_key_state": float(
            np.mean([r["max_unintended_key_state"] for r in episode_results])
        ),
        "clean_episode_count": int(
            sum(r["clean_target_press_count"] > 0 for r in episode_results)
        ),
        "near_clean_episode_count": int(
            sum(r["near_clean_partial_press_count"] > 0 for r in episode_results)
        ),
    }


def _result_summary(result):
    return {
        "outcome": result.outcome,
        "debug_return": result.debug_return,
        "native_reward_sum": result.native_reward_sum,
        "max_target_key_state": result.max_target_key_state,
        "max_unintended_key_state": result.max_unintended_key_state,
        "pressed_keys_seen": list(result.pressed_keys_seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--horizon-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_single_note_rl_midi(MIDI_PATH)

    baseline_summary = run_single_note_random_search(
        MIDI_PATH,
        candidate_count=10,
        horizon_steps=args.horizon_steps,
        seed=args.seed,
    )

    env = SingleNotePianoGymEnv(MIDI_PATH, horizon_steps=args.horizon_steps)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.95,
    )
    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    runtime_seconds = time.time() - start

    model_path = OUT_DIR / "single_note_ppo_model"
    model.save(model_path)
    eval_summary = evaluate_policy(
        model,
        MIDI_PATH,
        episodes=3,
        horizon_steps=args.horizon_steps,
    )
    summary = {
        "target": {
            "midi": baseline_summary.target_midi,
            "key": baseline_summary.target_key,
            "note": baseline_summary.target_note,
        },
        "timesteps": args.timesteps,
        "horizon_steps": args.horizon_steps,
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
        "model_path": str(model_path) + ".zip",
        "zero_baseline": _result_summary(baseline_summary.zero_result),
        "scripted_baseline": _result_summary(baseline_summary.scripted_result),
        "random_search_baseline": _result_summary(baseline_summary.best_result),
        "ppo_policy": eval_summary,
    }
    summary_path = OUT_DIR / "single_note_ppo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={summary_path}")
    print(f"model_path={model_path}.zip")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"target={summary['target']}")
    print(f"timesteps={args.timesteps}")
    print(f"horizon_steps={args.horizon_steps}")
    print(f"zero_baseline={summary['zero_baseline']}")
    print(f"scripted_baseline={summary['scripted_baseline']}")
    print(f"random_search_baseline={summary['random_search_baseline']}")
    print(f"ppo_policy={summary['ppo_policy']}")


if __name__ == "__main__":
    main()
