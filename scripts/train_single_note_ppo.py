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


def evaluate_policy(
    model,
    midi_path: Path,
    *,
    deterministic: bool,
    episodes: int = 3,
    horizon_steps: int = 32,
):
    episode_results = []
    all_actions = []
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
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            all_actions.append(action.copy())
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
        "deterministic": deterministic,
        "episodes": episode_results,
        "action_stats": _action_stats(all_actions),
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


def train_budget(
    *,
    timesteps: int,
    seed: int,
    horizon_steps: int,
    baseline_summary,
) -> dict:
    env = SingleNotePianoGymEnv(MIDI_PATH, horizon_steps=horizon_steps)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.95,
    )
    start = time.time()
    model.learn(total_timesteps=timesteps)
    runtime_seconds = time.time() - start

    model_path = OUT_DIR / f"single_note_ppo_{timesteps}_steps"
    model.save(model_path)
    deterministic_eval = evaluate_policy(
        model,
        MIDI_PATH,
        deterministic=True,
        episodes=3,
        horizon_steps=horizon_steps,
    )
    stochastic_eval = evaluate_policy(
        model,
        MIDI_PATH,
        deterministic=False,
        episodes=3,
        horizon_steps=horizon_steps,
    )
    return {
        "timesteps": timesteps,
        "horizon_steps": horizon_steps,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "model_path": str(model_path) + ".zip",
        "zero_baseline": _result_summary(baseline_summary.zero_result),
        "scripted_baseline": _result_summary(baseline_summary.scripted_result),
        "random_search_baseline": _result_summary(baseline_summary.best_result),
        "ppo_deterministic": deterministic_eval,
        "ppo_stochastic": stochastic_eval,
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


def _action_stats(actions) -> dict:
    if not actions:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean_abs": 0.0,
            "near_zero": True,
            "saturation_fraction": 0.0,
        }
    arr = np.asarray(actions, dtype=float)
    mean_abs = float(np.mean(np.abs(arr)))
    saturation_fraction = float(np.mean(np.abs(arr) >= 0.95))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": mean_abs,
        "near_zero": bool(mean_abs < 0.05),
        "saturation_fraction": saturation_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--budget-sweep", type=int, nargs="*", default=None)
    parser.add_argument("--horizon-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_single_note_rl_midi(MIDI_PATH)

    audit_env = SingleNotePianoGymEnv(MIDI_PATH, horizon_steps=args.horizon_steps)
    native_low, native_high = audit_env.native_action_bounds
    print("action_scaling=normalized Box(-1, 1) exposed to SB3; rescaled internally to native 22D bounds")
    print(f"gym_action_low_min={float(audit_env.action_space.low.min()):.1f}")
    print(f"gym_action_high_max={float(audit_env.action_space.high.max()):.1f}")
    print(f"native_action_low_min={float(native_low.min()):.6f}")
    print(f"native_action_high_max={float(native_high.max()):.6f}")

    baseline_summary = run_single_note_random_search(
        MIDI_PATH,
        candidate_count=10,
        horizon_steps=args.horizon_steps,
        seed=args.seed,
    )
    budgets = args.budget_sweep if args.budget_sweep else [args.timesteps]
    budget_results = []
    for timesteps in budgets:
        print(f"training_budget_start={timesteps}")
        result = train_budget(
            timesteps=timesteps,
            seed=args.seed,
            horizon_steps=args.horizon_steps,
            baseline_summary=baseline_summary,
        )
        budget_results.append(result)
        per_budget_path = OUT_DIR / f"single_note_ppo_{timesteps}_summary.json"
        per_budget_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"training_budget_done={timesteps} runtime_seconds={result['runtime_seconds']:.2f}")

    summary = {
        "target": {
            "midi": baseline_summary.target_midi,
            "key": baseline_summary.target_key,
            "note": baseline_summary.target_note,
        },
        "action_scaling": "normalized_minus_one_to_one_rescaled_to_native_22d",
        "horizon_steps": args.horizon_steps,
        "seed": args.seed,
        "budgets": budgets,
        "results": budget_results,
    }
    summary_path = OUT_DIR / "single_note_ppo_budget_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={summary_path}")
    print(f"target={summary['target']}")
    print(f"horizon_steps={args.horizon_steps}")
    print(f"seed={args.seed}")
    print(f"zero_baseline={_result_summary(baseline_summary.zero_result)}")
    print(f"scripted_baseline={_result_summary(baseline_summary.scripted_result)}")
    print(f"random_search_baseline={_result_summary(baseline_summary.best_result)}")
    for result in budget_results:
        print(
            f"budget={result['timesteps']} "
            f"deterministic_mean_target={result['ppo_deterministic']['mean_max_target_key_state']:.6f} "
            f"deterministic_mean_unintended={result['ppo_deterministic']['mean_max_unintended_key_state']:.6f} "
            f"deterministic_action_stats={result['ppo_deterministic']['action_stats']} "
            f"stochastic_mean_target={result['ppo_stochastic']['mean_max_target_key_state']:.6f} "
            f"stochastic_mean_unintended={result['ppo_stochastic']['mean_max_unintended_key_state']:.6f} "
            f"stochastic_action_stats={result['ppo_stochastic']['action_stats']}"
        )


if __name__ == "__main__":
    main()
