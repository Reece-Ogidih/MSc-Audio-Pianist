import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand"


def evaluate_policy(model, env_kwargs: dict, *, episodes: int = 3, deterministic: bool = True) -> dict:
    episode_results = []
    all_actions = []
    for episode in range(episodes):
        env = GeneralOneHandGoalEnv(**env_kwargs, clip_index=episode)
        obs, info = env.reset(seed=env_kwargs.get("seed", 0) + episode)
        total_reward = 0.0
        native_reward_sum = 0.0
        max_target = 0.0
        max_unintended = 0.0
        pressed_keys = set()
        for _ in range(env.horizon_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            all_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            native_reward_sum += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            pressed_keys.update(info["pressed_keys"])
            if terminated or truncated:
                break
        episode_results.append(
            {
                "clip_pitches": list(env.curriculum_clip.pitches if env.curriculum_clip else ()),
                "debug_return": total_reward,
                "native_reward_sum": native_reward_sum,
                "max_target_key_state": max_target,
                "max_unintended_key_state": max_unintended,
                "pressed_keys": sorted(pressed_keys),
                "final_info": dict(info),
            }
        )
    return {
        "deterministic": deterministic,
        "episodes": episode_results,
        "action_stats": _action_stats(all_actions),
        "mean_debug_return": float(np.mean([item["debug_return"] for item in episode_results])),
        "mean_max_target_key_state": float(
            np.mean([item["max_target_key_state"] for item in episode_results])
        ),
        "mean_max_unintended_key_state": float(
            np.mean([item["max_unintended_key_state"] for item in episode_results])
        ),
    }


def _action_stats(actions) -> dict:
    if not actions:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean_abs": 0.0,
            "saturation_fraction": 0.0,
        }
    arr = np.asarray(actions, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "saturation_fraction": float(np.mean(np.abs(arr) >= 0.95)),
    }


def parse_reward_config(raw: str | None) -> GeneralRewardConfig:
    if raw is None:
        return GeneralRewardConfig()
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(raw)
    return GeneralRewardConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--midi-min", type=int, default=73)
    parser.add_argument("--midi-max", type=int, default=75)
    parser.add_argument("--curriculum", default="single_notes")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = parse_reward_config(args.reward_config)
    env_kwargs = {
        "generated_midi_dir": str(output_dir / "generated_midi"),
        "curriculum": args.curriculum,
        "midi_min": args.midi_min,
        "midi_max": args.midi_max,
        "seed": args.seed,
        "note_count": 4,
        "lookahead": args.lookahead,
        "horizon_steps": args.horizon_steps,
        "reward_config": reward_config,
    }

    env = GeneralOneHandGoalEnv(**env_kwargs)
    obs, info = env.reset(seed=args.seed)
    print("env=GeneralOneHandGoalEnv")
    print(f"action_space={env.action_space}")
    print(f"observation_shape={env.observation_space.shape}")
    print(f"native_goal_shape={env.native_goal_shape}")
    print(f"lookahead={env.lookahead}")
    print(f"curriculum_clip_pitches={env.curriculum_clip.pitches if env.curriculum_clip else ()}")
    print(f"reset_target_keys={info['target_keys']}")
    if args.dry_run:
        return

    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=max(5000, args.timesteps),
        learning_starts=min(1000, max(100, args.timesteps // 5)),
        batch_size=64,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
    )
    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    runtime_seconds = time.time() - start

    run_name = (
        f"general_one_hand_sac_{args.curriculum}_midi{args.midi_min}_{args.midi_max}_"
        f"lookahead{args.lookahead}_seed{args.seed}_{args.timesteps}"
    )
    model_path = output_dir / run_name
    model.save(model_path)
    deterministic_eval = evaluate_policy(model, env_kwargs, deterministic=True)
    stochastic_eval = evaluate_policy(model, env_kwargs, deterministic=False)
    summary = {
        "model_path": str(model_path) + ".zip",
        "runtime_seconds": runtime_seconds,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "lookahead": args.lookahead,
        "midi_min": args.midi_min,
        "midi_max": args.midi_max,
        "curriculum": args.curriculum,
        "reward_config": reward_config.__dict__,
        "native_goal_shape": tuple(env.native_goal_shape),
        "observation_shape": env.observation_space.shape,
        "action_space": "Box(-1, 1, shape=(22,))",
        "deterministic_eval": deterministic_eval,
        "stochastic_eval": stochastic_eval,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"model_path={summary['model_path']}")
    print(f"summary_path={summary_path}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(
        "deterministic "
        f"mean_target={deterministic_eval['mean_max_target_key_state']:.6f} "
        f"mean_unintended={deterministic_eval['mean_max_unintended_key_state']:.6f} "
        f"mean_return={deterministic_eval['mean_debug_return']:.6f} "
        f"action_stats={deterministic_eval['action_stats']}"
    )
    print(
        "stochastic "
        f"mean_target={stochastic_eval['mean_max_target_key_state']:.6f} "
        f"mean_unintended={stochastic_eval['mean_max_unintended_key_state']:.6f} "
        f"mean_return={stochastic_eval['mean_debug_return']:.6f} "
        f"action_stats={stochastic_eval['action_stats']}"
    )


if __name__ == "__main__":
    main()
