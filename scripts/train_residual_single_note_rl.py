import argparse
import json
from pathlib import Path
import time

from stable_baselines3 import SAC

from ala_pianist.rl.residual_env import ResidualSingleNoteEnv, evaluate_residual_action


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "residual_single_note"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--reward-mode", choices=["target_travel_first", "cleanliness"], default="target_travel_first")
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = ResidualSingleNoteEnv(
        midi_path=ROOT / "tmp" / "residual_single_note_train.mid",
        horizon_steps=24,
        residual_scale=args.residual_scale,
        reward_mode=args.reward_mode,
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        learning_starts=min(1000, max(100, args.timesteps // 5)),
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.95,
        buffer_size=50_000,
    )
    base_metrics = evaluate_residual_action(
        residual_scale=args.residual_scale,
        reward_mode=args.reward_mode,
    )
    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    runtime = time.time() - start
    model_path = args.output_dir / f"residual_sac_{args.reward_mode}_scale_{args.residual_scale:g}"
    model.save(model_path)
    saved_model_path = model_path if model_path.exists() else model_path.with_suffix(model_path.suffix + ".zip")

    summary = {
        "timesteps": args.timesteps,
        "reward_mode": args.reward_mode,
        "residual_scale": args.residual_scale,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "model_path": str(saved_model_path),
        "base_action_metrics": base_metrics,
    }
    summary_path = args.output_dir / f"residual_sac_{args.reward_mode}_{args.timesteps}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"timesteps={args.timesteps}")
    print(f"reward_mode={args.reward_mode}")
    print(f"residual_scale={args.residual_scale}")
    print(f"runtime_seconds={runtime:.2f}")
    print(f"model_path={saved_model_path}")
    print(f"summary_path={summary_path}")
    print(f"base_action_metrics={base_metrics}")


if __name__ == "__main__":
    main()
