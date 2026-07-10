import argparse
import json
from pathlib import Path
import time

from stable_baselines3 import PPO, SAC, TD3

from ala_pianist.rl import KeysetPianoGymEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "keyset_rl"


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=sorted(ALGOS), default="SAC")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--horizon-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = KeysetPianoGymEnv(
        midi_dir=ROOT / "tmp" / "keyset_rl_train",
        horizon_steps=args.horizon_steps,
    )
    algo_cls = ALGOS[args.algo]
    kwargs = {"seed": args.seed, "verbose": 0}
    if args.algo == "SAC":
        kwargs.update(
            learning_starts=min(1000, max(100, args.timesteps // 5)),
            batch_size=64,
            train_freq=1,
            gradient_steps=1,
            learning_rate=3e-4,
            gamma=0.95,
            buffer_size=50_000,
        )
    elif args.algo == "TD3":
        kwargs.update(
            learning_starts=min(1000, max(100, args.timesteps // 5)),
            batch_size=64,
            train_freq=1,
            gradient_steps=1,
            learning_rate=3e-4,
            gamma=0.95,
            buffer_size=50_000,
        )
    else:
        kwargs.update(n_steps=64, batch_size=32, n_epochs=4, learning_rate=3e-4, gamma=0.95)

    model = algo_cls("MlpPolicy", env, **kwargs)
    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    runtime = time.time() - start

    model_path = OUT_DIR / f"keyset_{args.algo.lower()}_model"
    model.save(model_path)
    summary = {
        "algo": args.algo,
        "timesteps": args.timesteps,
        "horizon_steps": args.horizon_steps,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "model_path": str(model_path) + ".zip",
    }
    summary_path = OUT_DIR / f"keyset_{args.algo.lower()}_{args.timesteps}_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"algo={args.algo}")
    print(f"timesteps={args.timesteps}")
    print(f"horizon_steps={args.horizon_steps}")
    print(f"seed={args.seed}")
    print(f"runtime_seconds={runtime:.2f}")
    print(f"model_path={model_path}.zip")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
