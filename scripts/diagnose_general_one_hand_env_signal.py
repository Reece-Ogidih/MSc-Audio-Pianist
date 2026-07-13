import argparse
import json
from pathlib import Path

from ala_pianist.rl import GeneralOneHandGoalEnv
from ala_pianist.rl.general_env_diagnostics import (
    D_SHARP_5_KEY,
    DEFAULT_DSHARP5_RESIDUAL_MODEL,
    goal_timing_diagnostic,
    run_dsharp5_residual_policy_diagnostic,
    run_known_dsharp5_base_diagnostic,
    run_random_action_diagnostics,
    run_zero_action_diagnostic,
    write_dsharp5_diagnostic_midi,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand_signal"


def make_env(midi_path: Path, *, midi_min: int, midi_max: int, lookahead: int, horizon_steps: int):
    return GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=midi_min,
        midi_max=midi_max,
        lookahead=lookahead,
        horizon_steps=horizon_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--midi-min", type=int, default=73)
    parser.add_argument("--midi-max", type=int, default=75)
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--dsharp-residual-model-path", default=str(DEFAULT_DSHARP5_RESIDUAL_MODEL))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = write_dsharp5_diagnostic_midi(
        output_dir / "diagnostic_dsharp5.mid",
        midi_min=args.midi_min,
        midi_max=args.midi_max,
    )

    timing_env = make_env(
        midi_path,
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        lookahead=args.lookahead,
        horizon_steps=args.horizon_steps,
    )
    goal = goal_timing_diagnostic(timing_env)

    zero = run_zero_action_diagnostic(
        make_env(
            midi_path,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            lookahead=args.lookahead,
            horizon_steps=args.horizon_steps,
        ),
        horizon_steps=args.horizon_steps,
    )
    random_results = run_random_action_diagnostics(
        midi_path,
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        lookahead=args.lookahead,
        seed=args.seed,
        horizon_steps=args.horizon_steps,
    )
    known_ramped = run_known_dsharp5_base_diagnostic(
        make_env(
            midi_path,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            lookahead=args.lookahead,
            horizon_steps=args.horizon_steps,
        ),
        horizon_steps=args.horizon_steps,
        ramped=True,
    )
    known_held = run_known_dsharp5_base_diagnostic(
        make_env(
            midi_path,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            lookahead=args.lookahead,
            horizon_steps=args.horizon_steps,
        ),
        horizon_steps=args.horizon_steps,
        ramped=False,
    )
    residual_policy = run_dsharp5_residual_policy_diagnostic(
        make_env(
            midi_path,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            lookahead=args.lookahead,
            horizon_steps=args.horizon_steps,
        ),
        model_path=args.dsharp_residual_model_path,
        horizon_steps=args.horizon_steps,
    )

    summary = {
        "midi_path": str(midi_path),
        "target_midi": 75,
        "target_key": D_SHARP_5_KEY,
        "lookahead": args.lookahead,
        "midi_min": args.midi_min,
        "midi_max": args.midi_max,
        "horizon_steps": args.horizon_steps,
        "goal_timing": goal,
        "zero_action": zero.to_dict(),
        "random_actions": [result.to_dict() for result in random_results],
        "known_dsharp5_base_ramped": known_ramped.to_dict(),
        "known_dsharp5_base_held": known_held.to_dict(),
        "known_dsharp5_residual_policy": residual_policy.to_dict(),
        "action_semantics": {
            "general_env": "applies one normalized 22D action directly on every env.step call",
            "residual_and_search_baselines": "commonly ramp and then hold native actions with action_ramp(step)",
            "meaningful_difference": True,
            "recommendation": "consider an action-repeat/ramp/hold wrapper or policy output smoothing before longer training if known ramped actions outperform direct stepwise actions",
        },
    }
    summary_path = output_dir / "general_one_hand_signal_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={summary_path}")
    print(f"midi_path={midi_path}")
    print(f"native_goal_shape={goal['native_goal_shape']}")
    print(f"observation_shape={goal['observation_shape']}")
    print(f"target_key_seen_in_goal={goal['target_key_seen_in_goal']}")
    print(f"target_key_goal_steps={goal['target_key_goal_steps']}")
    _print_rollout(zero)
    for result in random_results:
        _print_rollout(result)
    _print_rollout(known_ramped)
    _print_rollout(known_held)
    _print_rollout(residual_policy)
    print("action_semantics=general env applies direct stepwise actions; residual/search diagnostics usually ramp/hold actions")


def _print_rollout(result) -> None:
    if result.skipped_reason:
        print(f"{result.name}: skipped={result.skipped_reason}")
        return
    print(
        f"{result.name}: "
        f"max_target={result.max_target_key_state:.6f} "
        f"max_unintended={result.max_unintended_key_state:.6f} "
        f"pressed={list(result.pressed_keys)} "
        f"shaped_return={result.shaped_return:.6f} "
        f"native_reward_sum={result.native_reward_sum:.6f} "
        f"positive_target_signal={result.positive_target_signal} "
        f"final_reward_breakdown={result.final_reward_breakdown}"
    )


if __name__ == "__main__":
    main()
