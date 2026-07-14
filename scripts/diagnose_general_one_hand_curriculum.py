import argparse
from collections import Counter
import json
from pathlib import Path

from ala_pianist.rl import GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand_curriculum"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--midi-min", type=int, default=73)
    parser.add_argument("--midi-max", type=int, default=75)
    parser.add_argument("--curriculum", default="single_notes")
    parser.add_argument("--num-resets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=output_dir / "generated_midi",
        curriculum=args.curriculum,
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        seed=args.seed,
        lookahead=args.lookahead,
        horizon_steps=4,
    )

    records = []
    pitch_counts: Counter[int] = Counter()
    key_counts: Counter[int] = Counter()
    mismatches = []
    for reset_index in range(args.num_resets):
        _, info = env.reset(seed=args.seed if reset_index == 0 else None)
        sampled = info["sampled_midi_pitch"]
        target_keys = tuple(info["target_keys"])
        expected_key = None if sampled is None else int(sampled) - 21
        pitch_counts.update([sampled])
        key_counts.update(target_keys)
        match = expected_key in target_keys
        record = {
            "reset_index": reset_index,
            "sampled_midi_pitch": sampled,
            "expected_key": expected_key,
            "target_keys": target_keys,
            "goal_matches_sample": match,
            "native_goal_shape": info["native_goal_shape"],
            "curriculum_pitches": info["curriculum_pitches"],
        }
        records.append(record)
        if not match:
            mismatches.append(record)

    summary = {
        "curriculum": args.curriculum,
        "midi_min": args.midi_min,
        "midi_max": args.midi_max,
        "lookahead": args.lookahead,
        "num_resets": args.num_resets,
        "pitch_counts": {str(key): value for key, value in sorted(pitch_counts.items())},
        "target_key_counts": {str(key): value for key, value in sorted(key_counts.items())},
        "native_goal_shape": tuple(env.native_goal_shape),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "records": records,
    }
    summary_path = output_dir / "general_one_hand_curriculum_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"summary_path={summary_path}")
    print(f"curriculum={args.curriculum}")
    print(f"midi_range={args.midi_min}-{args.midi_max}")
    print(f"num_resets={args.num_resets}")
    print(f"native_goal_shape={tuple(env.native_goal_shape)}")
    print(f"pitch_counts={dict(sorted(pitch_counts.items()))}")
    print(f"target_key_counts={dict(sorted(key_counts.items()))}")
    print(f"mismatch_count={len(mismatches)}")
    if mismatches:
        print(f"first_mismatch={mismatches[0]}")


if __name__ == "__main__":
    main()
