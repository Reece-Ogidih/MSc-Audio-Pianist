from pathlib import Path

from ala_pianist.learning.random_search import (
    run_single_note_random_search,
    write_single_note_learning_midi,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "single_note_random_search_dsharp5.mid"
SUMMARY_PATH = ROOT / "experiments" / "debug_learning" / "single_note_random_search.json"


def _print_result(label, result) -> None:
    distance = result.min_fingertip_target_distance
    distance_text = "None" if distance is None else f"{distance:.6f}"
    print(
        f"{label}: outcome={result.outcome} "
        f"debug_return={result.debug_return:.6f} "
        f"native_reward_sum={result.native_reward_sum:.6f} "
        f"max_target_state={result.max_target_key_state:.6f} "
        f"max_unintended_state={result.max_unintended_key_state:.6f} "
        f"min_distance_m={distance_text} "
        f"target_contact={result.any_target_contact} "
        f"any_key_contact={result.any_key_contact} "
        f"pressed={list(result.pressed_keys_seen)} "
        f"candidate={result.candidate_index}"
    )


def main() -> None:
    write_single_note_learning_midi(MIDI_PATH)
    summary = run_single_note_random_search(
        MIDI_PATH,
        candidate_count=30,
        horizon_steps=24,
        seed=7,
        output_path=SUMMARY_PATH,
    )

    print(f"midi_path={MIDI_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print(f"target_midi={summary.target_midi}")
    print(f"target_key={summary.target_key}")
    print(f"target_note={summary.target_note}")
    print(f"algorithm=seeded bounded random search over ramped open-loop 22D actions")
    print(f"seed={summary.seed}")
    print(f"candidate_count={summary.candidate_count}")
    print(f"horizon_steps={summary.horizon_steps}")
    _print_result("zero_baseline", summary.zero_result)
    _print_result("scripted_baseline", summary.scripted_result)
    _print_result("best_random_search", summary.best_result)
    improved = (
        summary.best_result.max_target_key_state
        > max(
            summary.zero_result.max_target_key_state,
            summary.scripted_result.max_target_key_state,
        )
    )
    print(f"target_travel_improved={improved}")


if __name__ == "__main__":
    main()
