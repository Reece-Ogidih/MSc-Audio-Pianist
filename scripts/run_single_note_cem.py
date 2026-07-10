from pathlib import Path

from ala_pianist.learning.cem import run_single_note_cem, write_single_note_cem_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "single_note_cem_dsharp5.mid"
SUMMARY_PATH = ROOT / "experiments" / "cem_single_note" / "single_note_cem_summary.json"


def _print_result(label, result) -> None:
    print(
        f"{label}: outcome={result.outcome} "
        f"debug_return={result.debug_return:.6f} "
        f"native_reward_sum={result.native_reward_sum:.6f} "
        f"max_target_state={result.max_target_key_state:.6f} "
        f"max_unintended_state={result.max_unintended_key_state:.6f} "
        f"pressed={list(result.pressed_keys_seen)} "
        f"wrong_pressed={result.wrong_pressed_key_count} "
        f"target_contact={result.any_target_contact} "
        f"any_key_contact={result.any_key_contact}"
    )


def main() -> None:
    write_single_note_cem_midi(MIDI_PATH)
    summary = run_single_note_cem(
        MIDI_PATH,
        iterations=4,
        candidate_count=20,
        elite_fraction=0.25,
        horizon_steps=24,
        seed=13,
        random_search_seed=7,
        random_search_candidates=30,
        output_path=SUMMARY_PATH,
    )

    print(f"midi_path={MIDI_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print(f"target_midi={summary.target_midi}")
    print(f"target_key={summary.target_key}")
    print(f"target_note={summary.target_note}")
    print("algorithm=CEM over native public 22D open-loop ramped action targets")
    print(f"seed={summary.seed}")
    print(f"iterations={summary.iterations}")
    print(f"candidate_count={summary.candidate_count}")
    print(f"elite_fraction={summary.elite_fraction}")
    print(f"horizon_steps={summary.horizon_steps}")
    _print_result("zero_baseline", summary.zero_result)
    _print_result("scripted_baseline", summary.scripted_result)
    _print_result("random_search_baseline", summary.random_search_result)
    _print_result("best_cem", summary.best_result.metrics)
    print("iterations_summary:")
    for item in summary.iteration_summaries:
        print(
            f"  iter={item.iteration} best_outcome={item.best_outcome} "
            f"best_target={item.best_target_key_state:.6f} "
            f"best_unintended={item.best_unintended_key_state:.6f} "
            f"best_debug_return={item.best_debug_return:.6f} "
            f"mean_action_abs={item.mean_action_abs:.6f} "
            f"std_action_mean={item.std_action_mean:.6f}"
        )
    best_action = summary.best_result.action
    print(
        "best_action_summary="
        f"min={min(best_action):.6f} max={max(best_action):.6f} "
        f"mean_abs={sum(abs(v) for v in best_action) / len(best_action):.6f}"
    )
    beats_random = (
        summary.best_result.metrics.max_target_key_state
        > summary.random_search_result.max_target_key_state
        and summary.best_result.metrics.max_unintended_key_state
        <= summary.random_search_result.max_unintended_key_state + 0.05
    )
    print(f"beats_random_cleaner_or_stronger={beats_random}")


if __name__ == "__main__":
    main()
