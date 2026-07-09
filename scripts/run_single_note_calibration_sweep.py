from collections import Counter
from pathlib import Path

from ala_pianist.baselines.calibration import (
    TARGET_KEY,
    TARGET_MIDI,
    TARGET_NOTE,
    best_calibration_result,
    generate_single_note_candidates,
    run_single_note_calibration_sweep,
    write_single_note_calibration_midi,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "single_note_calibration_dsharp5.mid"


def main() -> None:
    write_single_note_calibration_midi(MIDI_PATH)
    candidates = generate_single_note_candidates()
    results = run_single_note_calibration_sweep(MIDI_PATH, candidates=candidates)
    best = best_calibration_result(results)

    print(f"midi_path={MIDI_PATH}")
    print(f"target_midi={TARGET_MIDI}")
    print(f"target_key={TARGET_KEY}")
    print(f"target_note={TARGET_NOTE}")
    print("target_reason=prior scripted diagnostic produced partial key travel for D#5")
    print(f"candidate_count={len(candidates)}")
    print("candidates:")
    for result in results:
        distance = result.min_fingertip_target_distance
        distance_text = "None" if distance is None else f"{distance:.6f}"
        print(
            f"  idx={result.candidate.index:02d} "
            f"forearm_tx_frac={result.candidate.forearm_tx_fraction:.2f} "
            f"forearm_ty_frac={result.candidate.forearm_ty_fraction:.2f} "
            f"finger_flex_frac={result.candidate.finger_flexion_fraction:.2f} "
            f"max_target_state={result.max_target_key_state:.6f} "
            f"max_unintended_state={result.max_unintended_key_state:.6f} "
            f"min_distance_m={distance_text} "
            f"target_contact={result.any_target_contact} "
            f"any_key_contact={result.any_key_contact} "
            f"pressed_seen={list(result.pressed_keys_seen)} "
            f"best_reward={result.best_reward} "
            f"final_reward={result.final_reward} "
            f"outcome={result.outcome}"
        )
        if result.contact_pairs:
            print(f"    contact_pairs={list(result.contact_pairs[:4])}")

    counts = Counter(result.outcome for result in results)
    print("summary:")
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count}")
    print(
        "best_result="
        f"idx={best.candidate.index} "
        f"outcome={best.outcome} "
        f"max_target_state={best.max_target_key_state:.6f} "
        f"max_unintended_state={best.max_unintended_key_state:.6f} "
        f"pressed_seen={list(best.pressed_keys_seen)}"
    )


if __name__ == "__main__":
    main()
