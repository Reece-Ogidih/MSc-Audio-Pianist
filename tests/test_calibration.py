from ala_pianist.baselines.calibration import (
    CalibrationCandidate,
    TARGET_KEY,
    TARGET_MIDI,
    best_calibration_result,
    generate_single_note_candidates,
    run_single_note_calibration_sweep,
    write_single_note_calibration_midi,
)


def test_single_note_candidate_generator_is_bounded_and_deterministic():
    first = generate_single_note_candidates()
    second = generate_single_note_candidates()

    assert first == second
    assert len(first) == 12
    assert [candidate.index for candidate in first] == list(range(12))
    assert all(0.0 <= candidate.forearm_tx_fraction <= 1.0 for candidate in first)
    assert all(0.0 <= candidate.forearm_ty_fraction <= 1.0 for candidate in first)
    assert all(0.0 <= candidate.finger_flexion_fraction <= 1.0 for candidate in first)


def test_single_note_calibration_sweep_reports_diagnostics(tmp_path):
    midi_path = tmp_path / "single_note.mid"
    write_single_note_calibration_midi(midi_path)
    candidates = [
        CalibrationCandidate(
            index=0,
            forearm_tx_fraction=0.65,
            forearm_ty_fraction=0.70,
            finger_flexion_fraction=0.95,
        )
    ]

    results = run_single_note_calibration_sweep(
        midi_path,
        candidates=candidates,
        horizon_steps=8,
    )
    best = best_calibration_result(results)

    assert len(results) == 1
    result = results[0]
    assert result.target_midi == TARGET_MIDI
    assert result.target_key == TARGET_KEY
    assert result.target_note
    assert result.selected_finger == "little"
    assert result.max_target_key_state >= 0.0
    assert result.max_unintended_key_state >= 0.0
    assert result.min_fingertip_target_distance is None or (
        result.min_fingertip_target_distance >= 0.0
    )
    assert isinstance(result.any_target_contact, bool)
    assert isinstance(result.any_key_contact, bool)
    assert result.outcome in {
        "clean_target_press",
        "near_clean_partial_press",
        "target_press_with_unintended_keys",
        "contact_without_sufficient_travel",
        "no_contact_or_no_approach",
        "unstable_or_invalid",
    }
    assert best == result
