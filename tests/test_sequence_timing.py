from ala_pianist.music import (
    assign_right_hand_fingering,
    generate_curriculum_events,
    generate_sequence_events,
    note_windows,
    sequence_timing_from_profile,
)


def _event_tuple(event):
    return (
        event.pitch,
        event.start,
        event.duration,
        event.velocity,
        event.fingering,
    )


def test_aligned_training_and_evaluation_sequence_events_match():
    timing = sequence_timing_from_profile("aligned")
    training_events = generate_curriculum_events(
        mode="sequence_cleanup",
        midi_min=73,
        midi_max=75,
        sequence_pitches=((73, 75),),
        clip_index=0,
        note_duration=timing.note_duration,
        gap=timing.note_gap,
        velocity=timing.velocity,
    )
    evaluation_events = generate_sequence_events(
        (73, 75),
        midi_min=73,
        midi_max=75,
        timing=timing,
        fingering_fn=assign_right_hand_fingering,
    )

    assert tuple(map(_event_tuple, training_events)) == tuple(map(_event_tuple, evaluation_events))


def test_aligned_timing_schedule_maps_pitch_to_key_index():
    timing = sequence_timing_from_profile("aligned")
    windows = note_windows((73, 75), timing=timing)

    assert windows[0]["pitch"] == 73
    assert windows[0]["key_index"] == 52
    assert windows[0]["duration"] == 0.28
    assert windows[1]["pitch"] == 75
    assert windows[1]["key_index"] == 54
    assert windows[1]["start_seconds"] == 0.40


def test_legacy_curriculum_timing_differs_from_aligned():
    aligned = sequence_timing_from_profile("aligned")
    legacy = sequence_timing_from_profile("legacy_curriculum")

    assert aligned.note_duration == 0.28
    assert aligned.note_gap == 0.12
    assert legacy.note_duration == 0.45
    assert legacy.note_gap == 0.05
