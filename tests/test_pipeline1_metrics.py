from ala_pianist.pipelines.pipeline1 import (
    _strict_outcome,
    _trajectory_quality,
    pipeline1_events_from_pitches,
)


def test_pipeline1_strict_outcomes_and_quality_flags():
    assert _strict_outcome(53, {53}, 1.0, 0.1) == "clean_low_unintended"
    assert _strict_outcome(53, {53}, 1.0, 0.7) == "clean_high_unintended"
    assert _strict_outcome(53, {53, 54}, 1.0, 1.0) == "dirty_pressed_wrong_key"
    assert _strict_outcome(53, set(), 0.3, 0.1) == "near_clean_partial"
    assert _strict_outcome(53, set(), 0.1, 0.1) == "missed"

    assert _trajectory_quality("clean_low_unintended", 0.1) == "gold_demo_candidate"
    assert _trajectory_quality("clean_high_unintended", 0.7) == "weak_demo_candidate"
    assert _trajectory_quality("clean_high_unintended", 0.9) == "not_demo_candidate"


def test_pipeline1_events_from_pitches():
    events = pipeline1_events_from_pitches([74, 75, 74, 75])
    assert [event.pitch for event in events] == [74, 75, 74, 75]
    assert [event.start for event in events] == [0.0, 0.4, 0.8, 1.2000000000000002]
