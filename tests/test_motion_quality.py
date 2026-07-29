from __future__ import annotations

import numpy as np
import pandas as pd

from ala_pianist.evaluation.motion_quality import (
    ACTION_NAMES,
    action_dimension_mapping,
    action_quality,
    annotate_rollout_frame,
    fingertip_motion_quality,
    phase_labels,
    saturation_fraction,
)


def test_action_finite_difference_metrics_are_deterministic():
    actions = np.asarray(
        [
            [0.0, 0.0],
            [0.5, -0.5],
            [1.0, -1.0],
        ],
        dtype=float,
    )

    metrics = action_quality(actions)

    assert metrics["mean_abs_action_delta"] == 0.5
    assert metrics["mean_squared_action_delta"] == 0.25
    assert metrics["max_abs_action_delta"] == 0.5
    assert metrics["action_saturation_fraction"] == 2 / 6


def test_phase_segmentation_uses_existing_target_markers():
    trace = pd.DataFrame(
        {
            "intended_key_indices": ["[]", "[52]", "[52]", "[]", "[54]", "[54]", "[]"],
        }
    )

    phases = list(phase_labels(trace))

    assert phases == [
        "approach_first_note",
        "first_note_hold",
        "first_note_hold",
        "release_transition",
        "second_note_hold",
        "second_note_hold",
        "terminal_release",
    ]


def test_action_mapping_has_22_public_dimensions_and_no_sustain():
    mapping = action_dimension_mapping()

    assert len(mapping) == 22
    assert ACTION_NAMES[-2:] == ("forearm_tx", "forearm_ty")
    assert all("sustain" not in item.name.lower() for item in mapping)


def test_saturation_metric_counts_near_bounds():
    values = np.asarray([-1.0, -0.94, 0.0, 0.95, 0.2])

    assert saturation_fraction(values, threshold=0.95) == 2 / 5


def test_fingertip_motion_finite_differences():
    trace = pd.DataFrame(
        {
            "fingertip_ffdistal_site_x": [0.0, 0.1, 0.2],
            "fingertip_ffdistal_site_y": [0.0, 0.0, 0.0],
            "fingertip_ffdistal_site_z": [0.0, 0.0, 0.0],
        }
    )

    metrics = fingertip_motion_quality(trace, dt=0.1)

    assert np.isclose(metrics["mean_fingertip_displacement"], 0.1)
    assert np.isclose(metrics["mean_fingertip_speed"], 1.0)
    assert np.isclose(metrics["max_fingertip_acceleration"], 0.0)


def test_renderer_annotation_changes_pixels_and_preserves_shape():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    annotated = annotate_rollout_frame(
        frame,
        sequence_id="trained_73_72",
        simulation_time=0.1,
        intended_midi=[73],
        intended_keys=[52],
        pressed_midi=[73, 63],
        wrong_pressed_midi=[63],
        phase="first_note_hold",
        failure_labels=["previous_target_not_released"],
        target_activation=1.0,
        max_unintended=0.5,
    )

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > 0


def test_motion_helpers_do_not_mutate_trace_dataframe():
    trace = pd.DataFrame(
        {
            "intended_key_indices": ["[52]", "[]"],
            "action_00": [0.0, 1.0],
        }
    )
    before = trace.copy(deep=True)

    _ = phase_labels(trace)

    pd.testing.assert_frame_equal(trace, before)

