import importlib.util
from pathlib import Path

import numpy as np

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics


def _evaluation_script_module():
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/evaluate_general_one_hand_policy.py")
    spec = importlib.util.spec_from_file_location("evaluate_general_one_hand_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pressed_key_metrics_perfect_match():
    metrics = pressed_key_metrics({52, 54}, {52, 54})

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_pressed_key_metrics_extra_wrong_keys_reduce_precision_and_f1():
    metrics = pressed_key_metrics({52, 54}, {52, 53, 54, 55})

    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert np.isclose(metrics.f1, 2.0 / 3.0)
    assert metrics.false_positives == 2


def test_pressed_key_metrics_missing_targets_reduce_recall_and_f1():
    metrics = pressed_key_metrics({52, 54}, {52})

    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert np.isclose(metrics.f1, 2.0 / 3.0)
    assert metrics.false_negatives == 1


def test_pressed_key_metrics_no_pressed_keys_with_targets_is_zero():
    metrics = pressed_key_metrics({52}, set())

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_pressed_key_metrics_no_targets_and_no_presses_is_perfect_noop():
    metrics = pressed_key_metrics(set(), set())

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_timestep_key_metrics_on_toy_vectors():
    target = [
        binary_key_vector({52}, n_keys=56),
        binary_key_vector({54}, n_keys=56),
        binary_key_vector(set(), n_keys=56),
    ]
    pressed = [
        binary_key_vector({52, 53}, n_keys=56),
        binary_key_vector(set(), n_keys=56),
        binary_key_vector({55}, n_keys=56),
    ]

    metrics = timestep_key_metrics(target, pressed)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 1
    assert np.isclose(metrics.precision, 1.0 / 3.0)
    assert metrics.recall == 0.5
    assert np.isclose(metrics.f1, 0.4)


def test_evaluation_script_metric_fields_are_added():
    module = _evaluation_script_module()
    result = module._with_key_metrics(
        {"pressed_keys": [52, 53], "target_recall": 1.0},
        target_keys={52},
        pressed_keys={52, 53},
        target_vectors=[binary_key_vector({52})],
        pressed_vectors=[binary_key_vector({52, 53})],
    )

    assert result["pressed_key_precision"] == 0.5
    assert result["pressed_key_recall"] == 1.0
    assert np.isclose(result["pressed_key_f1"], 2.0 / 3.0)
    assert result["timestep_precision"] == 0.5
    assert result["timestep_recall"] == 1.0
    assert np.isclose(result["timestep_f1"], 2.0 / 3.0)
