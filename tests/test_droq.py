import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from ala_pianist.rl import (
    CriticEnsemble,
    DroQAgent,
    DroQConfig,
    DroQPolicy,
    ReplayBuffer,
    SquashedGaussianActor,
    load_droq_checkpoint,
    replay_buffer_from_checkpoint,
)


def _small_config() -> DroQConfig:
    return DroQConfig(
        observation_dim=8,
        action_dim=3,
        hidden_dim=16,
        critic_ensemble_size=2,
        batch_size=4,
        buffer_size=32,
        utd_ratio=2,
        device="cpu",
    )


def _training_script_module():
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/train_droq_general_one_hand_policy.py")
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("train_droq_general_one_hand_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_droq_actor_output_shape_and_range():
    actor = SquashedGaussianActor(observation_dim=8, action_dim=3, hidden_dim=16)
    observation = torch.zeros(5, 8)

    action, log_prob = actor.sample(observation)
    deterministic = actor.deterministic(observation)

    assert action.shape == (5, 3)
    assert deterministic.shape == (5, 3)
    assert log_prob.shape == (5, 1)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_droq_critic_ensemble_forward_shape():
    critics = CriticEnsemble(
        observation_dim=8,
        action_dim=3,
        hidden_dim=16,
        ensemble_size=2,
        dropout=0.1,
    )

    q_values = critics(torch.zeros(4, 8), torch.zeros(4, 3))

    assert q_values.shape == (2, 4, 1)
    assert torch.isfinite(q_values).all()


def test_replay_buffer_add_and_sample():
    buffer = ReplayBuffer(observation_dim=8, action_dim=3, capacity=16)
    for index in range(6):
        buffer.add(
            np.full(8, index, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            float(index),
            np.full(8, index + 1, dtype=np.float32),
            done=False,
        )

    batch = buffer.sample(4, device=torch.device("cpu"))

    assert buffer.size == 6
    assert batch["observations"].shape == (4, 8)
    assert batch["actions"].shape == (4, 3)
    assert batch["rewards"].shape == (4, 1)


def test_droq_one_update_has_finite_losses():
    agent = DroQAgent(_small_config())
    buffer = ReplayBuffer(observation_dim=8, action_dim=3, capacity=32)
    rng = np.random.default_rng(3)
    for _ in range(12):
        buffer.add(
            rng.normal(size=8).astype(np.float32),
            rng.uniform(-1.0, 1.0, size=3).astype(np.float32),
            float(rng.normal()),
            rng.normal(size=8).astype(np.float32),
            done=False,
        )

    losses = agent.update(buffer.sample(4, device=agent.device))

    assert {"critic_loss", "actor_loss", "alpha_loss", "alpha", "mean_q"}.issubset(losses)
    assert all(np.isfinite(value) for value in losses.values())


def test_droq_checkpoint_save_load_and_predict(tmp_path):
    agent = DroQAgent(_small_config())
    buffer = ReplayBuffer(observation_dim=8, action_dim=3, capacity=16)
    buffer.add(
        np.zeros(8, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        1.0,
        np.ones(8, dtype=np.float32),
        done=False,
    )
    checkpoint = agent.save(tmp_path / "droq.pt", replay_buffer=buffer, extra={"step": 12})

    loaded = DroQPolicy.load(checkpoint)
    payload = load_droq_checkpoint(checkpoint)
    action, _ = loaded.predict(np.zeros(8, dtype=np.float32), deterministic=True)

    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert payload["extra"]["step"] == 12
    assert payload["replay_buffer"]["size"] == 1
    assert "rng_state" in payload


def test_replay_buffer_restores_from_checkpoint_payload(tmp_path):
    agent = DroQAgent(_small_config())
    buffer = ReplayBuffer(observation_dim=8, action_dim=3, capacity=16)
    for index in range(5):
        buffer.add(
            np.full(8, index, dtype=np.float32),
            np.full(3, index, dtype=np.float32),
            float(index),
            np.full(8, index + 1, dtype=np.float32),
            done=index == 4,
        )
    checkpoint = agent.save(tmp_path / "buffer.pt", replay_buffer=buffer, extra={"step": 5})
    payload = load_droq_checkpoint(checkpoint)

    restored = replay_buffer_from_checkpoint(
        payload,
        observation_dim=8,
        action_dim=3,
        fallback_capacity=32,
    )
    reset = ReplayBuffer(observation_dim=8, action_dim=3, capacity=32)

    assert restored.size == 5
    assert restored.capacity == 16
    assert reset.size == 0


def test_droq_load_can_reset_optimizers(tmp_path):
    agent = DroQAgent(_small_config())
    buffer = ReplayBuffer(observation_dim=8, action_dim=3, capacity=32)
    rng = np.random.default_rng(4)
    for _ in range(8):
        buffer.add(
            rng.normal(size=8).astype(np.float32),
            rng.uniform(-1.0, 1.0, size=3).astype(np.float32),
            float(rng.normal()),
            rng.normal(size=8).astype(np.float32),
            done=False,
        )
    agent.update(buffer.sample(4, device=agent.device))
    checkpoint = agent.save(tmp_path / "optimizer.pt")

    restored = DroQAgent.load(checkpoint)
    reset = DroQAgent.load(checkpoint, reset_optimizers=True)

    assert agent.actor_optimizer.state_dict()["state"]
    assert restored.actor_optimizer.state_dict()["state"]
    assert reset.actor_optimizer.state_dict()["state"] == {}


def test_evaluator_can_load_droq_checkpoint(tmp_path):
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/evaluate_general_one_hand_policy.py")
    spec = importlib.util.spec_from_file_location("evaluate_general_one_hand_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    agent = DroQAgent(_small_config())
    checkpoint = agent.save(tmp_path / "eval_droq.pt")
    policy = module.load_policy(checkpoint, model_kind="droq")
    action, _ = policy.predict(np.zeros(8, dtype=np.float32), deterministic=True)

    assert action.shape == (3,)


def test_droq_resume_cli_helpers_and_checkpoint_numbering():
    module = _training_script_module()

    assert module._checkpoint_step({"extra": {"step": 5000}}) == 5000
    assert module._checkpoint_step({}) == 0
    assert module._safe_stage_name("resume") == "resume"
    assert module._safe_stage_name(None) == "droq_general_one_hand"


def test_droq_resume_cli_validation_rejects_ambiguous_timesteps():
    module = _training_script_module()
    args = type(
        "Args",
        (),
        {
            "resume_checkpoint": "checkpoint.pt",
            "additional_timesteps": 500,
            "timesteps": 100,
        },
    )()

    try:
        module.validate_training_mode_args(args)
    except ValueError as exc:
        assert "--additional-timesteps" in str(exc)
    else:
        raise AssertionError("Expected ambiguous resume/timesteps args to raise ValueError.")


def test_droq_from_scratch_cli_validation_sets_default_timesteps():
    module = _training_script_module()
    args = type(
        "Args",
        (),
        {
            "resume_checkpoint": None,
            "additional_timesteps": None,
            "timesteps": None,
        },
    )()

    module.validate_training_mode_args(args)

    assert args.timesteps == 5000


def test_droq_cuda_request_fails_clearly_when_unavailable():
    module = _training_script_module()
    if torch.cuda.is_available():
        module.validate_device("cuda")
        return

    try:
        module.validate_device("cuda")
    except RuntimeError as exc:
        assert "Refusing to silently fall back to CPU" in str(exc)
    else:
        raise AssertionError("Expected unavailable CUDA request to raise RuntimeError.")


def test_droq_resume_shape_mismatch_raises_clear_error():
    module = _training_script_module()
    config = DroQConfig(observation_dim=8, action_dim=3)
    env = type("Env", (), {"native_goal_shape": (178,)})()
    args = type(
        "Args",
        (),
        {
            "lookahead": 1,
            "action_mode": "direct",
            "action_repeat": 1,
            "reward_profile": "transition_cleanup",
            "sequence_timing_profile": "aligned",
        },
    )()

    try:
        module._validate_resume_config(
            payload={"config": {"observation_dim": 9, "action_dim": 3}, "extra": {}},
            config=config,
            env=env,
            args=args,
        )
    except ValueError as exc:
        assert "observation_dim" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise ValueError.")
