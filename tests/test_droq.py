import importlib.util
from pathlib import Path

import numpy as np
import torch

from ala_pianist.rl import (
    CriticEnsemble,
    DroQAgent,
    DroQConfig,
    DroQPolicy,
    ReplayBuffer,
    SquashedGaussianActor,
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
    checkpoint = agent.save(tmp_path / "droq.pt")

    loaded = DroQPolicy.load(checkpoint)
    action, _ = loaded.predict(np.zeros(8, dtype=np.float32), deterministic=True)

    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


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
