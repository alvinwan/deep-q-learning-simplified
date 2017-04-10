"""Check Q-learning is correct."""

from run_dqn import get_env
from run_dqn import simplified_learn

import tensorflow as tf
import pytest
import numpy as np


@pytest.fixture
def dqn_model():
    reader = tf.train.NewCheckpointReader('./tests/models/dqn/step-final.ckpt')
    return reader.get_tensor('q_func/action_value/fully_connected/weights')


def test_q_learning(dqn_model):
    envid = 'SpaceInvadersNoFrameskip-v3'
    learning_starts = 40000
    timesteps = 80000
    seed = 0
    batch_size = 32

    env = get_env(envid, seed)
    simp_model = simplified_learn(
        env,
        num_timesteps=timesteps,
        batch_size=batch_size,
        learning_starts=learning_starts)

    difference = np.linalg.norm(dqn_model - simp_model['W0'], ord='fro')
    print(difference)
    assert difference < 1e-2, difference