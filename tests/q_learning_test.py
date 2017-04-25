"""Check Q-learning is correct."""

from run_dqn import get_env
from run_dqn import simplified_learn

import os.path
import tensorflow as tf
import pytest
import numpy as np


@pytest.fixture
def dqn_model_fc():
    reader = tf.train.NewCheckpointReader('./tests/models/dqn/step-final.ckpt')
    return reader.get_tensor('q_func/action_value/fully_connected/weights')


def test_q_learning(dqn_model_fc):
    envid = 'SpaceInvadersNoFrameskip-v3'
    learning_starts = 40000
    timesteps = 80000
    seed = 0
    batch_size = 32
    restore = './tests/models/dqn/init.ckpt'

    if os.path.exists('q_learning_test_q0.npy'):
        pass

    env = get_env(envid, seed)
    simp_model = simplified_learn(
        env,
        num_timesteps=timesteps,
        batch_size=batch_size,
        learning_starts=learning_starts,
        restore=restore)
    simp_model_fc = simp_model['W0']

    entries = np.prod(dqn_model_fc.shape)
    avg_difference = np.sum(dqn_model_fc - simp_model['W0']) / entries
    max_difference = np.linalg.norm(dqn_model_fc - simp_model_fc, ord=np.inf)
    np.save('q_learning_test_q0.npy', simp_model_fc)
    assert max_difference < 1e-6, (max_difference, avg_difference)