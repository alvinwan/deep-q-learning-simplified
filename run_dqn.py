"""

Usage:
    run_dqn.py [options]

Options:
    --batch-size=<size>     Batch size [default: 32]
    --envid=<envid>         Environment id [default: SpaceInvadersNoFrameskip-v3]
    --timesteps=<steps>     Number of timesteps to run [default: 40000000]
"""

import dqn

from atari_wrappers import wrap_deepmind
from dqn_utils import get_wrapper_by_name
from dqn_utils import PiecewiseSchedule

import docopt
import gym
import numpy as np

from ad import adnumber
from gym import wrappers
from skimage.transform import pyramid_reduce

import os.path
import random


def initialize_model(num_actions: int):
    """Initialize the model"""
    return {'W0': adnumber(np.random.random((num_actions, 1)))}


def evaluate(X: np.ndarray, model) -> np.ndarray:
    """Evaluate the neural network."""
    l0, W0 = X, model['W0']
    l1 = np.array([sigmoid(l0.dot(w)) for w in W0])
    return l1


def featurize(X: np.ndarray, target=(84, 84, 4)) -> np.ndarray:
    """Featurize the provided data.

    Simple image compression, for now.
    """
    return pyramid_reduce(X, scale=X.shape[0]/target[0])


def sigmoid(x):
    """Sigmoid activation function."""
    return 1/(1+np.exp(-x))


def set_global_seeds(i):
    """Set global random seeds."""
    np.random.seed(i)
    random.seed(i)


def get_env(env_id, seed):
    """Get gym environment, per id and seed."""
    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env


def simplified_learn(env,
                num_timesteps,
                batch_size=32):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
             (0,                   1e-4 * lr_multiplier),
             (num_iterations / 10, 1e-4 * lr_multiplier),
             (num_iterations / 2,  5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier)

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        initialize_model=initialize_model,
        q_func=evaluate,
        lr_schedule=lr_schedule,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=batch_size,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000
    )
    env.close()


def main():
    arguments = docopt.docopt(__doc__)

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(arguments['--envid'], seed)

    batch_size = int(arguments['--batch-size'])
    simplified_learn(
        env,
        num_timesteps=int(arguments['--timesteps']),
        batch_size=batch_size)


if __name__ == '__main__':
    main()