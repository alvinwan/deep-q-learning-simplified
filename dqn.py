import itertools
import sys
import time

import numpy as np
import gym.spaces

from typing import Callable

from dqn_utils import get_wrapper_by_name
from dqn_utils import LinearSchedule
from dqn_utils import ReplayBuffer

np.random.seed(1)


def learn(env,
          q_func,
          train_func,
          initialize_model,
          batch_size=32,
          exploration=LinearSchedule(1000000, 0.1),
          frame_history_len: int=4,
          learning_starts=50000,
          learning_freq=4,
          replay_buffer_size: int=1000000,
          start_time=time.time(),
          stopping_criterion: Callable[[int], bool]=None,
          target_update_freq=10000):
    """Train a two-layer neural network.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Structured after github.com/alvinwan/deep-q-learning

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    start_time: datetime
        The time of training start
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    def update_target_func():
        pass

    ###########
    # RUN ENV #
    ###########

    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    learning_rate = exploration.value(0)

    for t in itertools.count():

        # 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(t):
            break

        # 2. Step the env and store the transition
        t_obs_idx = replay_buffer.store_frame(last_obs)

        if np.random.random() < exploration.value(t) \
                or not model_initialized \
                or not replay_buffer.can_sample(batch_size):
            action = env.action_space.sample()
        else:
            r_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
            curr_q_eval = q_func(r_obs, num_actions)
            action = np.argmax(curr_q_eval)

        last_obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(t_obs_idx, action, reward, done)

        if done:
            last_obs = env.reset()

        # 3. Perform experience relay and train the network.
        if (t > learning_starts \
            and t % learning_freq == 0 \
            and replay_buffer.can_sample(batch_size)):

            obs_t, act_t, rew_t, obs_tp1, done_mask = \
                replay_buffer.sample(batch_size)

            if not model_initialized:
                model_initialized = True
                model = initialize_model(num_actions)

            learning_rate = exploration.value(t)
            model = train_func(
                obs_t, act_t, rew_t, obs_tp1, done_mask, learning_rate)

            if t % target_update_freq == 0:
                update_target_func()
                num_param_updates += 1

        # 4. Log progress
        episode_rewards = get_wrapper_by_name(env,
                                              "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward,
                                           mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Time %s s" % int(time.time() - start_time))
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % learning_rate)
            sys.stdout.flush()