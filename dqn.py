import itertools
import sys
import time

from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import gym.spaces

from gym import wrappers

from dqn_utils import get_wrapper_by_name
from dqn_utils import LinearSchedule
from dqn_utils import one_hot
from dqn_utils import ReplayBuffer

np.random.seed(1)


def learn(env,
          q_func,
          initialize_model: Callable[[Tuple, int], Dict],
          batch_size=32,
          exploration=LinearSchedule(1000000, 0.1),
          frame_history_len: int=4,
          gamma: float=0.99,
          learning_starts=50000,
          lr_schedule=LinearSchedule(1000000, 0.1),
          learning_freq=4,
          replay_buffer_size: int=1000000,
          start_time=time.time(),
          stopping_criterion: Callable[[wrappers.Monitor, int], bool]=None,
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
    lr_schedule: rl_algs.deepq.utils.schedules.Schedule
        schedule for learning rate.
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

    def update_target_func(model_curr: Dict, model_target: Dict):
        model_curr.update(model_target)

    def train_func(
            obs_t: np.ndarray,
            act_t: np.ndarray,
            rew_t: np.ndarray,
            obs_tp1: np.ndarray,
            done_mask: np.ndarray,
            learning_rate: float,
            model_curr: Dict,
            model_target: Dict) -> Dict:
        """Train function, minimizing loss per q-learning objective.

        This assumes the q_function is a one-layer fc neural network, where the
        loss function is squared error.
        """

        curr_q = q_func(obs_t, model_curr)
        target_q = q_func(obs_tp1, model_target)
        actions = one_hot(act_t, num_actions)
        q_target_max = np.max(target_q, axis=1)
        q_target_val = rew_t + gamma * (1. - done_mask) * q_target_max
        q_candidate_val = np.sum(curr_q * actions, axis=1)
        _ = sum((q_target_val - q_candidate_val) ** 2)

        d = obs_t.shape[1] * obs_t.shape[2] * obs_t.shape[3]
        grad_loss = 2*(q_target_val - q_candidate_val).reshape((-1, 1))
        obs_t = obs_t.reshape([-1, d])
        gradient = np.zeros(model_curr['W0'].shape)
        for grad_loss, x, action in zip(grad_loss, obs_t, actions):
            x = x.reshape((x.shape[0], 1))
            action = action.reshape((1, action.shape[0]))
            gradient += np.asscalar(grad_loss) * x.dot(action)
        model_curr['W0'] += learning_rate * gradient
        return model_curr


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
    model_curr = {}
    model_target = {}
    run_id = str(start_time)[-5:].replace('.', '')

    for t in itertools.count():

        # 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # 2. Step the env and store the transition
        t_obs_idx = replay_buffer.store_frame(last_obs)

        if np.random.random() < exploration.value(t) \
                or not model_initialized \
                or not replay_buffer.can_sample(batch_size):
            action = env.action_space.sample()
        else:
            r_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
            curr_q_eval = q_func(r_obs, model_curr)
            action = np.argmax(curr_q_eval)

        last_obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(t_obs_idx, action, reward, done)

        if done:
            last_obs = env.reset()

        # 3. Perform experience relay and train the network.
        if (t > learning_starts
                and t % learning_freq == 0
                and replay_buffer.can_sample(batch_size)):

            obs_t, act_t, rew_t, obs_tp1, done_mask = \
                replay_buffer.sample(batch_size)

            if not model_initialized:
                model_initialized = True
                model_curr = initialize_model(input_shape, num_actions)
                model_target = model_curr

            learning_rate = lr_schedule.value(t)
            model_curr = train_func(
                obs_t=obs_t,
                act_t=act_t,
                rew_t=rew_t,
                obs_tp1=obs_tp1,
                done_mask=done_mask,
                learning_rate=learning_rate,
                model_curr=model_curr,
                model_target=model_target
            )

            if t % target_update_freq == 0:
                update_target_func(model_curr, model_target)
                num_param_updates += 1

        # 4. Log progress
        episode_rewards = get_wrapper_by_name(
            env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward,
                                           mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            if start_time is not None:
                print("Time %s s" % int(time.time() - start_time))
            start_time = time.time()
            print("Timestep %d" % t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % learning_rate)
            sys.stdout.flush()
    return model_curr