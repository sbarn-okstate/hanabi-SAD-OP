"""
Code based on C++ PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/rela/dqn_actor.h
"""

import tensorflow as tf
import numpy as np
from collections import deque
import random

class MultiStepTransitionBuffer:
    def __init__(self, multi_step, batch_size, gamma):
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.gamma = gamma

        self.obs_history = deque(maxlen=multi_step + 1)
        self.action_history = deque(maxlen=multi_step + 1)
        self.reward_history = deque(maxlen=multi_step + 1)
        self.terminal_history = deque(maxlen=multi_step + 1)

    def push_obs_and_action(self, obs, action):
        assert len(self.obs_history) <= self.multi_step
        assert len(self.action_history) <= self.multi_step
        self.obs_history.append(obs)
        self.action_history.append(action)

    def push_reward_and_terminal(self, reward, terminal):
        assert len(self.reward_history) == len(self.terminal_history)
        assert len(self.reward_history) == len(self.obs_history) - 1
        assert reward.shape == terminal.shape
        assert reward.shape[0] == self.batch_size

        self.reward_history.append(reward)
        self.terminal_history.append(terminal)

    def size(self):
        return len(self.obs_history)

    def can_pop(self):
        return len(self.obs_history) == self.multi_step + 1

    def pop_transition(self):
        assert len(self.obs_history) == self.multi_step + 1
        assert len(self.action_history) == self.multi_step + 1
        assert len(self.reward_history) == self.multi_step + 1
        assert len(self.terminal_history) == self.multi_step + 1

        obs = self.obs_history.popleft()
        action = self.action_history.popleft()
        terminal = self.terminal_history.popleft()

        bootstrap = np.ones(self.batch_size)
        next_obs_indices = np.zeros(self.batch_size, dtype=int)

        #Calculate bootstrap and next state indices
        for i in range(self.batch_size):
            for step in range(self.multi_step):
                if self.terminal_history[step][i]:
                    bootstrap[i] = 0.0
                    next_obs_indices[i] = step
                    break
            if bootstrap[i] > 1e-6:
                next_obs_indices[i] = self.multi_step

        #Calculate discounted rewards
        reward = np.zeros_like(self.reward_history[0])
        for i in range(self.batch_size):
            initial = self.multi_step - 1 if bootstrap[i] else next_obs_indices[i]
            for step in range(initial, -1, -1):
                reward[i] = self.reward_history[step][i] + self.gamma * reward[i]

        next_obs = self.obs_history[-1]

        return obs, action, reward, terminal, bootstrap, next_obs

    def clear(self):
        self.obs_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.terminal_history.clear()


class DQNActor:
    def __init__(self, model_locker, multi_step, batch_size, gamma, replay_buffer=None):
        self.batch_size = batch_size
        self.model_locker = model_locker
        self.transition_buffer = MultiStepTransitionBuffer(multi_step, batch_size, gamma)
        self.replay_buffer = replay_buffer
        self.num_act = 0

    def num_act(self):
        return self.num_act

    def act(self, obs):
        input_obs = self.tensor_dict_to_tensor(obs)

        #Get model and perform action
        model = self.model_locker.get_model()
        action = model(input_obs)

        if self.replay_buffer is not None:
            self.transition_buffer.push_obs_and_action(obs, action)

        self.num_act += self.batch_size
        return action

    def set_reward_and_terminal(self, reward, terminal):
        assert self.replay_buffer is not None
        self.transition_buffer.push_reward_and_terminal(reward, terminal)

    def post_step(self):
        assert self.replay_buffer is not None
        if not self.transition_buffer.can_pop():
            return

        obs, action, reward, terminal, bootstrap, next_obs = self.transition_buffer.pop_transition()
        priority = self.compute_priority(obs, action, reward, terminal, next_obs)
        self.replay_buffer.add((obs, action, reward, terminal, bootstrap, next_obs), priority)

    def compute_priority(self, obs, action, reward, terminal, next_obs):
        #Convert observations and actions to TensorFlow tensors and use the model to compute priority
        input_data = {
            'obs': tf.convert_to_tensor(obs),
            'action': tf.convert_to_tensor(action),
            'reward': tf.convert_to_tensor(reward),
            'terminal': tf.convert_to_tensor(terminal),
            'next_obs': tf.convert_to_tensor(next_obs)
        }
        model = self.model_locker.get_model()
        priority = model.compute_priority(input_data)
        return priority

    def tensor_dict_to_tensor(self, tensor_dict):
        #Convert a dictionary of tensors to a single tensor for the model input
        return tf.stack([tensor_dict[key] for key in sorted(tensor_dict.keys())])

#Tensorflow manages models across threads so we don't need this but for now its just a placeholder
class ModelLocker:
    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def release_model(self):
        pass
