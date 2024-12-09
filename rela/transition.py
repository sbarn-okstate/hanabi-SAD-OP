"""
Code based on C++ PyTorch code from the following files:
https://github.com/codeaudit/hanabi_SAD/blob/master/rela/types.h
https://github.com/codeaudit/hanabi_SAD/blob/master/rela/types.cc
"""

import tensorflow as tf

class Transition:
    def __init__(self, obs, action, reward, terminal, bootstrap, next_obs):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.bootstrap = bootstrap
        self.next_obs = next_obs

    @classmethod
    def make_batch(cls, transitions, device):
        # Handle batching logic here, e.g., stacking tensors
        batch_obs = tf.stack([t.obs for t in transitions])
        batch_action = tf.stack([t.action for t in transitions])
        batch_reward = tf.stack([t.reward for t in transitions])
        batch_terminal = tf.stack([t.terminal for t in transitions])
        batch_bootstrap = tf.stack([t.bootstrap for t in transitions])
        batch_next_obs = tf.stack([t.next_obs for t in transitions])
        return cls(batch_obs, batch_action, batch_reward, batch_terminal, batch_bootstrap, batch_next_obs)

# Example usage
obs = tf.zeros((10, 4))  # Example observation tensor
action = tf.zeros((10, 2))  # Example action tensor
reward = tf.zeros((10, 1))  # Example reward tensor
terminal = tf.zeros((10, 1))  # Example terminal tensor
bootstrap = tf.zeros((10, 1))  # Example bootstrap tensor
next_obs = tf.zeros((10, 4))  # Example next observation tensor

transition = Transition(obs, action, reward, terminal, bootstrap, next_obs)
