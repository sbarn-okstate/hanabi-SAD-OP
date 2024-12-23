"""
Code based on C++ PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/rela/env.h
"""

import tensorflow as tf
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any


class Env:
    def __init__(self):
        pass

    def reset(self) -> Dict[str, tf.Tensor]:
        """Resets the environment and returns the initial observation."""
        raise NotImplementedError("The reset method must be implemented by subclasses.")

    def step(self, action: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], float, bool]:
        """Performs an environment step with the given action."""
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def terminated(self) -> bool:
        """Returns whether the environment is in a terminal state."""
        raise NotImplementedError("The terminated method must be implemented by subclasses.")


class VectorEnv:
    def __init__(self):
        self.envs: List[Env] = []

    def append(self, env: Env):
        """Adds a new environment to the vector."""
        self.envs.append(env)

    def size(self) -> int:
        """Returns the number of environments in the vector."""
        return len(self.envs)

    def reset(self, input_tensor_dict: Dict[int, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Resets environments that have reached the end of their terminal state.
        Returns the new observation.
        """
        batch_obs = []
        for i, env in enumerate(self.envs):
            if env.terminated():
                obs = env.reset()
                batch_obs.append(obs)
            else:
                #Use the provided input for the environment
                batch_obs.append({key: value[i] for key, value in input_tensor_dict.items()})

        #Combine the batch of observations
        return self.tensor_dict_join(batch_obs)

    def step(self, action: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Takes a step in each environment and returns the observations, rewards, and terminal states.
        """
        batch_obs = []
        batch_reward = tf.zeros(len(self.envs), dtype=tf.float32)
        batch_terminal = tf.zeros(len(self.envs), dtype=tf.bool)

        for i, env in enumerate(self.envs):
            action_for_env = {key: value[i] for key, value in action.items()}
            obs, reward, terminal = env.step(action_for_env)

            batch_obs.append(obs)
            batch_reward = batch_reward.numpy()
            batch_reward[i] = reward
            batch_terminal = batch_terminal.numpy()
            batch_terminal[i] = terminal

        #Combine the batch of observations
        return self.tensor_dict_join(batch_obs), batch_reward, batch_terminal

    def any_terminated(self) -> bool:
        """Returns whether any environment in the vector is in a terminal state."""
        return any(env.terminated() for env in self.envs)

    def all_terminated(self) -> bool:
        """Returns whether all environments in the vector are in a terminal state."""
        return all(env.terminated() for env in self.envs)

    def tensor_dict_join(self, batch: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """
        Joins a list of tensor dictionaries along the first axis (batch dimension).
        Returns a new tensor dictionary with concatenated tensors.
        """
        joined_dict = {}
        for key in batch[0].keys():
            joined_dict[key] = tf.concat([obs[key] for obs in batch], axis=0)
        return joined_dict
