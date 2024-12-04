from abc import ABC, abstractmethod
import numpy as np

class Actor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, obs):
        """
        Given the observations, returns the actions to take.

        :param obs: The observations from the environment (could be a dictionary or tensor).
        :return: A dictionary of actions (or a tensor representing actions).
        """
        pass

    @abstractmethod
    def set_reward_and_terminal(self, r, t):
        """
        Sets the reward and terminal state for the actor after taking an action.

        :param r: The reward received after taking an action.
        :param t: The terminal condition of the environment.
        """
        pass

    @abstractmethod
    def post_step(self):
        """
        Perform any necessary updates or clean-up after the step.
        """
        pass
