import threading
import numpy as np
import logging
from typing import List
from rela.thread_loop import ThreadLoop
# Assuming Actor and VectorEnv are defined elsewhere in the project.
from rela.actor import Actor
from rela.env import VectorEnv

class HanabiVDNThreadLoop(ThreadLoop):
    def __init__(self, actor, env, eval=False):
        super().__init__()
        self.actor = actor
        self.env = env
        self.eval = eval
        if self.eval:
            assert len(self.env) == 1  # Eval only works for a single environment.

    def main_loop(self):
        obs = {}
        r, t = None, None
        while not self.terminated():
            obs = self.env.reset(obs)
            while not self.env.any_terminated():
                if self.terminated():
                    break

                if self.paused():
                    self.wait_until_resume()

                # Actor performs an action in the environment.
                action = self.actor.act(obs)
                obs, r, t = self.env.step(action)

                if self.eval:
                    continue

                # Set reward and terminal state for the actor.
                self.actor.set_reward_and_terminal(r, t)
                self.actor.post_step()

            # In evaluation mode, only run for one game.
            if self.eval:
                break


class HanabiIQLThreadLoop(ThreadLoop):
    def __init__(self, actors: List[Actor], env: VectorEnv, eval=False, log_file=""):
        super().__init__()
        self.actors = actors
        self.env = env
        self.eval = eval
        self.log_file = log_file
        if self.eval:
            assert len(self.env) == 1  # Eval only works for a single environment.

    def main_loop(self):
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)

        obs = {}
        r, t = None, None
        while not self.terminated():
            obs = self.env.reset(obs)
            while not self.env.any_terminated():
                if self.terminated():
                    break

                if self.paused():
                    self.wait_until_resume()

                actions = []
                greedy_actions = []

                # Process actions for each actor
                for i, actor in enumerate(self.actors):
                    input_data = self._narrow_tensor(obs, 1, i, 1, True)

                    if self.log_file:
                        self._log_state(input_data)

                    reply = actor.act(input_data)
                    actions.append(reply["a"])
                    greedy_actions.append(reply["greedy_a"])

                    if self.log_file:
                        self._log_action(reply)

                action = {
                    "a": np.stack(actions, axis=1),
                    "greedy_a": np.stack(greedy_actions, axis=1),
                }

                obs, r, t = self.env.step(action)

                if self.eval:
                    continue

                for actor in self.actors:
                    actor.set_reward_and_terminal(r, t)
                    actor.post_step()

            # Eval only runs for one game
            if self.eval:
                break

    def _narrow_tensor(self, obs, dim, start, size, squeeze=False):
        # Simulate the narrowing operation on the tensor.
        # This would slice the observation tensor appropriately
        # For now, assume obs is a dictionary with tensors and return a subset of one.
        return {key: value[start:start+size] for key, value in obs.items()}

    def _log_state(self, input_data):
        for key, value in input_data.items():
            if key in ["s", "legal_move"]:
                logging.info(f"{key}: {value.tolist()}")

    def _log_action(self, reply):
        logging.info(f"action: {reply['a']}")
        logging.info("----------")
