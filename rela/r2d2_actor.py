import tensorflow as tf
import numpy as np
from collections import deque
from rela.actor import Actor
from rela.transition_buffer import MultiStepTransitionBuffer

class R2D2TransitionBuffer:
    def __init__(self, batch_size, num_players, multi_step, seq_len):
        self.batch_size = batch_size
        self.num_players = num_players
        self.multi_step = multi_step
        self.seq_len = seq_len
        self.batch_next_idx = [0] * batch_size
        self.batch_h0 = [None] * batch_size  # Placeholder for h0
        self.batch_seq_transition = [[None] * seq_len for _ in range(batch_size)]
        self.batch_seq_priority = [[0.0] * seq_len for _ in range(batch_size)]
        self.batch_len = [0] * batch_size
        self.can_pop = False

    def push(self, transition, priority, tensor_dict):
        assert priority.shape[0] == self.batch_size

        for i in range(self.batch_size):
            next_idx = self.batch_next_idx[i]
            assert 0 <= next_idx < self.seq_len

            # (Optional) Simplification of h0 handling
            if next_idx == 0:
                pass  # Placeholder for logic related to batch_h0[i]

            t = transition[i]  # Assume transition is a list or tensor
            if next_idx != 0:
                # Ensure no transitions after terminal
                assert not t['terminal']
                assert self.batch_len[i] == 0

            self.batch_seq_transition[i][next_idx] = t
            self.batch_seq_priority[i][next_idx] = priority[i]

            self.batch_next_idx[i] += 1
            if not t['terminal']:
                continue

            self.batch_len[i] = self.batch_next_idx[i]
            while self.batch_next_idx[i] < self.seq_len:
                self.batch_seq_transition[i][self.batch_next_idx[i]] = t  # Pad with the last transition
                self.batch_seq_priority[i][self.batch_next_idx[i]] = 0
                self.batch_next_idx[i] += 1

            self.can_pop = True

    def can_pop(self):
        return self.can_pop

    def pop_transition(self):
        assert self.can_pop

        batch_transition = []
        batch_seq_priority = []
        batch_len = []

        for i in range(self.batch_size):
            if self.batch_len[i] == 0:
                continue

            assert self.batch_next_idx[i] == self.seq_len

            batch_seq_priority.append(tf.convert_to_tensor(self.batch_seq_priority[i], dtype=tf.float32))
            batch_len.append(float(self.batch_len[i]))

            # Create RNNTransition (equivalent)
            t = {'sequence': self.batch_seq_transition[i], 'h0': self.batch_h0[i], 'length': float(self.batch_len[i])}
            batch_transition.append(t)

            self.batch_len[i] = 0
            self.batch_next_idx[i] = 0

        self.can_pop = False
        assert len(batch_transition) > 0

        return batch_transition, tf.stack(batch_seq_priority, axis=0), tf.convert_to_tensor(batch_len, dtype=tf.float32)


class R2D2Actor(Actor):
    def __init__(self, model_locker, multi_step, batch_size, gamma, seq_len, greedy_eps, num_players, replay_buffer=None):
        self.model_locker = model_locker
        self.batch_size = batch_size
        self.greedy_eps = greedy_eps
        self.num_players = num_players
        self.r2d2_buffer = R2D2TransitionBuffer(batch_size, num_players, multi_step, seq_len)
        self.multi_step_buffer = MultiStepTransitionBuffer(multi_step, batch_size, gamma)
        self.replay_buffer = replay_buffer
        self.hidden = self.get_h0(1 * num_players)
        self.num_act = 0
        self.history_hidden = deque()

    def num_act(self):
        return self.num_act

    def act(self, obs):
        # Equivalent to NoGradGuard in PyTorch
        assert self.hidden is not None

        if self.replay_buffer is not None:
            self.history_hidden.append(self.hidden)

        eps = tf.fill([self.batch_size, self.num_players], self.greedy_eps)
        obs["eps"] = eps

        # Convert observation to input for model (TensorFlow equivalent of tensorDictToTorchDict)
 
        self.hidden["h0"] = tf.stop_gradient(self.hidden["h0"])
        obs["s"] = tf.stop_gradient(obs["s"])
        obs["legal_move"] = tf.stop_gradient(obs["legal_move"])
        obs["eps"] = tf.stop_gradient(obs["eps"])
        
        output = self.model_locker.get_model().act(obs, self.hidden)

        action = output[0]
        self.hidden = output[1]

        if self.replay_buffer is not None:
            self.multi_step_buffer.push_obs_and_action(obs, action)

        self.num_act += self.batch_size
        return action

    def set_reward_and_terminal(self, r, t):
        if self.replay_buffer is None:
            return

        self.multi_step_buffer.push_reward_and_terminal(r, t)

        # Reset hidden states for terminal states
        h0 = self.get_h0(1)
        terminal = t#.numpy()

        for i in range(len(terminal)):
            if terminal[i]:
                for name, tensor in self.hidden.items():
                    self.hidden[name][i * self.num_players:(i + 1) * self.num_players] = h0[name]

    def post_step(self):
        if self.replay_buffer is None:
            return

        if not self.multi_step_buffer.can_pop():
            assert not self.r2d2_buffer.can_pop
            return

        # Get transition and calculate priority
        transition = list(self.multi_step_buffer.pop_transition())
        hid = self.history_hidden[0]
        next_hid = self.history_hidden[-1]
        self.history_hidden.pop()

        priority = self.compute_priority(transition, hid, next_hid)
        self.r2d2_buffer.push(transition, priority, hid)

        if self.r2d2_buffer.can_pop():
            batch, seq_batch_priority, batch_len = self.r2d2_buffer.pop_transition()
            priority = self.aggregate_priority(seq_batch_priority, batch_len)
            self.replay_buffer.add(batch, priority)

    def get_h0(self, batch_size):
        # Placeholder for getting initial hidden state
        return self.model_locker.get_model().get_h0(batch_size)

    def compute_priority(self, transition, hid, next_hid):
        input_data =  transition + [hid, next_hid]        
        priority = self.model_locker.get_model().compute_priority(*input_data)
        return priority

    def aggregate_priority(self, priority, length):
        input_data = [priority, length]
        agg_priority = self.model_locker.get_model().aggregate_priority(*input_data)
        return agg_priority
