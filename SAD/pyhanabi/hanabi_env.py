import numpy as np
import tensorflow as tf
from hanabi_learning_environment import pyhanabi

class HanabiEnv:
    def __init__(self, game_params, max_len, greedy_extra, verbose):
        self.game = pyhanabi.HanabiGame(game_params)
        self.obs_encoder = pyhanabi.ObservationEncoder(self.game)
        self.state = None
        self.max_len = max_len
        self.num_step = 0
        self.greedy_extra = greedy_extra
        self.verbose = verbose

        if self.verbose:
            print("Hanabi game created, with parameters:")
            for key, value in self.game.parameters.items():
                print(f"  {key}={value}")

    def feature_size(self):
        size = self.obs_encoder.shape()[0]
        if self.greedy_extra:
            size += HanabiLearningEnv.LastActionSectionLength(self.game)
        return size

    def num_action(self):
        return self.game.max_moves() + 1

    def no_op_uid(self):
        return self.num_action() - 1

    def hand_feature_size(self):
        return self.game.hand_size * self.game.num_colors * self.game.num_ranks

    def reset(self):
        assert self.terminated()
        self.state = HanabiState(self.game)

        # Chance player
        while self.state.cur_player == -1 #HanabiLearningEnv.kChancePlayerId: <- constexpr = -1
            self.state.apply_random_chance()

        self.num_step = 0
        return self.compute_feature_and_legal_move(self.state)

    def step(self, action):
        assert not self.terminated()
        self.num_step += 1

        prev_score = self.state.score()

        # Perform action for only current player
        cur_player = self.state.cur_player
        action_uid = action['a'][cur_player].numpy()
        move = self.game.get_move(action_uid)
        if not self.state.move_is_legal(move):
            raise ValueError("Error: move is not legal")

        clone_state = None
        if self.greedy_extra:
            clone_state = HanabiState(self.state)
            greedy_action_uid = action['greedy_a'][cur_player].numpy()
            greedy_move = self.game.get_move(greedy_action_uid)
            if not self.state.move_is_legal(greedy_move):
                raise ValueError("Error: greedy move is not legal")
            clone_state.apply_move(greedy_move)

        self.state.apply_move(move)

        terminal = self.state.is_terminal()
        reward = self.state.score() - prev_score

        # Forced termination, lose all points
        if self.max_len > 0 and self.num_step == self.max_len:
            terminal = True
            reward = -prev_score

        if not terminal:
            # Chance player
            while self.state.cur_player == -1 #HanabiLearningEnv.kChancePlayerId: <- constexpr = -1
                self.state.apply_random_chance()

        obs = self.compute_feature_and_legal_move(clone_state)
        return obs, reward, terminal

    def terminated(self):
        if self.state is None:
            return True
        if self.max_len <= 0:
            return self.state.is_terminal()
        else:
            return self.state.is_terminal() or self.num_step >= self.max_len

    def get_episode_reward(self):
        assert self.state is not None
        return self.state.score()

    def deck_history(self):
        return self.state.deck_history()

    def compute_feature_and_legal_move(self, clone_state):
        s = []
        legal_move = []

        for i in range(self.game.num_players):
            obs = HanabiObservation(self.state, i, False)
            v_s = self.obs_encoder.encode(obs)
            if self.greedy_extra:
                assert clone_state is not None
                extra_obs = HanabiObservation(clone_state, i, False)
                v_greedy_action = self.obs_encoder.encode_last_action(extra_obs)
                v_s.extend(v_greedy_action)

            s.append(tf.convert_to_tensor(v_s, dtype=tf.float32))

            legal_moves = self.state.legal_moves(i)
            move_uids = np.zeros(self.num_action())
            for move in legal_moves:
                uid = self.game.get_move_uid(move)
                if uid >= self.no_op_uid():
                    raise ValueError(f"Error: legal move id should be < {self.num_action() - 1}")
                move_uids[uid] = 1
            if len(legal_moves) == 0:
                move_uids[self.no_op_uid()] = 1

            legal_move.append(tf.convert_to_tensor(move_uids, dtype=tf.float32))

        return {'s': tf.stack(s, 0), 'legal_move': tf.stack(legal_move, 0)}
