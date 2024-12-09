"""
Code based on Python PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/pyhanabi/utils.py
"""

import time
import tensorflow as tf
import numpy as np
from create_envs import *  # Placeholder for environment setup utilities
from hanabi_env import *

class Tachometer:
    def __init__(self):
        self.num_act = 0
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self):
        self.t = time.time()

    def lap(self, actors, replay_buffer, num_train, factor):
        t = time.time() - self.t
        self.total_time += t
        num_act = get_num_acts(actors)
        act_rate = factor * (num_act - self.num_act) / t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        print(
            f"Speed: train: {train_rate:.1f}, act: {act_rate:.1f}, buffer_add: {buffer_rate:.1f}, buffer_size: {replay_buffer.size()}"
        )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            f"Total Time: {common_utils.sec2str(self.total_time)}, {int(self.total_time)}s"
        )
        print(
            f"Total Sample: train: {common_utils.num2str(self.num_train)}, act: {common_utils.num2str(self.num_act)}"
        )


def to_device(batch, device):
    if isinstance(batch, tf.Tensor):
        return tf.convert_to_tensor(batch, dtype=device)
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    elif isinstance(batch, FFTransition):
        batch.obs = to_device(batch.obs, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.next_obs = to_device(batch.next_obs, device)
        return batch
    elif isinstance(batch, RNNTransition):
        batch.obs = to_device(batch.obs, device)
        batch.h0 = to_device(batch.h0, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.seq_len = to_device(batch.seq_len, device)
        return batch
    else:
        raise TypeError(f"Unsupported type: {type(batch)}")


def get_game_info(num_player, greedy_extra):
    game = HanabiEnv({"players": str(num_player)}, -1, greedy_extra, False)

    info = {"input_dim": game.feature_size(), "num_action": game.num_action()}
    return info


def compute_input_dim(num_player):
    hand = 126 * num_player
    board = 76
    discard = 50
    last_action = 51 + 2 * num_player
    card_knowledge = num_player * 5 * 35
    return hand + board + discard + last_action + card_knowledge


def get_num_acts(actors):
    total_acts = sum(actor.num_act() for actor in actors)
    return total_acts


def get_frame_stat(num_game_per_thread, time_elapsed, num_acts, num_buffer, frame_stat):
    total_sample = (num_acts - frame_stat["num_acts"]) * num_game_per_thread
    act_rate = total_sample / time_elapsed
    buffer_rate = (num_buffer - frame_stat["num_buffer"]) / time_elapsed
    frame_stat["num_acts"] = num_acts
    frame_stat["num_buffer"] = num_buffer
    return total_sample, act_rate, buffer_rate


def generate_actor_eps(base_eps, alpha, num_actor):
    if num_actor == 1:
        return [base_eps]

    eps_list = []
    for i in range(num_actor):
        eps = base_eps ** (1 + i / (num_actor - 1) * alpha)
        eps_list.append(max(eps, 0))
    return eps_list


@tf.function
def get_v1(v0_joind, card_counts, ref_mask):
    v0_joind = tf.convert_to_tensor(v0_joind)
    card_counts = tf.convert_to_tensor(card_counts)

    batch, num_player, dim = v0_joind.shape
    num_player = 3
    v0_joind = tf.reshape(v0_joind, [batch, 1, num_player * 5, 25])

    mask = tf.cast(v0_joind > 0, tf.float32)
    total_viable_cards = tf.reduce_sum(mask)
    v1_old = v0_joind
    thres = 0.0001
    max_count = 100
    weight = 0.1
    v1_new = v1_old
    for _ in range(max_count):
        hand_cards = tf.reduce_sum(v1_old, axis=2)
        total_cards = card_counts - hand_cards
        excluding_self = tf.clip_by_value(total_cards[:, :, None, :] + v1_old, 0, np.inf)
        v1_new = excluding_self * mask
        v1_new = v1_old * (1 - weight) + weight * v1_new
        v1_new = v1_new / (tf.reduce_sum(v1_new, axis=-1, keepdims=True) + 1e-8)
        v1_old = v1_new

    return v1_new


@tf.function
def check_v1(v0, v1, card_counts, mask):
    ref_v1 = get_v1(v0, card_counts, mask)
    batch, num_player, dim = v1.shape
    v1 = tf.reshape(v1, [batch, 1, 3 * 5, 25])
    diff = tf.reduce_max(tf.abs(ref_v1 - v1))
    print("diff: ", diff)
    tf.debugging.assert_less_equal(diff, 1e-4, "V1 check failed")


def check_trajectory(batch):
    assert tf.reduce_sum(batch.obs["h"][0][0]) == 0
    length = batch.obs["h"][0].shape[0]
    end = 0
    for i in range(length):
        t = batch.terminal[0][i]
        if end != 0:
            assert t
        if t:
            end = i
    print("Trajectory ends at:", end)
