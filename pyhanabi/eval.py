"""
Code based on Python PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/pyhanabi/eval.py
"""

import time
import numpy as np
import tensorflow as tf
from create_envs import create_eval_env
import utils


def evaluate(
    model_lockers,
    num_game,
    seed,
    eval_eps,
    num_player,
    bomb,
    greedy_extra,
    *,
    log_prefix=None,
):
    context, games = create_eval_env(
        seed,
        num_game,
        model_lockers,
        eval_eps,
        num_player,
        bomb,
        greedy_extra,
        log_prefix,
    )
    context.start()
    while not context.terminated():
        time.sleep(0.5)

    context.terminate()
    while not context.terminated():
        time.sleep(0.5)

    scores = [g.get_episode_reward() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def evaluate_saved_model(
    weight_files, num_game, seed, bomb, num_run=1, log_prefix=None, verbose=True
):
    model_lockers = []
    greedy_extra = 0
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    for weight_file in weight_files:
        if verbose:
            print(
                f"Evaluating: {weight_file}\n\tfor {num_run}x{num_game} games"
            )
        if (
            "GREEDY_EXTRA1" in weight_file
            or "sad" in weight_file
            or "aux" in weight_file
        ):
            player_greedy_extra = 1
            greedy_extra = 1
        else:
            player_greedy_extra = 0

        device = tf.device("/CPU:0")
        game_info = utils.get_game_info(num_player, player_greedy_extra)
        input_dim = game_info["input_dim"]
        output_dim = game_info["num_action"]
        hid_dim = 512

        #Initialize TensorFlow R2D2 agent
        actor = iql_r2d2.R2D2Agent(1, 0.99, 0.9, device, input_dim, hid_dim, output_dim)

        #Load saved weights
        state_dict = tf.saved_model.load(weight_file)
        if "pred.weight" in state_dict:
            del state_dict["pred.bias"]
            del state_dict["pred.weight"]

        #Assuming the TensorFlow equivalent of load_state_dict is a weight assignment
        actor.online_net.set_weights([state_dict[key] for key in actor.online_net.trainable_variables])

        #Add the model to lockers, although I don't think we need this because tensorflow should handle model locking
        model_lockers.append(utils.ModelLocker([actor], device))

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            model_lockers,
            num_game,
            num_game * i + seed,
            0,
            num_player,
            bomb,
            greedy_extra,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(f"Score: {mean:.6f} +/- {sem:.6f}; Perfect: {perfect_rate}")
    return mean, sem, perfect_rate
