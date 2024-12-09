from __future__ import print_function
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import pprint
import utils
from utils import *
import argparse
from hanabi_learning_environment import pyhanabi
import common_utils
from create_envs import *
import vdn_r2d2
from rela.prioritized_replay import *
import iql_r2d2
from rela.transition_buffer import *
from rela.r2d2_actor import *

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")

    # game settings
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--greedy_extra", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument(
        "--batchsize", type=int, default=128,
    )
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--epoch_len", type=int, default=500)
    parser.add_argument("--num_update_between_sync", type=int, default=500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=1, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=1)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    return args
    """Example code demonstrating the Python Hanabi interface."""


if __name__ == "__main__":
        # Parse arguments (replace with your preferred argument parser)
    args = parse_args()
    #assert pyhanabi.cdef_loaded(), "cdef failed to load"
    #assert pyhanabi.lib_loaded(), "lib failed to load"
    
    # Create directories
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Logger setup
    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 10)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    game_info = get_game_info(args.num_player, args.greedy_extra)

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    if args.method == "vdn":
        agent = vdn_r2d2.R2D2Agent(
            args.multi_step,
            args.gamma,
            0.9,
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )

    eval_agents = []
    eval_lockers = []
    for _ in range(args.num_player):
        ea = iql_r2d2.R2D2Agent(
            1,
            0.99,
            0.9,
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        locker = ModelLocker(ea)
        eval_agents.append(ea)
        eval_lockers.append(locker)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=args.eps)
    print(agent)

    # Replay buffer
    replay_buffer = PrioritizedReplay(
        capacity=args.replay_buffer_size,
        seed=args.seed,
        alpha=args.priority_exponent,
        beta=args.priority_weight,
        prefetch=args.prefetch
    )

    ref_models = []
    model_lockers = []
    act_devices = args.act_device.split(",")
    for act_device in act_devices:
        ref_model = [agent.clone() for _ in range(3)]
        ref_models.extend(ref_model)
        model_locker = ModelLocker(ref_model[0])
        model_lockers.append(model_locker)

    # Actor epsilon values
    actor_eps = generate_actor_eps(args.act_base_eps, args.act_eps_alpha, args.num_thread)
    print("actor eps", actor_eps)

    if args.method == "vdn":
        actor_cons = []
        for _ in range(args.num_player):
            actor_cons.append(
                lambda thread_idx: R2D2Actor(
                    model_lockers[thread_idx % len(model_lockers)],
                    args.multi_step,
                    args.num_game_per_thread,
                    args.gamma,
                    args.max_len,
                    actor_eps[thread_idx],
                    args.num_player,
                    replay_buffer,
                )
            )
    # Create training environment
    context, games, actors, threads = create_train_env(
        args.method,
        args.seed,
        args.num_thread,
        args.num_game_per_thread,
        actor_cons,
        args.max_len,
        args.num_player,
        args.train_bomb,
        args.greedy_extra,
    )

    # Warm up replay buffer
    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("Warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    # Training loop
    tachometer = Tachometer()
    for epoch in range(args.num_epoch):
        print(f"Starting epoch {epoch}")
        stat = common_utils.MultiCounter(args.save_dir)
        stat.reset()
        tachometer.start()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()

            # Sample from replay buffer (PrioritizedReplay)
            batch, weights, sampledIds = replay_buffer.sample(args.batchsize, args.train_device)
            
            with tf.GradientTape() as tape:
                loss, priorities = agent.loss(batch)
                weighted_loss = tf.reduce_mean(loss * weights)

            # Backpropagation
            grads = tape.gradient(weighted_loss, agent.trainable_variables)
            grads = [tf.clip_by_norm(g, args.grad_clip) for g in grads]
            optimizer.apply_gradients(zip(grads, agent.trainable_variables))

            # Update priorities in replay buffer
            replay_buffer.updatePriority(priorities.numpy())

            # Log statistics
            stat["loss"].feed(weighted_loss.numpy())

        # Evaluate
        context.pause()
        score, perfect = evaluate(agent, args.num_eval_games, args.seed, args.num_player)
        print(f"Epoch {epoch} evaluation: Score {score:.4f}, Perfect games {perfect:.2f}%")

        # Save model
        model_saved = saver.save(agent, score)
        print(f"Model saved: {model_saved}")
        context.resume()