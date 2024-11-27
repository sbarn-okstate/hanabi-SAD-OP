import set_path

set_path.append_sys_path()

import os
import pprint
import tensorflow as tf
from rela import *
from hanabi_learning_environment import pyhanabi
from hanabi_env import *
from rela.env import *
from rela.r2d2_actor import *
from rela.context import *
from thread_loop import *

def create_train_env(
    method,
    seed,
    num_thread,
    num_game_per_thread,
    actor_cons,
    max_len,
    num_player,
    bomb,
    greedy_extra,
):
    assert method in ["vdn", "iql"]
    context = Context()
    games = []
    actors = []
    threads = []
    print(f"Training with bomb: {bomb}")

    for thread_idx in range(num_thread):
        env = VectorEnv()
        for game_idx in range(num_game_per_thread):
            unique_seed = seed + game_idx + thread_idx * num_game_per_thread
            game = HanabiEnv(
                {
                    "players": str(num_player),
                    "seed": str(unique_seed),
                    "bomb": str(bomb),
                },
                max_len,
                greedy_extra,
                False,
            )
            games.append(game)
            env.append(game)

        assert max_len > 0
        if method == "vdn":
            actor = actor_cons[thread_idx]
            actors.append(actor)
            thread = HanabiVDNThreadLoop(actor, env, False)
        else:
            assert len(actor_cons) == num_player
            env_actors = [actor_cons[i](thread_idx) for i in range(num_player)]
            actors.extend(env_actors)
            thread = HanabiIQLThreadLoop(env_actors, env, False)

        threads.append(thread)
        context.pushThreadLoop(thread)

    print(
        f"Finished creating environments with {len(games)} games and {len(actors)} actors"
    )
    return context, games, actors, threads


def create_eval_env(
    seed,
    num_thread,
    model_lockers,
    eval_eps,
    num_player,
    bomb,
    greedy_extra,
    log_prefix=None,
):
    context = rela.Context()
    games = []

    for i in range(num_thread):
        game = HanabiEnv(
            {"players": str(num_player), "seed": str(seed + i), "bomb": str(bomb)},
            -1,
            greedy_extra,
            False,
        )
        games.append(game)
        env = env.VectorEnv()
        env.append(game)

        env_actors = [
            R2D2Actor(model_lockers[j], 1, eval_eps) for j in range(num_player)
        ]

        if log_prefix is None:
            thread = HanabiIQLThreadLoop(env_actors, env, True)
        else:
            log_file = os.path.join(log_prefix, f"game{i}.txt")
            thread = HanabiIQLThreadLoop(env_actors, env, True, log_file)

        context.push_env_thread(thread)

    return context, games
