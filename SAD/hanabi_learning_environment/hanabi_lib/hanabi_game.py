import random
from collections import defaultdict
from typing import Dict, List, Tuple

class HanabiGame:
    class AgentObservationType:
        kMinimal = 0
        kCardKnowledge = 1
        kSeer = 2

    def __init__(self, params: Dict[str, str]):
        self.params = params
        self.num_players = self.get_param("players", 2)
        self.num_colors = self.get_param("colors", 5)
        self.num_ranks = self.get_param("ranks", 5)
        self.hand_size = self.get_param("hand_size", self.hand_size_from_rules())
        self.max_information_tokens = self.get_param("max_information_tokens", 8)
        self.max_life_tokens = self.get_param("max_life_tokens", 3)
        self.seed = self.get_param("seed", -1)
        self.random_start_player = self.get_param("random_start_player", False)
        self.observation_type = self.get_param("observation_type", self.AgentObservationType.kCardKnowledge)
        self.bomb = self.get_param("bomb", 0)

        if self.seed == -1:
            self.seed = random.randint(0, 2**32 - 1)
        self.rng = random.Random(self.seed)

        self.cards_per_color = self.calculate_cards_per_color()
        self.moves = [self.construct_move(uid) for uid in range(self.max_moves())]
        self.chance_outcomes = [self.construct_chance_outcome(uid) for uid in range(self.max_chance_outcomes())]

    def get_param(self, key: str, default):
        return int(self.params.get(key, default))

    def max_moves(self):
        return self.max_discard_moves() + self.max_play_moves() + self.max_reveal_color_moves() + self.max_reveal_rank_moves()

    def max_chance_outcomes(self):
        return self.num_colors * self.num_ranks

    def max_discard_moves(self):
        return self.hand_size

    def max_play_moves(self):
        return self.hand_size

    def max_reveal_color_moves(self):
        return (self.num_players - 1) * self.num_colors

    def max_reveal_rank_moves(self):
        return (self.num_players - 1) * self.num_ranks

    def hand_size_from_rules(self):
        return 5 if self.num_players < 4 else 4

    def calculate_cards_per_color(self):
        cards = 0
        for rank in range(self.num_ranks):
            cards += self.number_card_instances(0, rank)
        return cards

    def number_card_instances(self, color, rank):
        if color < 0 or color >= self.num_colors or rank < 0 or rank >= self.num_ranks:
            return 0
        if rank == 0:
            return 3
        elif rank == self.num_ranks - 1:
            return 1
        return 2

    def construct_move(self, uid):
        if uid < 0 or uid >= self.max_moves():
            return HanabiMove(HanabiMove.kInvalid)

        if uid < self.max_discard_moves():
            return HanabiMove(HanabiMove.kDiscard, card_index=uid)
        uid -= self.max_discard_moves()

        if uid < self.max_play_moves():
            return HanabiMove(HanabiMove.kPlay, card_index=uid)
        uid -= self.max_play_moves()

        if uid < self.max_reveal_color_moves():
            return HanabiMove(HanabiMove.kRevealColor,
                              target_offset=1 + uid // self.num_colors,
                              color=uid % self.num_colors)
        uid -= self.max_reveal_color_moves()

        return HanabiMove(HanabiMove.kRevealRank,
                          target_offset=1 + uid // self.num_ranks,
                          rank=uid % self.num_ranks)

    def construct_chance_outcome(self, uid):
        if uid < 0 or uid >= self.max_chance_outcomes():
            return HanabiMove(HanabiMove.kInvalid)

        return HanabiMove(HanabiMove.kDeal,
                          color=(uid // self.num_ranks) % self.num_colors,
                          rank=uid % self.num_ranks)

    def get_sampled_start_player(self):
        if self.random_start_player:
            return self.rng.randint(0, self.num_players - 1)
        return 0

    def pick_random_chance(self, chance_outcomes: Tuple[List[HanabiMove], List[float]]):
        distribution = chance_outcomes[1]
        choices = chance_outcomes[0]
        return self.rng.choices(choices, weights=distribution)[0]