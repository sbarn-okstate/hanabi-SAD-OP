import numpy as np
import random
from collections import deque

class HanabiState:
    class HanabiDeck:
        def __init__(self, game):
            self.card_count = np.zeros(game.num_colors * game.num_ranks, dtype=int)
            self.total_count = 0
            self.num_ranks = game.num_ranks
            for color in range(game.num_colors):
                for rank in range(game.num_ranks):
                    count = game.number_card_instances(color, rank)
                    self.card_count[self.card_to_index(color, rank)] = count
                    self.total_count += count
            self.deck_history = deque()

        def card_to_index(self, color, rank):
            return color * self.num_ranks + rank

        def index_to_color(self, index):
            return index // self.num_ranks

        def index_to_rank(self, index):
            return index % self.num_ranks

        def deal_card(self, rng=None, color=None, rank=None):
            if self.empty():
                return None  # or HanabiCard() if defined
            if color is not None and rank is not None:
                index = self.card_to_index(color, rank)
                if self.card_count[index] <= 0:
                    return None
                self.card_count[index] -= 1
                self.total_count -= 1
                self.deck_history.append(index)
                return HanabiCard(self.index_to_color(index), self.index_to_rank(index))
            else:
                dist = np.array(self.card_count, dtype=float) / self.total_count
                index = np.random.choice(len(dist), p=dist)
                assert self.card_count[index] > 0
                self.card_count[index] -= 1
                self.total_count -= 1
                self.deck_history.append(index)
                return HanabiCard(self.index_to_color(index), self.index_to_rank(index))

        def empty(self):
            return self.total_count == 0

    def __init__(self, parent_game, start_player):
        self.parent_game = parent_game
        self.deck = HanabiState.HanabiDeck(parent_game)
        self.hands = [HanabiHand() for _ in range(parent_game.num_players)]
        self.cur_player = kChancePlayerId
        self.next_non_chance_player = start_player if 0 <= start_player < parent_game.num_players else parent_game.sampled_start_player
        self.information_tokens = parent_game.max_information_tokens
        self.life_tokens = parent_game.max_life_tokens
        self.fireworks = np.zeros(parent_game.num_colors, dtype=int)
        self.turns_to_play = parent_game.num_players

    def advance_to_next_player(self):
        if not self.deck.empty() and self.player_to_deal() >= 0:
            self.cur_player = kChancePlayerId
        else:
            self.cur_player = self.next_non_chance_player
            self.next_non_chance_player = (self.cur_player + 1) % len(self.hands)

    def increment_information_tokens(self):
        if self.information_tokens < self.parent_game.max_information_tokens:
            self.information_tokens += 1
            return True
        return False

    def decrement_information_tokens(self):
        assert self.information_tokens > 0
        self.information_tokens -= 1

    def decrement_life_tokens(self):
        assert self.life_tokens > 0
        self.life_tokens -= 1

    def add_to_fireworks(self, card):
        if self.card_playable_on_fireworks(card):
            self.fireworks[card.color] += 1
            if self.fireworks[card.color] == self.parent_game.num_ranks:
                return True, self.increment_information_tokens()
            return True, False
        else:
            self.decrement_life_tokens()
            return False, False

    def hinting_is_legal(self, move):
        if self.information_tokens <= 0:
            return False
        if not (1 <= move.target_offset < self.parent_game.num_players):
            return False
        return True

    def player_to_deal(self):
        for i, hand in enumerate(self.hands):
            if len(hand.cards) < self.parent_game.hand_size:
                return i
        return -1

    def move_is_legal(self, move):
        if move.move_type == HanabiMove.k_deal:
            if self.cur_player != kChancePlayerId:
                return False
            if self.deck.card_count[self.card_to_index(move.color, move.rank)] == 0:
                return False
        elif move.move_type == HanabiMove.k_discard:
            if self.information_tokens >= self.parent_game.max_information_tokens:
                return False
            if move.card_index >= len(self.hands[self.cur_player].cards):
                return False
        elif move.move_type == HanabiMove.k_play:
            if move.card_index >= len(self.hands[self.cur_player].cards):
                return False
        elif move.move_type in {HanabiMove.k_reveal_color, HanabiMove.k_reveal_rank}:
            if not self.hinting_is_legal(move):
                return False
            cards = self.hand_by_offset(move.target_offset).cards
            card_check = (lambda card: card.color == move.color) if move.move_type == HanabiMove.k_reveal_color else (lambda card: card.rank == move.rank)
            if not any(card_check(card) for card in cards):
                return False
        else:
            return False
        return True

    def apply_move(self, move):
        assert self.move_is_legal(move)
        if self.deck.empty():
            self.turns_to_play -= 1
        history = HanabiHistoryItem(move)
        history.player = self.cur_player
        if move.move_type == HanabiMove.k_deal:
            history.deal_to_player = self.player_to_deal()
            card_knowledge = HanabiHand.CardKnowledge(self.parent_game.num_colors, self.parent_game.num_ranks)
            if self.parent_game.observation_type == HanabiGame.k_seer:
                card_knowledge.apply_is_color_hint(move.color)
                card_knowledge.apply_is_rank_hint(move.rank)
            self.hands[history.deal_to_player].add_card(self.deck.deal_card(move.color, move.rank), card_knowledge)
        elif move.move_type == HanabiMove.k_discard:
            history.information_token = self.increment_information_tokens()
            history.color = self.hands[self.cur_player].cards[move.card_index].color
            history.rank = self.hands[self.cur_player].cards[move.card_index].rank
            self.hands[self.cur_player].remove_from_hand(move.card_index, self.discard_pile)
        elif move.move_type == HanabiMove.k_play:
            history.color = self.hands[self.cur_player].cards[move.card_index].color
            history.rank = self.hands[self.cur_player].cards[move.card_index].rank
            history.scored, history.information_token = self.add_to_fireworks(self.hands[self.cur_player].cards[move.card_index])
            self.hands[self.cur_player].remove_from_hand(move.card_index, None if history.scored else self.discard_pile)
        elif move.move_type == HanabiMove.k_reveal_color:
            self.decrement_information_tokens()
            history.reveal_bitmask = self.hand_color_bitmask(self.hand_by_offset(move.target_offset), move.color)
            history.newly_revealed_bitmask = self.hand_by_offset(move.target_offset).reveal_color(move.color)
        elif move.move_type == HanabiMove.k_reveal_rank:
            self.decrement_information_tokens()
            history.reveal_bitmask = self.hand_rank_bitmask(self.hand_by_offset(move.target_offset), move.rank)
            history.newly_revealed_bitmask = self.hand_by_offset(move.target_offset).reveal_rank(move.rank)
        else:
            raise ValueError("Unexpected move type")
        self.move_history.append(history)
        self.advance_to_next_player()

    def chance_outcome_prob(self, move):
        return self.deck.card_count[self.card_to_index(move.color, move.rank)] / self.deck.total_count

    def apply_random_chance(self):
        chance_outcomes = self.chance_outcomes()
        assert chance_outcomes[1]
        self.apply_move(self.parent_game.pick_random_chance(chance_outcomes))

    def legal_moves(self, player):
        if player != self.cur_player:
            return []
        return [move for move in range(self.parent_game.max_moves) if self.move_is_legal(move)]

    def card_playable_on_fireworks(self, color, rank):
        if color < 0 or color >= self.parent_game.num_colors:
            return False
        return rank == self.fireworks[color]

    def chance_outcomes(self):
        outcomes = ([], [])
        for uid in range(self.parent_game.max_chance_outcomes):
            move = self.parent_game.get_chance_outcome(uid)
            if self.move_is_legal(move):
                outcomes[0].append(move)
                outcomes[1].append(self.chance_outcome_prob(move))
        return outcomes

    def to_string(self):
        result = f"Life tokens: {self.life_tokens}\nInfo tokens: {self.information_tokens}\nFireworks: "
        result += " ".join(f"{self.color_index_to_char(i)}{firework}" for i, firework in enumerate(self.fireworks)) + "\nHands:\n"
        for i, hand in enumerate(self.hands):
            if i
