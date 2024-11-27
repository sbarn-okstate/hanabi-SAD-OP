import itertools
import hanabi_hand
import util
import hanabi_move
import hanabi_card
import hanabi_history_item

class HanabiObservation:
    def __init__(self, state, observing_player, show_cards=False):
        num_players = state.parent_game.num_players
        self.cur_player_offset = self._player_to_offset(state.cur_player(), observing_player, num_players)
        self.observing_player = observing_player
        self.discard_pile = state.discard_pile()
        self.fireworks = state.fireworks()
        self.deck_size = state.deck().size()
        self.information_tokens = state.information_tokens()
        self.life_tokens = state.life_tokens()
        self.legal_moves = state.legal_moves(observing_player)
        self.parent_game = state.parent_game

        self.hands = []
        hide_knowledge = state.parent_game.observation_type == "Minimal"
        show_cards = show_cards or state.parent_game.observation_type == "Seer"

        # Observing player's own hand
        self.hands.append(HanabiHand(state.hands()[observing_player], not show_cards, hide_knowledge))

        # Other players' hands
        for offset in range(1, state.parent_game.num_players):
            self.hands.append(HanabiHand(state.hands()[(observing_player + offset) % num_players], False, hide_knowledge))

        history = state.move_history()
        start = next((item for item in history if item.player != "Chance"), None)
        history.reverse()
        last_moves = []
        for item in history:
            last_moves.append(item)
            self._change_history_item_to_observer_relative(observing_player, num_players, show_cards, last_moves[-1])

            if item.player == observing_player:
                break

        self.last_moves = last_moves

    @staticmethod
    def _player_to_offset(pid, observer_pid, num_players):
        """Returns the offset of player pid relative to observer_pid"""
        return pid if pid < 0 else (pid - observer_pid + num_players) % num_players

    def _change_history_item_to_observer_relative(self, observer_pid, num_players, show_cards, item):
        if item.move.move_type == "Deal":
            assert item.player < 0 and item.deal_to_player >= 0
            item.deal_to_player = (item.deal_to_player - observer_pid + num_players) % num_players
            if item.deal_to_player == 0 and not show_cards:
                item.move = HanabiMove("Deal", -1, -1, -1, -1)
        else:
            assert item.player >= 0
            item.player = (item.player - observer_pid + num_players) % num_players

    def __str__(self):
        result = f"Life tokens: {self.life_tokens}\n"
        result += f"Info tokens: {self.information_tokens}\n"
        result += "Fireworks: " + " ".join(f"{color_index_to_char(i)}{self.fireworks[i]}" for i in range(self.parent_game.num_colors())) + "\n"
        result += "Hands:\n"
        for i, hand in enumerate(self.hands):
            if i > 0:
                result += "-----\n"
            if i == self.cur_player_offset:
                result += "Cur player\n"
            result += str(hand)
        result += f"Deck size: {self.deck_size}\n"
        result += "Discards: " + " ".join(str(card) for card in self.discard_pile)
        return result

    def card_playable_on_fireworks(self, color, rank):
        if color < 0 or color >= self.parent_game.num_colors():
            return False
        return rank == self.fireworks[color]