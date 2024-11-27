import util
import hanabi_move

class HanabiHistoryItem:
    def __init__(self, move_made):
        self.move = move_made
        self.player = -1
        self.scored = False
        self.information_token = False
        self.color = -1
        self.rank = -1
        self.reveal_bitmask = 0
        self.newly_revealed_bitmask = 0
        self.deal_to_player = -1

    def __str__(self):
        result = f"<{str(self.move)}"
        if self.player >= 0:
            result += f" by player {self.player}"
        if self.scored:
            result += " scored"
        if self.information_token:
            result += " info_token"
        if self.color >= 0:
            assert self.rank >= 0
            result += f" {color_index_to_char(self.color)}{rank_index_to_char(self.rank)}"
        if self.reveal_bitmask:
            result += " reveal "
            first = True
            for i in range(8):  # 8 bits in reveal_bitmask
                if self.reveal_bitmask & (1 << i):
                    if first:
                        first = False
                    else:
                        result += ","
                    result += str(i)
        result += ">"
        return result


def change_to_observer_relative(observer_pid, player_count, item):
    if item.move.move_type == "Deal":
        assert item.player < 0 and item.deal_to_player >= 0
        item.deal_to_player = (item.deal_to_player - observer_pid + player_count) % player_count
        if item.deal_to_player == 0:
            # Hide cards dealt to observer.
            item.move = HanabiMove("Deal", -1, -1, -1, -1)
    else:
        assert item.player >= 0
        item.player = (item.player - observer_pid + player_count) % player_count