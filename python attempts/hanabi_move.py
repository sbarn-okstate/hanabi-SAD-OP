import util

class HanabiMove:
    class Type:
        INVALID = "Invalid"
        PLAY = "Play"
        DISCARD = "Discard"
        REVEAL_COLOR = "RevealColor"
        REVEAL_RANK = "RevealRank"
        DEAL = "Deal"

    def __init__(self, move_type: str, card_index: int, target_offset: int, color: int, rank: int):
        self.move_type = move_type
        self.card_index = card_index
        self.target_offset = target_offset
        self.color = color
        self.rank = rank

    def __eq__(self, other):
        if self.move_type != other.move_type:
            return False
        if self.move_type in (self.Type.PLAY, self.Type.DISCARD):
            return self.card_index == other.card_index
        if self.move_type == self.Type.REVEAL_COLOR:
            return self.target_offset == other.target_offset and self.color == other.color
        if self.move_type == self.Type.REVEAL_RANK:
            return self.target_offset == other.target_offset and self.rank == other.rank
        if self.move_type == self.Type.DEAL:
            return self.color == other.color and self.rank == other.rank
        return True

    def is_valid(self):
        return self.move_type != self.Type.INVALID

    def __str__(self):
        if self.move_type == self.Type.PLAY:
            return f"(Play {self.card_index})"
        if self.move_type == self.Type.DISCARD:
            return f"(Discard {self.card_index})"
        if self.move_type == self.Type.REVEAL_COLOR:
            return f"(Reveal player +{self.target_offset} color {color_index_to_char(self.color)})"
        if self.move_type == self.Type.REVEAL_RANK:
            return f"(Reveal player +{self.target_offset} rank {rank_index_to_char(self.rank)})"
        if self.move_type == self.Type.DEAL:
            if self.color >= 0:
                return f"(Deal {color_index_to_char(self.color)}{rank_index_to_char(self.rank)})"
            else:
                return "(Deal XX)"
        return "(INVALID)"

    # Getters
    def get_move_type(self):
        return self.move_type

    def get_card_index(self):
        return self.card_index

    def get_target_offset(self):
        return self.target_offset

    def get_color(self):
        return self.color

    def get_rank(self):
        return self.rank