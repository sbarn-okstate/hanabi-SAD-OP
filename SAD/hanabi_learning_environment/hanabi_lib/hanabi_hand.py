from typing import List, Optional


class ValueKnowledge:
    """
    Represents knowledge about an unknown integer in the range [0, value_range - 1].
    Tracks hints that either reveal the exact value or eliminate possibilities.
    """
    def __init__(self, value_range: int):
        assert value_range > 0, "Value range must be greater than zero."
        self.value = -1  # -1 indicates the value is not directly hinted
        self.value_plausible = [True] * value_range  # All values are plausible initially

    def range(self) -> int:
        """Returns the range of possible values."""
        return len(self.value_plausible)

    def value_hinted(self) -> bool:
        """Returns True if the exact value is known."""
        return self.value >= 0

    def is_plausible(self, value: int) -> bool:
        """Returns True if the value is still plausible."""
        return self.value_plausible[value]

    def apply_is_value_hint(self, value: int):
        """Records a hint that the value is exactly this."""
        assert 0 <= value < len(self.value_plausible)
        assert self.value < 0 or self.value == value
        assert self.value_plausible[value]
        self.value = value
        self.value_plausible = [False] * len(self.value_plausible)
        self.value_plausible[value] = True

    def apply_is_not_value_hint(self, value: int):
        """Records a hint that the value is not this."""
        assert 0 <= value < len(self.value_plausible)
        assert self.value < 0 or self.value != value
        self.value_plausible[value] = False


class CardKnowledge:
    """
    Tracks hinted knowledge about the color and rank of a card.
    """
    def __init__(self, num_colors: int, num_ranks: int):
        self.color = ValueKnowledge(num_colors)
        self.rank = ValueKnowledge(num_ranks)

    def num_colors(self) -> int:
        return self.color.range()

    def color_hinted(self) -> bool:
        return self.color.value_hinted()

    def color(self) -> int:
        return self.color.value

    def color_plausible(self, color: int) -> bool:
        return self.color.is_plausible(color)

    def apply_is_color_hint(self, color: int):
        self.color.apply_is_value_hint(color)

    def apply_is_not_color_hint(self, color: int):
        self.color.apply_is_not_value_hint(color)

    def num_ranks(self) -> int:
        return self.rank.range()

    def rank_hinted(self) -> bool:
        return self.rank.value_hinted()

    def rank(self) -> int:
        return self.rank.value

    def rank_plausible(self, rank: int) -> bool:
        return self.rank.is_plausible(rank)

    def apply_is_rank_hint(self, rank: int):
        self.rank.apply_is_value_hint(rank)

    def apply_is_not_rank_hint(self, rank: int):
        self.rank.apply_is_not_value_hint(rank)

    def __str__(self) -> str:
        color_str = chr(65 + self.color()) if self.color_hinted() else "X"
        rank_str = str(self.rank() + 1) if self.rank_hinted() else "X"
        plausible_colors = "".join(chr(65 + i) for i in range(self.num_colors()) if self.color_plausible(i))
        plausible_ranks = "".join(str(i + 1) for i in range(self.num_ranks()) if self.rank_plausible(i))
        return f"{color_str}{rank_str}|{plausible_colors}{plausible_ranks}"


class HanabiHand:
    """
    Represents a player's hand in Hanabi and tracks knowledge about cards.
    """
    def __init__(self):
        self.cards = []  # List of HanabiCard objects
        self.card_knowledge = []  # List of CardKnowledge objects

    def add_card(self, card, initial_knowledge: CardKnowledge):
        assert card.is_valid(), "Card must be valid."
        self.cards.append(card)
        self.card_knowledge.append(initial_knowledge)

    def remove_from_hand(self, card_index: int, discard_pile: Optional[List] = None):
        if discard_pile is not None:
            discard_pile.append(self.cards[card_index])
        del self.cards[card_index]
        del self.card_knowledge[card_index]

    def reveal_color(self, color: int) -> int:
        mask = 0
        assert len(self.cards) <= 8, "More than 8 cards is not supported."
        for i, card in enumerate(self.cards):
            if card.color() == color:
                if not self.card_knowledge[i].color_hinted():
                    mask |= (1 << i)
                self.card_knowledge[i].apply_is_color_hint(color)
            else:
                self.card_knowledge[i].apply_is_not_color_hint(color)
        return mask

    def reveal_rank(self, rank: int) -> int:
        mask = 0
        assert len(self.cards) <= 8, "More than 8 cards is not supported."
        for i, card in enumerate(self.cards):
            if card.rank() == rank:
                if not self.card_knowledge[i].rank_hinted():
                    mask |= (1 << i)
                self.card_knowledge[i].apply_is_rank_hint(rank)
            else:
                self.card_knowledge[i].apply_is_not_rank_hint(rank)
        return mask

    def __str__(self) -> str:
        assert len(self.cards) == len(self.card_knowledge)
        return "\n".join(f"{card} || {knowledge}" for card, knowledge in zip(self.cards, self.card_knowledge))
