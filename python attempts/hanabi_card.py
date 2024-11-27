import util

class HanabiCard:
    def __init__(self, color=-1, rank=-1):
        """
        Initializes a HanabiCard instance.
        :param color: The color of the card (0-indexed). Defaults to -1 (invalid).
        :param rank: The rank of the card (0-indexed). Defaults to -1 (invalid).
        """
        self.color = color
        self.rank = rank

    def __eq__(self, other_card):
        """
        Compares two HanabiCard objects for equality.
        :param other_card: The other HanabiCard to compare.
        :return: True if both cards have the same color and rank, False otherwise.
        """
        return self.color == other_card.color and self.rank == other_card.rank

    def is_valid(self):
        """
        Checks if the card is valid.
        :return: True if the card has a valid color and rank, False otherwise.
        """
        return self.color >= 0 and self.rank >= 0

    def to_string(self):
        """
        Returns a string representation of the card.
        :return: "XX" if the card is invalid, otherwise a string containing the
                 card's color and rank.
        """
        if not self.is_valid():
            return "XX"

        return f"{self.color_index_to_char(self.color)}{self.rank_index_to_char(self.rank)}"