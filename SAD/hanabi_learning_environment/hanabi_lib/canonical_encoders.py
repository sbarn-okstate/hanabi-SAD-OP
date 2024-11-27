import numpy as np
import tensorflow as tf

class CanonicalObservationEncoder:
    def __init__(self, parent_game):
        self.parent_game = parent_game

    def shape(self):
        l = (hands_section_length(self.parent_game) +
             board_section_length(self.parent_game) +
             discard_section_length(self.parent_game) +
             last_action_section_length(self.parent_game) +
             (0 if self.parent_game.observation_type() == HanabiGame.kMinimal
              else card_knowledge_section_length(self.parent_game)))
        return [l]

    def encode_last_action(self, obs):
        encoding = [0] * last_action_section_length(self.parent_game)
        offset = 0
        offset += encode_last_action(self.parent_game, obs, offset, encoding)
        assert offset == len(encoding)
        return encoding

    def encode_v0_belief(self, obs):
        encoding = [0] * card_knowledge_section_length(self.parent_game)
        length = encode_v0_belief(self.parent_game, obs, 0, encoding)
        assert length == len(encoding)
        belief = extract_belief(encoding, self.parent_game)
        return belief

    def encode_v1_belief(self, obs):
        encoding = [0] * card_knowledge_section_length(self.parent_game)
        length = encode_v1_belief(self.parent_game, obs, 0, encoding)
        assert length == len(encoding)
        belief = extract_belief(encoding, self.parent_game)
        return belief

    def encode_hand_mask(self, obs):
        encoding = [0] * card_knowledge_section_length(self.parent_game)
        encode_card_knowledge(self.parent_game, obs, 0, encoding)
        hand_mask = extract_belief(encoding, self.parent_game)
        return hand_mask

    def encode_card_count(self, obs):
        encoding = []
        card_count = compute_card_count(self.parent_game, obs)
        for count in card_count:
            encoding.append(float(count))
        return encoding

    def encode_own_hand(self, obs):
        # Here, assuming 5 cards with 3 bits per card (hard-coded)
        length = 5 * 3
        encoding = [0] * length
        offset = encode_own_hand(self.parent_game, obs, 0, encoding)
        assert offset <= length
        return encoding

    def encode(self, obs, show_own_cards):
        # Make an empty bit string of the proper size
        encoding = [0] * flat_length(self.shape())
        offset = 0

        offset += encode_hands(self.parent_game, obs, offset, encoding, show_own_cards)
        offset += encode_board(self.parent_game, obs, offset, encoding)
        offset += encode_discards(self.parent_game, obs, offset, encoding)
        offset += encode_last_action(self.parent_game, obs, offset, encoding)
        
        if self.parent_game.observation_type() != HanabiGame.kMinimal:
            offset += encode_v0_belief(self.parent_game, obs, offset, encoding)
        
        assert offset == len(encoding)
        return encoding
        
    def type(self):
        """
        Returns the type of the encoder (Canonical).
        """
        return 'Canonical'


def flat_length(shape):
    """Calculate the flat length of a shape (i.e., total number of elements)."""
    return np.prod(shape)

def get_last_non_deal_move(past_moves):
    """Get the last non-deal move from a list of past moves."""
    for item in reversed(past_moves):
        if item.move.move_type != 'Deal':  # Assuming move_type is a string like 'Deal'
            return item
    return None

def bits_per_card(game):
    """Calculate the number of bits needed to represent a card."""
    return game.num_colors * game.num_ranks

def card_index(color, rank, num_ranks):
    """Calculate the one-hot index for a card based on its color and rank."""
    return color * num_ranks + rank

def hands_section_length(game):
    """Calculate the total length of the hands section in the encoding."""
    return game.num_players * game.hand_size * bits_per_card(game) + game.num_players

def encode_hands(game, obs, start_offset, encoding, show_own_cards):
    bits_per_card = bits_per_card(game)  # Assuming this function is available
    num_ranks = game.num_ranks  # Assuming the game class has these attributes
    num_players = game.num_players
    hand_size = game.hand_size  # Assuming this attribute is available

    offset = start_offset
    hands = obs.hands  # List of hands, each is a player's hand (HanabiHand)
    assert len(hands) == num_players

    for player in range(num_players):
        cards = hands[player].cards  # Get the list of cards for the player
        num_cards = 0

        for card in cards:
            # Ensure the card is valid and has the expected properties
            assert 0 <= card.color < game.num_colors  # Card color within valid range
            assert 0 <= card.rank < num_ranks  # Card rank within valid range
            if player == 0:
                if show_own_cards:
                    assert card.is_valid()  # Assuming card has an is_valid() method
                    # Encoding the card at the correct index based on its color and rank
                    encoding[offset + card_index(card.color, card.rank, num_ranks)] = 1
                else:
                    assert not card.is_valid()  # Card should be invalid
            else:
                assert card.is_valid()
                encoding[offset + card_index(card.color, card.rank, num_ranks)] = 1

            num_cards += 1
            offset += bits_per_card

        # For players with fewer cards than the hand size, skip the absent cards
        if num_cards < hand_size:
            offset += (hand_size - num_cards) * bits_per_card

    # Set a bit for players missing cards in their hand
    for player in range(num_players):
        if len(hands[player].cards) < game.hand_size:
            encoding[offset + player] = 1
    offset += num_players

    # Ensure the correct length of the encoding section
    assert offset - start_offset == hands_section_length(game)  # Assuming this function is defined
    return offset - start_offset

def encode_own_hand(self, obs):
    """
    Encodes the player's own hand.
    This function will encode the player's hand state for each card.
    """
    bits_per_card = 3  # BitsPerCard(game)
    num_ranks = game.num_ranks  # Assuming the game class has these attributes

    offset = start_offset
    hands = obs.hands  # Assuming the observation has hands data as a list of player hands
    player = 0  # We are encoding for the first player (index 0)
    cards = hands[player].cards  # Get the player's cards (assuming 'cards' is a list of card objects)

    fireworks = obs.fireworks  # Fireworks state (one integer per color indicating the highest rank played)

    for card in cards:
        # Ensure the card is valid and has the expected properties
        assert 0 <= card.color < game.num_colors  # Card color within valid range
        assert 0 <= card.rank < num_ranks  # Card rank within valid range
        assert card.is_valid()  # Assuming card has an is_valid() method

        firework = fireworks[card.color]
        
        # Encoding the card based on its rank and the firework state for that color
        if card.rank == firework:
            encoding[offset] = 1
        elif card.rank < firework:
            encoding[offset + 1] = 1
        else:
            encoding[offset + 2] = 1

        offset += bits_per_card

    return offset

def board_section_length(game):
    """Calculate the length of the board section for the encoding."""
    return (game.max_deck_size - game.num_players * game.hand_size +
            game.num_colors * game.num_ranks +
            game.max_information_tokens +
            game.max_life_tokens)

def encode_board(game, obs, start_offset, encoding):
    """Encode the board state, including deck size, fireworks, info tokens, and life tokens."""
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    num_players = game.num_players
    hand_size = game.hand_size
    max_deck_size = game.max_deck_size

    offset = start_offset

    # Encode the deck size
    for i in range(obs.deck_size):
        encoding[offset + i] = 1
    offset += (max_deck_size - hand_size * num_players)

    # Encode fireworks
    fireworks = obs.fireworks
    for c in range(num_colors):
        if fireworks[c] > 0:
            encoding[offset + fireworks[c] - 1] = 1
        offset += num_ranks

    # Encode information tokens
    assert 0 <= obs.information_tokens <= game.max_information_tokens
    for i in range(obs.information_tokens):
        encoding[offset + i] = 1
    offset += game.max_information_tokens

    # Encode life tokens
    assert 0 <= obs.life_tokens <= game.max_life_tokens
    for i in range(obs.life_tokens):
        encoding[offset + i] = 1
    offset += game.max_life_tokens

    assert offset - start_offset == board_section_length(game)
    return offset - start_offset

def discard_section_length(game):
    """Calculate the length of the discard section for the encoding."""
    return game.max_deck_size

def encode_discards(game, obs, start_offset, encoding):
    """Encode the discard pile state using a thermometer representation."""
    num_colors = game.num_colors
    num_ranks = game.num_ranks

    offset = start_offset
    discard_counts = [0] * (num_colors * num_ranks)
    
    # Count the number of discarded cards for each color and rank
    for card in obs.discard_pile:
        discard_counts[card.color * num_ranks + card.rank] += 1

    # Encode the discard pile
    for c in range(num_colors):
        for r in range(num_ranks):
            num_discarded = discard_counts[c * num_ranks + r]
            for i in range(num_discarded):
                encoding[offset + i] = 1
            offset += game.number_card_instances(c, r)

    assert offset - start_offset == discard_section_length(game)
    return offset - start_offset

def encode_last_action(game, obs, start_offset, encoding):
    """Encode the last action taken in the game."""
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    num_players = game.num_players
    hand_size = game.hand_size

    offset = start_offset
    last_move = get_last_non_deal_move(obs.last_moves)
    
    if last_move is None:
        offset += last_action_section_length(game)
    else:
        last_move_type = last_move.move.move_type

        # Player ID
        encoding[offset + last_move.player] = 1
        offset += num_players

        # Move type
        if last_move_type == HanabiMove.Type.kPlay:
            encoding[offset] = 1
        elif last_move_type == HanabiMove.Type.kDiscard:
            encoding[offset + 1] = 1
        elif last_move_type == HanabiMove.Type.kRevealColor:
            encoding[offset + 2] = 1
        elif last_move_type == HanabiMove.Type.kRevealRank:
            encoding[offset + 3] = 1
        else:
            raise ValueError("Unknown move type")
        offset += 4

        # Target player (if hint action)
        if last_move_type == HanabiMove.Type.kRevealColor or last_move_type == HanabiMove.Type.kRevealRank:
            observer_relative_target = (last_move.player + last_move.move.target_offset) % num_players
            encoding[offset + observer_relative_target] = 1
        offset += num_players

        # Color (if hint action)
        if last_move_type == HanabiMove.Type.kRevealColor:
            encoding[offset + last_move.move.color] = 1
        offset += num_colors

        # Rank (if hint action)
        if last_move_type == HanabiMove.Type.kRevealRank:
            encoding[offset + last_move.move.rank] = 1
        offset += num_ranks

        # Outcome (if hint action)
        if last_move_type == HanabiMove.Type.kRevealColor or last_move_type == HanabiMove.Type.kRevealRank:
            for i in range(hand_size):
                if (last_move.reveal_bitmask & (1 << i)) > 0:
                    encoding[offset + i] = 1
            offset += hand_size

        # Position (if play or discard action)
        if last_move_type == HanabiMove.Type.kPlay or last_move_type == HanabiMove.Type.kDiscard:
            encoding[offset + last_move.move.card_index] = 1
        offset += hand_size

        # Card (if play or discard action)
        if last_move_type == HanabiMove.Type.kPlay or last_move_type == HanabiMove.Type.kDiscard:
            assert last_move.color >= 0
            assert last_move.rank >= 0
            encoding[offset + card_index(last_move.color, last_move.rank, num_ranks)] = 1
        offset += bits_per_card(game)

        # Success and/or added information token (if play action)
        if last_move_type == HanabiMove.Type.kPlay:
            if last_move.scored:
                encoding[offset] = 1
            if last_move.information_token:
                encoding[offset + 1] = 1
            offset += 2

    assert offset - start_offset == last_action_section_length(game)
    return offset - start_offset


def card_knowledge_section_length(game):
    """Calculate the length of the card knowledge section."""
    return game.num_players * game.hand_size * (bits_per_card(game) + game.num_colors + game.num_ranks)


def encode_card_knowledge(game, obs, start_offset, encoding):
    """Encode the common card knowledge for each card in each player's hand."""
    bits_per_card = bits_per_card(game)
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    num_players = game.num_players
    hand_size = game.hand_size

    offset = start_offset
    hands = obs.hands
    assert len(hands) == num_players
    
    for player in range(num_players):
        knowledge = hands[player].knowledge()
        num_cards = 0

        for card_knowledge in knowledge:
            # Add bits for plausible cards (color-major ordering)
            for color in range(num_colors):
                if card_knowledge.color_plausible(color):
                    for rank in range(num_ranks):
                        if card_knowledge.rank_plausible(rank):
                            encoding[offset + card_index(color, rank, num_ranks)] = 1
            offset += bits_per_card

            # Add bits for explicitly revealed colors and ranks
            if card_knowledge.color_hinted():
                encoding[offset + card_knowledge.color()] = 1
            offset += num_colors

            if card_knowledge.rank_hinted():
                encoding[offset + card_knowledge.rank()] = 1
            offset += num_ranks

            num_cards += 1

        # A player's hand may have fewer cards than the initial hand size.
        # Skip bits for the missing cards.
        if num_cards < hand_size:
            offset += (hand_size - num_cards) * (bits_per_card + num_colors + num_ranks)

    assert offset - start_offset == card_knowledge_section_length(game)
    return offset - start_offset

def compute_card_count(game, obs):
    """Compute the count of each card in the deck."""
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    card_count = [0] * (num_colors * num_ranks)
    total_count = 0

    # Full deck card count
    for color in range(num_colors):
        for rank in range(num_ranks):
            count = game.number_card_instances(color, rank)
            card_count[color * num_ranks + rank] = count
            total_count += count

    # Remove discard pile cards
    for card in obs.discard_pile:
        card_count[card.color * num_ranks + card.rank] -= 1
        total_count -= 1

    # Remove fireworks on board
    fireworks = obs.fireworks
    for c in range(num_colors):
        if fireworks[c] > 0:
            for rank in range(fireworks[c]):
                card_count[c * num_ranks + rank] -= 1
                total_count -= 1

    # Sanity check
    total_hand_size = sum(len(hand.cards) for hand in obs.hands)
    assert total_count == obs.deck_size + total_hand_size, \
        f"Size mismatch: {total_count} vs {obs.deck_size + total_hand_size}"

    return card_count

def encode_v0_belief(game, obs, start_offset, encoding, ret_card_count=None):
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    num_players = game.num_players
    hand_size = game.hand_size

    card_count = compute_card_count(game, obs)
    if ret_card_count is not None:
        ret_card_count = card_count

    # Card knowledge
    len_ = encode_card_knowledge(game, obs, start_offset, encoding)
    player_offset = len_ // num_players
    per_card_offset = len_ // (hand_size * num_players)
    assert per_card_offset == num_colors * num_ranks + num_colors + num_ranks

    hands = obs.hands
    for player_id in range(num_players):
        num_cards = len(hands[player_id].cards)
        for card_idx in range(num_cards):
            total = 0
            for i in range(num_colors * num_ranks):
                offset = start_offset + player_offset * player_id + card_idx * per_card_offset + i
                assert offset - start_offset < len_
                encoding[offset] *= card_count[i]
                total += encoding[offset]
            if total <= 0:
                print(len(hands[0].cards), len(hands[1].cards), "total = 0")
                assert False
            for i in range(num_colors * num_ranks):
                offset = start_offset + player_offset * player_id + card_idx * per_card_offset + i
                encoding[offset] /= total
    return len_

def encode_v1_belief(game, obs, start_offset, encoding):
    num_colors = game.num_colors
    num_ranks = game.num_ranks
    num_players = game.num_players
    hand_size = game.hand_size
    hands = obs.hands

    card_knowledge = [0] * card_knowledge_section_length(game)
    len_ = encode_card_knowledge(game, obs, 0, card_knowledge)
    assert len_ == len(card_knowledge)

    v0_belief = list(card_knowledge)
    card_count = []
    len_ = encode_v0_belief_(game, obs, 0, v0_belief, card_count)
    assert len_ == len(card_knowledge)

    player_offset = len_ // num_players
    per_card_offset = len_ // (hand_size * num_players)
    assert per_card_offset == num_colors * num_ranks + num_colors + num_ranks

    v1_belief = list(v0_belief)
    new_v1_belief = list(v1_belief)
    total_cards = [0] * len(card_count)

    for step in range(100):
        # Compute total card remaining
        for i in range(num_colors * num_ranks):
            total_cards[i] = card_count[i]
            for player_id in range(num_players):
                num_cards = len(hands[player_id].cards)
                for card_idx in range(num_cards):
                    offset = player_offset * player_id + card_idx * per_card_offset + i
                    total_cards[i] -= v1_belief[offset]

        # Compute new belief
        for player_id in range(num_players):
            num_cards = len(hands[player_id].cards)
            for card_idx in range(num_cards):
                base_offset = player_offset * player_id + card_idx * per_card_offset
                for i in range(num_colors * num_ranks):
                    offset = base_offset + i
                    p = max(total_cards[i] + v1_belief[offset], 0.0)
                    new_v1_belief[offset] = p * card_knowledge[offset]

        # Interpolate and normalize
        for player_id in range(num_players):
            num_cards = len(hands[player_id].cards)
            for card_idx in range(num_cards):
                total = 0
                base_offset = player_offset * player_id + card_idx * per_card_offset
                for i in range(num_colors * num_ranks):
                    offset = base_offset + i
                    v1_belief[offset] = (1 - 0.1) * v1_belief[offset] + 0.1 * new_v1_belief[offset]
                    total += v1_belief[offset]
                if total <= 0:
                    print("total = 0")
                    assert False
                for i in range(num_colors * num_ranks):
                    offset = base_offset + i
                    v1_belief[offset] /= total

    encoding[start_offset:start_offset + len(v1_belief)] = v1_belief
    return len(v1_belief)

def last_action_section_length(game):
    return (game.num_players +  # player id
            4 +                   # move types (play, discard, reverse color, reverse rank)
            game.num_players +    # target player id (if hint action)
            game.num_colors +     # color (if hint action)
            game.num_ranks +      # rank (if hint action)
            game.hand_size +      # outcome (if hint action)
            game.hand_size +      # position (if play action)
            bits_per_card(game) + # card (if play or discard action)
            2)                    # play (successful, added information token)


def extract_belief(encoding, game):
    bits_per_card = bits_per_card(game)
    num_colors = game.num_colors()
    num_ranks = game.num_ranks()
    num_players = game.num_players()
    hand_size = game.hand_size()
    encoding_sector_len = bits_per_card + num_colors + num_ranks
    assert encoding_sector_len * hand_size * num_players == len(encoding)

    belief = [0] * (num_players * hand_size * bits_per_card)
    for i in range(num_players):
        for j in range(hand_size):
            for k in range(bits_per_card):
                belief_offset = (i * hand_size + j) * bits_per_card + k
                encoding_offset = (i * hand_size + j) * encoding_sector_len + k
                belief[belief_offset] = encoding[encoding_offset]
    return belief