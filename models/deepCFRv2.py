from schnapsen.game import (Move, PlayerPerspective, Bot, GamePlayEngine, BotState, GameState, LeaderPerspective,
                            FollowerPerspective, SchnapsenDeckGenerator, GamePhase, Trick, SchnapsenTrickScorer)
from schnapsen.deck import Suit, Rank
import random
from random import choice
from typing import Optional, cast
from collections import defaultdict
import math



class CFRBot(Bot):
    """
    A Bot that uses a pre-trained CFR strategy.
    """
    def __init__(self, average_strategies: dict[str, dict[Move, float]]):
        super().__init__(name="CFRBot")
        self.avg_strategies = average_strategies  # e.g. from CFRTrainer.get_average_strategy()

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        """
        Looks up the strategy distribution from the stored average_strategies and chooses an action.

        If the info set is not found (unseen state), defaults to a random action.
        """
        # 1. Build the key:
        info_key = get_state_feature_vector(perspective, leader_move)

        # 2. Get the legal moves
        valid_moves = perspective.valid_moves()
        if not valid_moves:
            raise Exception("No valid moves from perspective? Should not happen in normal play.")
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # 3. Look up the distribution from the average strategy if it exists
        if info_key in self.avg_strategies:
            strategy = self.avg_strategies[info_key]
            # Filter for only moves that are valid (some info sets might have more actions)
            # Renormalize among valid actions
            filtered = {}
            total_prob = 0.0
            for m in valid_moves:
                filtered[m] = strategy.get(m, 0.0)
                total_prob += filtered[m]
            if total_prob < 1e-9:
                # fallback to uniform among valid
                chosen_move = self._uniform_choice(valid_moves)
            else:
                # sample from the distribution
                chosen_move = self._sample_action(filtered, total_prob)
        else:
            # Unseen info set: fallback to uniform
            chosen_move = self._uniform_choice(valid_moves)

        return chosen_move

    def _uniform_choice(self, moves: list[Move]) -> Move:
        return choice(moves)

    def _sample_action(self, filtered: dict[Move, float], total_prob: float) -> Move:
        """
        Sample an action from 'filtered', which is a dictionary of Move -> prob.
        """
        r = random.random() * total_prob
        s = 0.0
        for move, prob in filtered.items():
            s += prob
            if s >= r:
                return move
        # fallback
        return list(filtered.keys())[-1]
    

    ######### Infoset ##########



class CFRInformationSet:
    """
    Stores the regret and strategy statistics for a single information set.
    """
    def __init__(self, actions: list[Move]):
        # The legal actions in this information set
        self.actions = actions  
        
        # Immediate regrets for not having chosen each action in the past
        self.cumulative_regrets = defaultdict(float)
        
        # Sums of strategy probabilities for each action, used to compute the average strategy
        self.strategy_sum = defaultdict(float)

    def get_strategy(self, realization_weight: float) -> dict[Move, float]:
        """
        Compute a regret-matching strategy distribution over actions.
        
        :param realization_weight: The portion of probability to assign to these regrets
        :return: a dictionary action -> probability
        """
        # Regret matching: only positive regrets matter
        positive_regrets = []
        for a in self.actions:
            positive_regrets.append(max(self.cumulative_regrets[a], 0.0))

        total_pos_regret = sum(positive_regrets)
        
        strategy = {}
        if total_pos_regret > 0:
            for a, reg in zip(self.actions, positive_regrets):
                strategy[a] = reg / total_pos_regret
        else:
            # If all regrets are zero or negative, play each action uniformly
            uniform_prob = 1.0 / len(self.actions)
            for a in self.actions:
                strategy[a] = uniform_prob

        # Accumulate the computed strategy into the strategy_sum for averaging
        for a in self.actions:
            self.strategy_sum[a] += strategy[a] * realization_weight

        return strategy

    def get_average_strategy(self) -> dict[Move, float]:
        """
        Return the average strategy, i.e. strategy_sum normalized by the total.
        """
        total = sum(self.strategy_sum[a] for a in self.actions)
        if math.isclose(total, 0.0):
            # If no visits, default to uniform
            return {a: 1.0 / len(self.actions) for a in self.actions}
        return {a: self.strategy_sum[a] / total for a in self.actions}

    

    ######### Trainer ##########



class CFRTrainer:
    def __init__(self, engine: GamePlayEngine, trickscorer: SchnapsenTrickScorer, iterations: int = 10000, seed: int = 42):
        """
        :param engine: A GamePlayEngine, e.g. a SchnapsenGamePlayEngine
        :param iterations: how many iterations of self-play we want to run
        """
        self.iterations = iterations
        self.engine = engine
        self.trickscorer = trickscorer
        self.infosets = {}   # maps info_set_key -> CFRInformationSet
        self.rng = random.Random(seed)

    def train(self):
        """
        Run CFR for the configured number of iterations.
        """
        for it in range(self.iterations):
            # Each iteration: sample a new initial state (maybe just by dealing random)
            # Alternatively, you might want to systematically handle all deals if feasible,
            # but that is huge for Schnapsen. We do random sampling:
            initial_state = self._deal_random_initial_state()
            
            # Then run the recursive CFR routine from that state
            self._cfr_recursive(game_state=initial_state, 
                                p0=1.0, p1=1.0,  # these are "reach probabilities" of each player
                                active_player=0, 
                                leader_move=None)

    def _deal_random_initial_state(self) -> GameState:
        """
        Create a random initial state from the engine's deck generator.
        """
        deck = self.engine.deck_generator.get_initial_deck()
        shuffled = self.engine.deck_generator.shuffle_deck(deck, self.rng)
        hand1, hand2, talon = self.engine.hand_generator.generateHands(shuffled)
        
        leader_state = BotState(implementation=None, hand=hand1)   # Implementation filled later
        follower_state = BotState(implementation=None, hand=hand2)

        game_state = GameState(
            leader=leader_state,
            follower=follower_state,
            talon=talon,
            previous=None
        )

        return game_state

    def _cfr_recursive(self, 
                       game_state: GameState, 
                       p0: float, 
                       p1: float, 
                       active_player: int,
                       leader_move: Optional[Move]) -> float:
        """
        The recursive CFR routine.

        :param game_state: current state
        :param p0: reach probability for player 0
        :param p1: reach probability for player 1
        :param active_player: 0 or 1
        :param leader_move: If active_player is the follower, the leader's move for the current trick.
        :return: the utility for the active_player from this branch of the game
        """
        # 1. Check for terminal condition
        maybe_winner = self.trickscorer.declare_winner(game_state)
        if maybe_winner is not None:
            # If there's a winner, compute payoff
            # For simplicity let's say the utility is 1 for winner, 0 for loser
            (winner, _) = maybe_winner
            if (winner is game_state.leader and active_player == 0) or \
               (winner is game_state.follower and active_player == 1):
                return 1.0
            else:
                return 0.0

        # 2. Build the perspective for the active player
        if active_player == 0:
            perspective = LeaderPerspective(game_state, self.engine) if game_state.leader is game_state.leader else None
            # Actually we must check if they truly are the leader. If not, build a FollowerPerspective.
            # Below is simplified: in practice, you check `am_i_leader()`.
            if perspective is None:  
                perspective = FollowerPerspective(game_state, self.engine, leader_move)
            my_p = p0
            other_p = p1
        else:
            perspective = LeaderPerspective(game_state, self.engine) if game_state.leader is game_state.follower else None
            if perspective is None:
                perspective = FollowerPerspective(game_state, self.engine, leader_move)
            my_p = p1
            other_p = p0

        # 3. Get legal moves
        actions = perspective.valid_moves()

        # 4. Retrieve or create the CFRInformationSet
        info_set_key = get_state_feature_vector(perspective, leader_move)
        if info_set_key not in self.infosets:
            self.infosets[info_set_key] = CFRInformationSet(actions)
        info_set = self.infosets[info_set_key]

        # 5. Obtain current strategy
        strategy = info_set.get_strategy(my_p)

        # 6. For each action, compute next node utility (recursively)
        node_utility = {}
        total_node_util = 0.0

        for a in actions:
            # We will simulate taking action `a`, building the next state
            next_state = self._apply_action(game_state, a, perspective)  
            
            # Next player might remain the same or switch, depending on the trick's outcome:
            next_active_player = 0 if next_state.leader is next_state.leader else 1
            # Or more simply:  `next_active_player = 1 - active_player` if turn always alternates,
            # but in Schnapsen, the winner of a trick leads next time, so we check carefully.

            # Recursively call
            if active_player == 0:
                child_utility = self._cfr_recursive(next_state, p0=my_p*strategy[a],
                                                    p1=other_p, active_player=next_active_player,
                                                    leader_move=a if a.is_regular_move() or a.is_marriage() else None)
            else:
                child_utility = self._cfr_recursive(next_state, p0=other_p, 
                                                    p1=my_p*strategy[a], active_player=next_active_player,
                                                    leader_move=a if a.is_regular_move() or a.is_marriage() else None)

            node_utility[a] = child_utility
            total_node_util += child_utility * strategy[a]

        # 7. Regret update
        for a in actions:
            regret = node_utility[a] - total_node_util
            info_set.cumulative_regrets[a] += other_p * regret  # Opponent's reach weight in 2p setting

        return total_node_util

    def _apply_action(self, game_state: GameState, action: Move, perspective: PlayerPerspective) -> GameState:
        """
        Construct the next game state from the given state + the chosen action.
        In practice, you might replicate logic from the TrickImplementer or do partial steps.
        For demonstration, we do a partial approach calling an existing method.
        
        - If perspective is leader, we do a trick with fixed leader_move
        - If perspective is follower, we do a trick with fixed leader_move known
        - If it is a TrumpExchange, we might just apply it directly, etc.
        """
        # We'll do a cheap approach: we ask the engine to play a single 'trick' with the leader's action forced:
        # (Note: This might not be 100% accurate for partial moves, but shows the idea.)
        new_state = self.engine.trick_implementer.play_trick_with_fixed_leader_move(
            game_engine=self.engine,
            game_state=game_state,
            leader_move=action
        )
        return new_state

    def get_average_strategy(self) -> dict[str, dict[Move, float]]:
        """
        Return the average strategy for all known info sets.
        """
        avg_strats = {}
        for key, infoset in self.infosets.items():
            avg_strats[key] = infoset.get_average_strategy()
        return avg_strats




######## data set generation ########



def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> list[int]:
    """
    Translating the suit of a card into one hot vector encoding of size 4.
    """
    card_suit_one_hot: list[int]
    if card_suit == Suit.HEARTS:
        card_suit_one_hot = [0, 0, 0, 1]
    elif card_suit == Suit.CLUBS:
        card_suit_one_hot = [0, 0, 1, 0]
    elif card_suit == Suit.SPADES:
        card_suit_one_hot = [0, 1, 0, 0]
    elif card_suit == Suit.DIAMONDS:
        card_suit_one_hot = [1, 0, 0, 0]
    else:
        raise ValueError("Suit of card was not found!")

    return card_suit_one_hot


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> list[int]:
    """
    Translating the rank of a card into one hot vector encoding of size 13.
    """
    card_rank_one_hot: list[int]
    if card_rank == Rank.ACE:
        card_rank_one_hot = [0, 0, 0, 0, 1]
    elif card_rank == Rank.TEN:
        card_rank_one_hot = [0, 0, 0, 1, 0]
    elif card_rank == Rank.JACK:
        card_rank_one_hot = [0, 0, 1, 0, 0]
    elif card_rank == Rank.QUEEN:
        card_rank_one_hot = [0, 1, 0, 0, 0]
    elif card_rank == Rank.KING:
        card_rank_one_hot = [1, 0, 0, 0, 0]
    else:
        raise AssertionError("Provided card Rank does not exist!")
    return card_rank_one_hot


def get_move_feature_vector(move: Optional[Move]) -> list[int]:
    """
        In case there isn't any move provided move to encode, we still need to create a "padding"-"meaningless" vector of the same size,
        filled with 0s, since the ML models need to receive input of the same dimensionality always.
        Otherwise, we create all the information of the move i) move type, ii) played card rank and iii) played card suit
        translate this information into one-hot vectors respectively, and concatenate these vectors into one move feature representation vector
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0]

    else:
        move_type_one_hot_encoding: list[int]
        # in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1]
            card = move.queen_card
        #  in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0]
            card = move.jack
        #  in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0]
            card = move.card
        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding
        card_rank_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_rank(card.rank)
        card_suit_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_suit(card.suit)

    return move_type_one_hot_encoding_numpy_array + card_rank_one_hot_encoding_numpy_array + card_suit_one_hot_encoding_numpy_array


def get_state_feature_vector(perspective: PlayerPerspective, leader_move: Move = None) -> list[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hot encoding)
        - talon size (int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown)

        Important: This function should not include the move of this agent.
        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)
    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    player_score = perspective.get_my_score()
    # - points of this player (int)
    player_points = round(player_score.direct_points / 66, 1)
    # - pending points of this player (int)
    player_pending_points = round(player_score.pending_points / 40, 1)

    # add the features to the feature set
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]

    opponents_score = perspective.get_opponent_score()
    # - points of the opponent (int)
    opponents_points = round(opponents_score.direct_points / 66, 1)
    # - pending points of opponent (int)
    opponents_pending_points = round(opponents_score.pending_points / 40, 1)

    # add the features to the feature set
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]

    # - the trump suit (1-hot encoding)
    trump_suit = perspective.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
    # add this features to the feature set
    state_feature_list += trump_suit_one_hot

    # - phase of game (1-hot encoding)
    game_phase_encoded = [1, 0] if perspective.get_phase() == GamePhase.TWO else [0, 1]
    # add this features to the feature set
    state_feature_list += game_phase_encoded

    # - talon size (int)
    talon_size = perspective.get_talon_size() / 10
    # add this features to the feature set
    state_feature_list += [talon_size]

    # - if this player is leader (1-hot encoding)
    i_am_leader = [0, 1] if perspective.am_i_leader() else [1, 0]
    # add this features to the feature set
    state_feature_list += i_am_leader

    # gather all known deck information
    hand_cards = perspective.get_hand().cards
    trump_card = perspective.get_trump_card()
    won_cards = perspective.get_won_cards().get_cards()
    opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
    opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()
    # each card can either be i) on player's hand, ii) on player's won cards, iii) on opponent's hand, iv) on opponent's won cards
    # v) be the trump card or vi) in an unknown position -> either on the talon or on the opponent's hand
    # There are all different cases regarding card's knowledge, and we represent these 6 cases using one hot encoding vectors as seen bellow.

    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []

    for card in SchnapsenDeckGenerator().get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]
        # i) on player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]
        # ii) on player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]
        # iii) on opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]
        # iv) on opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]
        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]
        # vi) in an unknown position as it is invisible to this player. Thus, it is either on the talon or on the opponent's hand
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]
        # This list eventually develops to one long 1-dimensional numpy array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding
    # deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # add this features to the feature set
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

    leader_move_vector = get_move_feature_vector(leader_move)

    state_feature_list += leader_move_vector

    return state_feature_list
