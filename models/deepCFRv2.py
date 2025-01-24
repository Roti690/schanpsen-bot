from schnapsen.game import (Move, PlayerPerspective, Bot, GamePlayEngine, BotState, GameState, SchnapsenDeckGenerator, GamePhase,
                            LeaderPerspective, FollowerPerspective, RegularMove, RegularTrick, Marriage, SchnapsenGamePlayEngine,
                            TrumpExchange, ExchangeTrick, Previous, PartialTrick, Trick)
from schnapsen.deck import Suit, Rank
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Union, cast

######### PlayingBot #########

class DeepCFRBot(Bot):
    def __init__(self, regret_model: nn.Module, max_actions: int = 10, name="DeepCFRBot"):
        super().__init__(name=name)
        self.regret_model = regret_model.eval()  # put in eval mode
        self.max_actions = max_actions

    def get_move(self, perspective, leader_move):
        valid_moves = perspective.valid_moves()
        if len(valid_moves) == 0:
            raise RuntimeError("No valid moves?")
        if len(valid_moves) == 1:
            return valid_moves[0]

        # 1) Convert the perspective to feature vector
        features = get_state_feature_vector(perspective, leader_move)

        # 2) NN forward pass
        with torch.no_grad():
            feats_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            predicted_regrets = self.regret_model(feats_t)[0]  # shape: [max_actions]
        # Only use first len(valid_moves)
        predicted_regrets = predicted_regrets[:len(valid_moves)].tolist()

        # 3) Regret matching
        clipped = [max(r, 0.0) for r in predicted_regrets]
        total = sum(clipped)
        if total < 1e-9:
            strategy = [1.0 / len(valid_moves)] * len(valid_moves)
        else:
            strategy = [val / total for val in clipped]

        # 4) Sample from strategy
        r = random.random()
        accum = 0.0
        for i, move in enumerate(valid_moves):
            accum += strategy[i]
            if r <= accum:
                return move
        return valid_moves[-1]

    ######### Regretmodel ##########


class RegretModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # pick your hidden sizes
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)   # output_dim = max # of actions you want to handle
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: [batch_size, output_dim]
    

    ######### Replaybuffer #########


class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state_features: list[float], regrets: list[float]):
        # regrets is a list[float] of length = output_dim (or at least up to max_actions).
        self.buffer.append((state_features, regrets))

    def sample(self, batch_size: int):
        actual_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_size)
        states, regrets = [], []
        for s, r in batch:
            states.append(s)
            regrets.append(r)
        # Convert to PyTorch tensors
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(regrets, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
    

    ######### Trainer ##########


class DeepCFRTrainer:
    def __init__(self, 
                 engine: GamePlayEngine,
                 trick_scorer,
                 iterations: int = 10_000,
                 seed: int = 42,
                 max_actions: int = 10
                 ) -> None:
        self.iterations = iterations
        self.engine = engine
        self.trick_scorer = trick_scorer
        self.rng = random.Random(seed)

        # 1) Create the neural net for regret approximation
        self.max_actions = max_actions
        # figure out what input_dim is from your get_state_feature_vector
        # e.g. count how many ints you return; or compute once at runtime
        self.input_dim = self._compute_input_dim()
        self.regret_model = RegretModel(input_dim=self.input_dim, output_dim=self.max_actions)

        # 2) Create optimizer/loss
        self.optimizer = optim.Adam(self.regret_model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        # 3) Replay buffer
        self.replay_buffer = ReplayBuffer()

    def _compute_input_dim(self) -> int:
        """
        Create a 'fake' perspective with minimal details and pass it through
        get_state_feature_vector to figure out dimension. Or, if you already
        know from your code that get_state_feature_vector always returns
        e.g. 120 + 15, etc., just hardcode that here.
        """
        
        return 145 

    def train(self, 
              deals_per_iteration: int = 1,
              batch_size: int = 256,
              epochs_per_iter: int = 1):
        """
        Main training loop. 
        `deals_per_iteration` = how many random deals to sample per iteration 
        `batch_size` = how many examples to train on each mini-batch
        `epochs_per_iter` = how many times to loop over the mini-batch for each iteration
        """

        for it in range(self.iterations):
            # 1) Generate states by self-play
            for _ in range(deals_per_iteration):
                initial_state = self._deal_random_initial_state()
                # Start recursion with active_player=0
                self._cfr_recursive(
                    game_state=initial_state,
                    p0=1.0,
                    p1=1.0,
                    active_player=0,
                    leader_move=None
                )

            # 2) Train the regret model with data in replay buffer
            for _ in range(epochs_per_iter):
                self._train_regret_network(batch_size)

    def _deal_random_initial_state(self) -> GameState:
        """
        Create a random initial state from the engine's deck generator and return it.
        """
        deck = self.engine.deck_generator.get_initial_deck()
        shuffled = self.engine.deck_generator.shuffle_deck(deck, self.rng)
        hand1, hand2, talon = self.engine.hand_generator.generateHands(shuffled)

        leader_state = BotState(implementation=_DummyBot(), hand=hand1)
        follower_state = BotState(implementation=_DummyBot(), hand=hand2)

        return GameState(
            leader=leader_state,
            follower=follower_state,
            talon=talon,
            previous=None
        )

    def _cfr_recursive(self,
                       game_state,
                       p0: float,
                       p1: float,
                       active_player: int,
                       leader_move) -> float:
        """
        The main recursive function that:
        1) Checks if terminal -> returns payoff
        2) Builds perspective -> gets valid actions
        3) Predicts regrets -> does regret matching
        4) Recursively updates children -> obtains child utilities
        5) Logs regrets in the replay buffer
        """

        # 1. Terminal check
        maybe_winner = self.trick_scorer.declare_winner(game_state)
        if maybe_winner is not None:
            (winner, _) = maybe_winner
            # compute payoff ...
            return 1.0 if ((winner is game_state.leader and active_player == 0) or
                        (winner is game_state.follower and active_player == 1)) else 0.0

        # 2) Build perspective purely by seat:
        if active_player == 0:
            # "Seat 0" is the leader seat
            perspective = LeaderPerspective(game_state, self.engine)
            my_p = p0
            other_p = p1
        else:
            # "Seat 1" is the follower seat
            perspective = FollowerPerspective(game_state, self.engine, leader_move)
            my_p = p1
            other_p = p0

        # 3. Get valid actions
        actions = perspective.valid_moves()
        if not actions:
            return 0.0

        # 4. Generate state features
        features = get_state_feature_vector(perspective, leader_move)

        # 5. Predict regrets via the neural net
        with torch.no_grad():
            feats_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            pred_regrets = self.regret_model(feats_t)[0]  # shape [max_actions]
        # Only use first len(actions) regrets
        pred_regrets = pred_regrets[:len(actions)].tolist()

        # 6. Do regret matching to form a strategy
        clipped = [max(r, 0.0) for r in pred_regrets]
        sum_pos = sum(clipped)
        if sum_pos < 1e-9:
            strategy = [1.0 / len(actions)] * len(actions)
        else:
            strategy = [r / sum_pos for r in clipped]

        # 7. Recursively compute child utilities
        node_utility = {}
        for i, a in enumerate(actions):
            next_state = self._apply_action(game_state, a, perspective)
            # who acts next
            # for Schnapsen, usually the winner of the trick is next leader, so adapt as needed.
            next_active_player = 0 if (next_state.leader is game_state.leader) else 1

            if active_player == 0:
                child_utility = self._cfr_recursive(
                    game_state=next_state,
                    p0=my_p * strategy[i],
                    p1=other_p,
                    active_player=next_active_player,
                    leader_move=a if a.is_regular_move() else None
                )
            else:
                child_utility = self._cfr_recursive(
                    game_state=next_state,
                    p0=other_p,
                    p1=my_p * strategy[i],
                    active_player=next_active_player,
                    leader_move=a if a.is_regular_move() else None
                )
            node_utility[a] = child_utility

        # 8. Compute total node util
        total_node_util = sum(node_utility[a] * strategy[i] for i, a in enumerate(actions))

        # 9. Compute regrets and push to replay buffer
        regrets = []
        for i, a in enumerate(actions):
            # standard 2p CFR uses "other_p" factor, but you can store raw regrets
            reg = node_utility[a] - total_node_util
            regrets.append(reg)

        # Pad regrets to self.max_actions
        regret_vector = [0.0] * self.max_actions
        for i in range(len(actions)):
            regret_vector[i] = regrets[i]

        self.replay_buffer.push(features, regret_vector)

        return total_node_util

    def _apply_action(engine, old_state: GameState, 
                  action: Move, perspective: PlayerPerspective) -> GameState:
        """
        A method to apply 'action' to 'old_state' and return the resulting new state.

        This version uses Approach 2: If the leader plays a marriage or a regular move,
        it stores a PartialTrick in `old_state.previous` rather than finalizing the trick
        immediately. The follower later detects this PartialTrick and finalizes the trick.
        """
        # 1) Make a new state so we do not mutate old_state directly
        next_state = old_state.copy_for_next()
        
        if perspective.am_i_leader():
            # LEADER's move
            if action.is_trump_exchange():
                # Apply the exchange immediately (fully resolves in one step).
                return apply_trump_exchange(engine=engine,
                                            old_state=old_state,
                                            exchange=cast(TrumpExchange, action))
            elif action.is_marriage() or action.is_regular_move():
                # Instead of completing the trick, store a PartialTrick in next_state.previous
                partial = PartialTrick(leader_move=action)
                next_state.previous = Previous(
                    state=old_state,
                    trick=partial,
                    leader_remained_leader=True  
                    # Typically the same leader continues for now, 
                    # because the trick is not complete yet.
                )
                return next_state
            else:
                raise ValueError("Unrecognized leader action type.")

        else:
            # FOLLOWER's move
            # We need to see if there's a pending partial trick from the previous step
            old_previous = old_state.previous
            if old_previous is None:
                raise ValueError("Follower cannot act but no 'previous' is set!")

            # Was the previous trick a partial?
            partial_trick = old_previous.trick
            if isinstance(partial_trick, PartialTrick):
                # We do have a stored leader move. Let's finalize the trick.
                leader_move = partial_trick.leader_move
                # Follower must play a regular move to complete it
                if not action.is_regular_move():
                    raise ValueError("Follower must respond with a regular move, but got something else.")
                follower_move = cast(RegularMove, action)

                # Now finalize the trick using leader + follower moves
                return apply_leader_follower_moves(
                    engine=engine,
                    old_state=old_state,
                    leader_move=leader_move,
                    follower_move=follower_move
                )
            else:
                # If the previous trick was not partial, there's no "pending" leader move.
                # Possibly the trick was ExchangeTrick or a fully completed RegularTrick.
                # If the follower tries to do something else here, handle that logic or raise an error.
                raise ValueError("Follower tried to play, but no partial trick to complete.")


    def _train_regret_network(self, batch_size: int):
        states, regrets = self.replay_buffer.sample(batch_size)
        if states is None:
            return
        self.optimizer.zero_grad()
        pred = self.regret_model(states)   # shape [batch_size, max_actions]
        loss = self.criterion(pred, regrets)
        loss.backward()
        self.optimizer.step()

    def get_regret_model(self) -> nn.Module:
        # After training, you can retrieve the learned regret_model
        return self.regret_model




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
        - points of this player (normalized int)
        - points of the opponent (normalized int)
        - pending points of this player (normalized int)
        - pending points of opponent (normalized int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hot encoding)
        - talon size (normalized int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown) (1-hot encoding)
        - leader move if it is given (1-hot encoding)

        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)

        Returns 145 feature vectors
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


#########  Dummy ##########

class _DummyBot(Bot):
    """A no-op Bot used only to fill in BotState. We do not call get_move."""
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        raise NotImplementedError
    

######### CFR Trickimplementer ###########


def apply_trump_exchange(engine: SchnapsenGamePlayEngine,
                         old_state: GameState,
                         trump_exchange: TrumpExchange) -> GameState:
    """
    Apply a trump-exchange Move in a copy of 'old_state', returning a new state.
    Mimics the logic from SchnapsenTrickImplementer for an exchange.
    """
    assert trump_exchange.jack.rank == Rank.JACK, \
        "Trump exchange must use a Jack"

    next_state = old_state.copy_for_next()  # Copies everything, sets previous=None

    # Remove the Jack from the leader's hand
    next_state.leader.hand.remove(trump_exchange.jack)
    # Exchange with the card at the bottom of the talon
    old_trump_card = next_state.talon.trump_exchange(trump_exchange.jack)
    # Add the old trump to the leader's hand
    next_state.leader.hand.add(old_trump_card)

    # Build an ExchangeTrick for the 'previous' pointer, if you want game history
    exchange_trick = ExchangeTrick(exchange=trump_exchange, trump_card=old_trump_card)
    next_state.previous = Previous(
        state=old_state,
        trick=exchange_trick,
        leader_remained_leader=True  # After an exchange, the same leader continues
    )

    return next_state


def apply_leader_follower_moves(engine: SchnapsenGamePlayEngine,
                                old_state: GameState,
                                leader_move: Union[RegularMove, Marriage],
                                follower_move: RegularMove) -> GameState:
    """
    Apply the (leader_move, follower_move) in a copy of 'old_state' and return the new state,
    similar to _apply_regular_trick from SchnapsenTrickImplementer.
    """
    # 1) Copy
    next_state = old_state.copy_for_next()

    # 2) If the leader_move is a Marriage, handle marriage scoring & removing queen+king
    if leader_move.is_marriage():
        marriage_move: Marriage = cast(Marriage, leader_move)
        # Score the marriage (adds pending points, etc.)
        marriage_score = engine.trick_scorer.marriage(marriage_move, next_state)
        next_state.leader.score += marriage_score
        # We remove the King card from the leader's hand, because it's played in the marriage
        # But the actual "card" for the "leader move" is the King. The queen is only to declare.
        # If you follow the code in SchnapsenTrickImplementer, it does:
        #    next_state.leader.hand.remove(king_card)
        # but the code also removes the queen card from the leader's hand if it wants, depending on your engine.
        # Typically, only the King is "played" in the trick, but you might want to remove the Queen too if your rules do so.
        # We'll remove both to be safe:
        next_state.leader.hand.remove(marriage_move.queen_card)
        next_state.leader.hand.remove(marriage_move.king_card)

        leader_card = marriage_move.king_card  # The actual "played" card
    else:
        # It's a regular move
        regular_leader_move: RegularMove = cast(RegularMove, leader_move)
        leader_card = regular_leader_move.card
        next_state.leader.hand.remove(leader_card)

    # 3) Remove the follower's card
    next_state.follower.hand.remove(follower_move.card)

    # 4) Score the trick (who wins, how many points, etc.)
    # This returns (winnerBotState, loserBotState, leaderRemainsLeader)
    # The trick is effectively "RegularTrick(leader_move, follower_move)"
    trick = RegularTrick(leader_move=leader_move, follower_move=follower_move)
    new_leader, new_follower, leader_stays = engine.trick_scorer.score(
        trick, next_state.leader, next_state.follower, next_state.trump_suit
    )

    # 5) The new leader draws first from the talon, then the new follower
    if not next_state.talon.is_empty():
        drawn = next_state.talon.draw_cards(2)
        new_leader.hand.add(drawn[0])
        new_follower.hand.add(drawn[1])

    # 6) Reassign them to next_state.leader/follower in correct order
    next_state.leader = new_leader
    next_state.follower = new_follower

    # 7) Build a 'RegularTrick' for next_state.previous if you want game history:
    next_state.previous = Previous(
        state=old_state,
        trick=trick,
        leader_remained_leader=leader_stays
    )

    return next_state

