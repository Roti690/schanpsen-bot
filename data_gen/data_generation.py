from schnapsen.game import (Bot, PlayerPerspective, Move, GamePhase, SchnapsenDeckGenerator, SchnapsenGamePlayEngine, Trick,
                            Previous, GameState, Marriage, TrumpExchange, RegularMove, ExchangeTrick, RegularTrick, SchnapsenTrickScorer,
                            GamePlayEngine)
from schnapsen.deck import Suit, Rank
from typing import Optional, cast, Union, Dict
import pathlib
import random
from multiprocessing import Pool
import pandas as pd
import math
from schnapsen.bots import RandBot


class MLDataBot(Bot):
    """
    This class is defined to allow the creation of a training schnapsen bot dataset, that allows us to train a Machine Learning (ML) Bot
    Practically, it helps us record how the game plays out according to a provided Bot behaviour; build what is called a "replay memory"
    In more detail, we create one training sample for each decision the bot makes within a game, where a decision is an action selection for a specific game state.
    Then we relate each decision with the outcome of the game, i.e. whether this bot won or not.
    This way we can then train a bot according to the assumption that:
        "decisions in earlier games that ended up in victories should be preferred over decisions that lead to lost games"
    This class only records the decisions and game outcomes of the provided bot, according to its own perspective - incomplete game state knowledge.
    """

    def __init__(self, bot: Bot, replay_memory_location: pathlib.Path) -> None:
        """
        :param bot: the provided bot that will actually play the game and make decisions
        :param replay_memory_location: the filename under which the replay memory records will be
        """

        self.bot: Bot = bot
        self.replay_memory_file_path: pathlib.Path = replay_memory_location
        self._records_for_this_game = []

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        """
            This function simply calls the get_move of the provided bot
        """
        return self.bot.get_move(perspective=perspective, leader_move=leader_move)

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        """
        When the game ends, this function retrieves the game history and more specifically all the replay memories that can
        be derived from it, and stores them in the form of state-actions vector representations and the corresponding outcome of the game

        :param won: Did this bot win the game?
        :param state: The final state of the game.
        """
        # we retrieve the game history while actually discarding the last useless history record (which is after the game has ended),
        # we know none of the Tricks can be None because that is only for the last record
        full_history: list[tuple[PlayerPerspective, Trick]] = cast(list[tuple[PlayerPerspective, Trick]], perspective.get_game_history()[:-1])
        # we also save the training label "won or lost"
        won_label = won
        num_tricks = len(full_history) - 1


        for i in range(num_tricks):
            # The perspective and trick before trick i is played
            current_persp, current_trick = full_history[i]
            # The next perspective after trick i is scored (i.e., before trick i+1)
            next_persp, _ = full_history[i+1]

            # Score difference for "me" from before to after this trick
            old_score = current_persp.get_my_score().direct_points
            new_score = next_persp.get_my_score().direct_points
            points_this_trick = new_score - old_score

            # Identify the actual moves (leader/follower).
            if current_trick.is_trump_exchange():
                leader_move = current_trick.exchange
                follower_move = None
            else:
                leader_move = current_trick.leader_move
                follower_move = current_trick.follower_move

            # If I was leader, ignore the follower's move in the representation
            if current_persp.am_i_leader():
                follower_move = None

            # Vector encodings, as before
            state_vector = get_state_feature_vector(current_persp)
            leader_move_vec = get_move_feature_vector(leader_move)
            follower_move_vec = get_move_feature_vector(follower_move)

            # Possibly also gather valid moves
            valid_moves = current_persp.valid_moves()
            valid_moves_vecs = [get_move_feature_vector(m) for m in valid_moves]

            # Build a dict or row for this trick
            row = {
                "state_vector": state_vector,
                "valid_moves": valid_moves_vecs,
                "leader_move_vec": leader_move_vec,
                "follower_move_vec": follower_move_vec,
                "points_this_trick": points_this_trick,
                "did_win_game": won_label
            }

            # Now store or append this row to a list, or write to CSV, etc.
            # For demonstration, let's just append to a local list for the entire game
            # and convert to CSV at the end:
            self._records_for_this_game.append(row)

            # Write out at the end of the game
            self._write_to_csv()
            self._records_for_this_game.clear()

    def _write_to_csv(self):
        """
        Converts self._records_for_this_game into a DataFrame and appends it to the CSV.
        If the CSV doesn't exist, we write headers, else append without headers.
        """
        if not self._records_for_this_game:
            return

        df = pd.DataFrame(self._records_for_this_game)

        # Check if file already exists
        file_exists = self.replay_memory_file_path.exists()
        
        # Append mode="a", no header if file_exists
        df.to_csv(
            self.replay_memory_file_path,
            mode="a",
            header=not file_exists,  # write header only if file doesn't exist
            index=False
        )


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


def get_state_feature_vector(perspective: PlayerPerspective) -> list[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hoy encoding)
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

    return state_feature_list


def create_replay_memory_dataset(
    bot1: Bot,
    bot2: Bot,
    num_of_games: int = 10000,
    replay_memory_dir: str = "ML_replay_memories",
    replay_memory_filename: str = "replay_memory.csv",
    parallel: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Create an enhanced replay memory dataset for training.

    Args:
        bot1, bot2: The bots to simulate games.
        num_of_games: Total games to simulate.
        replay_memory_dir: Directory to store the dataset.
        replay_memory_filename: Name of the dataset file.
        parallel: Whether to use parallel processing.
    """


    # Prepare the replay memory location
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and replay_memory_location.exists():
        print(f"Existing dataset found at {replay_memory_location}. Overwriting...")
        replay_memory_location.unlink()  # Delete the existing file

    
    if parallel:
        # Use multiprocessing for parallel execution
        with Pool() as pool:
            pool.starmap(
                simulate_game,
                [(game_id, bot1, bot2, replay_memory_location) for game_id in range(1, num_of_games + 1)],
            )
    else:
        # Run sequentially
        for game_id in range(1, num_of_games + 1):
            simulate_game(game_id, bot1, bot2, replay_memory_location)
    
def simulate_game(game_id, bot1, bot2, replay_memory_location):
    """Simulate a single game and save its data."""
    engine = SchnapsenGamePlayEngine()
    random_seed = random.Random(game_id)
    engine.play_game(
        MLDataBot(bot1, replay_memory_location=replay_memory_location),
        MLDataBot(bot2, replay_memory_location=replay_memory_location),
        random_seed,
    )
    if game_id % 500 == 0:
        print(f"Game {game_id} completed.")


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

def explore_all_paths(
    engine: SchnapsenGamePlayEngine,
    state: GameState,
    path_so_far: list,
    all_completed_paths: list
) -> None:
    """
    Enumerate *all* possible ways the game can continue from 'state'.
    :param engine: a SchnapsenGamePlayEngine
    :param state: current GameState
    :param path_so_far: a list of (player_role, move) or something to track how we got here
    :param all_completed_paths: a list that will store completed paths. Each element could be:
         { "path": [...], "final_state": GameState, "winner": BotState, "game_points": int }
    """

    # 1) Check if there's already a winner
    maybe_winner = engine.trick_scorer.declare_winner(state)
    if maybe_winner is not None:
        winner_bot, game_points = maybe_winner
        # We record a completed path
        all_completed_paths.append({
            "path": list(path_so_far),  # copy
            "final_state": state,
            "winner": winner_bot,
            "game_points": game_points
        })
        return

    # 2) It's not over, so let's see which moves the leader can do:
    leader_moves = engine.move_validator.get_legal_leader_moves(engine, state)

    for lmove in leader_moves:
        if lmove.is_trump_exchange():
            # 2a) If it's a trump exchange, we skip the follower move
            next_state = apply_trump_exchange(engine, state, lmove.as_trump_exchange())

            # record that the leader did lmove
            path_so_far.append(("leader", lmove))
            explore_all_paths(engine, next_state, path_so_far, all_completed_paths)
            path_so_far.pop()

        else:
            # 2b) If it's a regular or marriage move, we must also consider *all*
            #     valid follower moves in the next step
            lmove_regular = cast(Union[RegularMove, Marriage], lmove)
            follower_moves = engine.move_validator.get_legal_follower_moves(
                engine, state, lmove_regular
            )

            for fmove in follower_moves:
                # apply
                next_state = apply_leader_follower_moves(
                    engine, state, lmove_regular, cast(RegularMove, fmove)
                )

                path_so_far.append(("leader", lmove))
                path_so_far.append(("follower", fmove))

                # Recurse
                explore_all_paths(engine, next_state, path_so_far, all_completed_paths)

                # backtrack
                path_so_far.pop()
                path_so_far.pop()


class _DummyBot(Bot):
    """A no-op Bot used only to fill in BotState. We do not call get_move."""
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        raise NotImplementedError


class MCTSbot(Bot):
    """
    Rdeep bot is a bot which performs many random rollouts of the game to decide which move to play.
    """
    def __init__(self, replay_memory_file_path: pathlib.Path, num_samples: int, depth: int, rand: random.Random, name: Optional[str] = None) -> None:
        """
        Create a new rdeep bot.

        :param num_samples: how many samples to take per move
        :param depth: how deep to sample
        :param rand: the source of randomness for this Bot
        :param name: the name of this Bot
        """
        self.replay_memory_file_path = replay_memory_file_path
        self.records_for_this_game = []

        super().__init__(name)
        assert num_samples >= 1, f"we cannot work with less than one sample, got {num_samples}"
        assert depth >= 1, f"it does not make sense to use a dept <1. got {depth}"
        self.__num_samples = num_samples
        self.__depth = depth
        self.__rand = rand

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # get the list of valid moves, and shuffle it such
        # that we get a random move of the highest scoring
        # ones if there are multiple highest scoring moves.
        moves = perspective.valid_moves()
        self.__rand.shuffle(moves)

        if len(moves) == 1:
            return moves[0]

        best_score = float('-inf')
        best_move = None
        for move in moves:
            sum_of_scores = 0.0
            for _ in range(self.__num_samples):
                gamestate = perspective.make_assumption(leader_move=leader_move, rand=self.__rand)
                score = self.__evaluate(gamestate, perspective.get_engine(), leader_move, move)
                sum_of_scores += score
            average_score = sum_of_scores / self.__num_samples
            if average_score > best_score:
                best_score = average_score
                best_move = move
        assert best_move is not None, "We went over all the moves, selecting the one we expect to lead to the highest average score. Simce there must have been at least one move at the start, this can never be None"
        return best_move

    def __evaluate(self, gamestate: GameState, engine: GamePlayEngine, leader_move: Optional[Move], my_move: Move) -> float:
        """
        Evaluates the value of the given state for the given player
        :param state: The state to evaluate
        :param player: The player for whom to evaluate this state (1 or 2)
        :return: A float representing the value of this state for the given player. The higher the value, the better the
                state is for the player.
        """
        me: Bot
        leader_bot: Bot
        follower_bot: Bot

        if leader_move:
            # we know what the other bot played
            leader_bot = FirstFixedMoveThenBaseBot(base_bot= RandBot(rand= self.__rand), first_move= leader_move, replay_memory_file_path= self.replay_memory_file_path)
            # I am the follower
            me = follower_bot = FirstFixedMoveThenBaseBot(RandBot(rand=self.__rand), my_move, self.replay_memory_file_path)
        else:
            # I am the leader bot
            me = leader_bot = FirstFixedMoveThenBaseBot(RandBot(rand=self.__rand), my_move, self.replay_memory_file_path)
            # We assume the other bot just random
            follower_bot = RandBot(self.__rand)

        new_game_state, _ = engine.play_at_most_n_tricks(game_state=gamestate, new_leader=leader_bot, new_follower=follower_bot, n=self.__depth)

        if new_game_state.leader.implementation is me:
            my_score = new_game_state.leader.score.direct_points
            opponent_score = new_game_state.follower.score.direct_points
        else:
            my_score = new_game_state.follower.score.direct_points
            opponent_score = new_game_state.leader.score.direct_points

        heuristic = my_score / (my_score + opponent_score)
        return heuristic
    
    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        """
        When the game ends, this function retrieves the game history and more specifically all the replay memories that can
        be derived from it, and stores them in the form of state-actions vector representations and the corresponding outcome of the game

        :param won: Did this bot win the game?
        :param state: The final state of the game.
        """
        # we retrieve the game history while actually discarding the last useless history record (which is after the game has ended),
        # we know none of the Tricks can be None because that is only for the last record
        full_history: list[tuple[PlayerPerspective, Trick]] = cast(list[tuple[PlayerPerspective, Trick]], perspective.get_game_history()[:-1])
        # we also save the training label "won or lost"
        if won:
            won_label = 1
        else:
            won_label = 0

        num_tricks = len(full_history) - 1


        for i in range(num_tricks):
            # The perspective and trick before trick i is played
            current_persp, current_trick = full_history[i]
            # The next perspective after trick i is scored (i.e., before trick i+1)
            next_persp, _ = full_history[i+1]

            # Score difference for "me" from before to after this trick
            old_score = current_persp.get_my_score().direct_points
            new_score = next_persp.get_my_score().direct_points
            points_this_trick = new_score - old_score

            # Identify the actual moves (leader/follower).
            if current_trick.is_trump_exchange():
                leader_move = current_trick.exchange
                follower_move = None
            else:
                leader_move = current_trick.leader_move
                follower_move = current_trick.follower_move

            # If I was leader, ignore the follower's move in the representation
            if current_persp.am_i_leader():
                follower_move = None

            # Vector encodings, as before
            state_vector = get_state_feature_vector(current_persp)
            leader_move_vec = get_move_feature_vector(leader_move)
            follower_move_vec = get_move_feature_vector(follower_move)

            # Possibly also gather valid moves
            valid_moves = current_persp.valid_moves()
            valid_moves_vecs = [get_move_feature_vector(m) for m in valid_moves]

            # Build a dict or row for this trick
            row = {
                "state_vector": state_vector,
                "valid_moves": valid_moves_vecs,
                "leader_move_vec": leader_move_vec,
                "follower_move_vec": follower_move_vec,
                "points_this_trick": points_this_trick,
                "did_win_game": won_label
            }

            # Now store or append this row to a list, or write to CSV, etc.
            # For demonstration, let's just append to a local list for the entire game
            # and convert to CSV at the end:
            self.records_for_this_game.append(row)

            # Write out at the end of the game
            self._write_to_csv()
            self.records_for_this_game.clear()

    def _write_to_csv(self):
        """
        Converts self._records_for_this_game into a DataFrame and appends it to the CSV.
        If the CSV doesn't exist, we write headers, else append without headers.
        """
        if not self.records_for_this_game:
            return

        df = pd.DataFrame(self.records_for_this_game)

        # Check if file already exists
        file_exists = self.replay_memory_file_path.exists()
        
        # Append mode="a", no header if file_exists
        df.to_csv(
            self.replay_memory_file_path,
            mode="a",
            header=not file_exists,  # write header only if file doesn't exist
            index=False
        )


class FirstFixedMoveThenBaseBot(Bot):
    def __init__(self, base_bot: Bot, first_move: Move, replay_memory_file_path: pathlib.Path) -> None:
        self.first_move = first_move
        self.first_move_played = False
        self.base_bot = base_bot
        self.records_for_this_game = []
        self.replay_memory_file_path = replay_memory_file_path

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        if not self.first_move_played:
            self.first_move_played = True
            return self.first_move
        return self.base_bot.get_move(perspective=perspective, leader_move=leader_move)
    
    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        """
        When the game ends, this function retrieves the game history and more specifically all the replay memories that can
        be derived from it, and stores them in the form of state-actions vector representations and the corresponding outcome of the game

        :param won: Did this bot win the game?
        :param state: The final state of the game.
        """
        # we retrieve the game history while actually discarding the last useless history record (which is after the game has ended),
        # we know none of the Tricks can be None because that is only for the last record
        full_history: list[tuple[PlayerPerspective, Trick]] = cast(list[tuple[PlayerPerspective, Trick]], perspective.get_game_history()[:-1])
        # we also save the training label "won or lost"
        if won:
            won_label = 1
        else:
            won_label = 0
            
        num_tricks = len(full_history) - 1


        for i in range(num_tricks):
            # The perspective and trick before trick i is played
            current_persp, current_trick = full_history[i]
            # The next perspective after trick i is scored (i.e., before trick i+1)
            next_persp, _ = full_history[i+1]

            # Score difference for "me" from before to after this trick
            old_score = current_persp.get_my_score().direct_points
            new_score = next_persp.get_my_score().direct_points
            points_this_trick = new_score - old_score

            # Identify the actual moves (leader/follower).
            if current_trick.is_trump_exchange():
                leader_move = current_trick.exchange
                follower_move = None
            else:
                leader_move = current_trick.leader_move
                follower_move = current_trick.follower_move

            # If I was leader, ignore the follower's move in the representation
            if current_persp.am_i_leader():
                follower_move = None

            # Vector encodings, as before
            state_vector = get_state_feature_vector(current_persp)
            leader_move_vec = get_move_feature_vector(leader_move)
            follower_move_vec = get_move_feature_vector(follower_move)

            # Possibly also gather valid moves
            valid_moves = current_persp.valid_moves()
            valid_moves_vecs = [get_move_feature_vector(m) for m in valid_moves]

            # Build a dict or row for this trick
            row = {
                "state_vector": state_vector,
                "valid_moves": valid_moves_vecs,
                "leader_move_vec": leader_move_vec,
                "follower_move_vec": follower_move_vec,
                "points_this_trick": points_this_trick,
                "did_win_game": won_label
            }

            # Now store or append this row to a list, or write to CSV, etc.
            # For demonstration, let's just append to a local list for the entire game
            # and convert to CSV at the end:
            self.records_for_this_game.append(row)

            # Write out at the end of the game
            self._write_to_csv()
            self.records_for_this_game.clear()

    def _write_to_csv(self):
        """
        Converts self._records_for_this_game into a DataFrame and appends it to the CSV.
        If the CSV doesn't exist, we write headers, else append without headers.
        """
        if not self.records_for_this_game:
            return

        df = pd.DataFrame(self.records_for_this_game)

        # Check if file already exists
        file_exists = self.replay_memory_file_path.exists()
        
        # Append mode="a", no header if file_exists
        df.to_csv(
            self.replay_memory_file_path,
            mode="a",
            header=not file_exists,  # write header only if file doesn't exist
            index=False
        )