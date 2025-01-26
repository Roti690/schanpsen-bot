import random
import pickle
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass

from schnapsen.game import Bot, Move, PlayerPerspective, Card, Suit, Rank, Marriage, TrumpExchange, RegularMove, GamePhase, Score
from schnapsen.deck import OrderedCardCollection, CardCollection

# ======================================
#  Utility function: sample an action 
# ======================================
def sample_action(action_prob_dict: Dict[Move, float]) -> Move:
    """Given a dictionary of {action: probability}, sample a Move."""
    actions = list(action_prob_dict.keys())
    probs = list(action_prob_dict.values())
    return random.choices(actions, weights=probs, k=1)[0]

# ======================================
#  MCCFRBot for inference / gameplay
# ======================================
class MCCFRBot(Bot):
    """
    This bot uses External Sampling MCCFR to learn a strategy through self-play.
    During actual gameplay (get_move), it uses the average strategy learned during training.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        # Maps: info_set_key -> {action -> cumulative regret or strategy sum}
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        
        # Cache for current iteration's strategy
        self.current_strategy = defaultdict(lambda: defaultdict(float))

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move] = None) -> Move:
        """
        Called by the game engine to pick an action. Uses the average strategy for stable performance.
        """
        valid_actions = perspective.valid_moves()
        if not valid_actions:
            raise ValueError("No valid actions available.")

        # Build info set key from the perspective
        info_set_key = self.make_info_set_key(perspective, leader_move)

        # Get average strategy (or uniform if no data)
        avg_strategy = self._get_average_strategy(info_set_key, valid_actions)

        # Sample an action using the average strategy
        return sample_action(avg_strategy)

    # -----------------------
    #  Key MCCFR methods
    # -----------------------

    def make_info_set_key(self, perspective: PlayerPerspective, leader_move: Optional[Move] = None) -> str:
        """
        Convert the player's perspective into a string key that represents the information set.
        Includes: hand, trump suit, talon size, scores, and leader's move if this player is following.
        """
        # Get my hand
        hand_str = "-".join(sorted(str(card) for card in perspective.get_hand()))
        
        # Get trump info
        trump_card = perspective.get_trump_card()
        trump_str = str(trump_card) if trump_card else "None"
        
        # Game state info
        talon_size = perspective.get_talon_size()
        my_score = perspective.get_my_score()
        opp_score = perspective.get_opponent_score()
        phase = perspective.get_phase()
        
        # If we're in phase 2, include opponent's hand since it's visible
        opp_hand_str = ""
        if phase == GamePhase.TWO:
            opp_hand = perspective.get_opponent_hand_in_phase_two()
            opp_hand_str = "-".join(sorted(str(card) for card in opp_hand))
        
        # Include leader's move if we're the follower
        leader_move_str = str(leader_move) if leader_move else "None"
        
        # Combine all information into a unique string
        key_parts = [
            f"Hand:{hand_str}",
            f"Trump:{trump_str}",
            f"TalonSize:{talon_size}",
            f"MyScore:{my_score}",
            f"OppScore:{opp_score}",
            f"Phase:{phase}",
            f"OppHand:{opp_hand_str}",
            f"LeaderMove:{leader_move_str}"
        ]
        
        return "|".join(key_parts)

    def get_current_strategy(self, info_set_key: str, valid_actions: List[Move]) -> Dict[Move, float]:
        """
        Convert regrets to probabilities via regret matching.
        Returns a dict: {action: probability}.
        """
        regrets = self.regret_sum[info_set_key]
        positive_regrets = []
        for a in valid_actions:
            positive_regrets.append(max(regrets[a], 0.0))
        total_pos = sum(positive_regrets)

        strategy = {}
        if total_pos > 0:
            for action, pos_reg in zip(valid_actions, positive_regrets):
                strategy[action] = pos_reg / total_pos
        else:
            # if all regrets <= 0, use uniform among valid actions
            for a in valid_actions:
                strategy[a] = 1.0 / len(valid_actions)

        # Store in current_strategy for usage in updating strategy_sum
        self.current_strategy[info_set_key] = strategy
        return strategy

    def update_regret(self, info_set_key: str, action: Move, regret: float):
        """Update the cumulative regret for an action in an information set."""
        self.regret_sum[info_set_key][action] += regret

    def update_strategy_sum(self, info_set_key: str, valid_actions: List[Move]):
        """
        After computing the current strategy, add it to the strategy_sum 
        for computing the average strategy later.
        """
        strategy = self.current_strategy[info_set_key]
        for a in valid_actions:
            self.strategy_sum[info_set_key][a] += strategy[a]

    def _get_average_strategy(self, info_set_key: str, valid_actions: List[Move]) -> Dict[Move, float]:
        """
        Returns the average strategy across all iterations for the given info_set_key.
        If no data is available, returns uniform random.
        """
        sum_strats = self.strategy_sum[info_set_key]
        total = sum(sum_strats[a] for a in valid_actions)
        if total > 1e-12:  # some small epsilon
            return {a: sum_strats[a] / total for a in valid_actions}
        else:
            # fallback to uniform
            return {a: 1.0 / len(valid_actions) for a in valid_actions}

    # -----------------------
    #  Model Persistence
    # -----------------------
    def save_model(self, filename: str):
        """Save the learned strategies and regrets to a file."""
        data = {
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum)
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, filename: str):
        """Load previously learned strategies and regrets from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Convert dictionaries back to defaultdict
        self.regret_sum = defaultdict(lambda: defaultdict(float), data['regret_sum'])
        self.strategy_sum = defaultdict(lambda: defaultdict(float), data['strategy_sum'])


# ======================================
#  Training Functions
# ======================================
def external_sampling_mccfr_traverse(
    perspective: PlayerPerspective,
    trainer_bot: MCCFRBot,
    updating_player_id: int,
    leader_move: Optional[Move] = None
) -> float:
    """
    Recursively traverse the game tree using external sampling for the
    specified 'updating_player_id'. The other player (the opponent) 
    is "sampled" - i.e., we pick only one of their actions 
    according to their current strategy.

    Returns the expected utility for updating_player_id in the current state.
    """
    # Check if game is over
    if not perspective.valid_moves():
        # Return utility relative to updating_player_id
        # Note: This is simplified - you might need to adjust based on actual scoring
        my_score = perspective.get_my_score()
        opp_score = perspective.get_opponent_score()
        if perspective.am_i_leader():
            return float(my_score.direct_points - opp_score.direct_points)
        else:
            return float(opp_score.direct_points - my_score.direct_points)

    current_player_id = 0 if perspective.am_i_leader() else 1
    valid_actions = perspective.valid_moves()

    # Build info set key
    info_set_key = trainer_bot.make_info_set_key(perspective, leader_move)

    # If current player is the one we're updating
    if current_player_id == updating_player_id:
        # Consider all actions to update regrets
        strategy = trainer_bot.get_current_strategy(info_set_key, valid_actions)
        
        action_values = {}
        node_value = 0.0

        for action in valid_actions:
            # Get the resulting state after this action
            # Note: This is where you'd need to implement state progression
            next_perspective = perspective.get_engine().play_one_trick(
                perspective.get_state_in_phase_two(),
                trainer_bot,
                trainer_bot
            )
            
            # Recursively get the value
            action_values[action] = external_sampling_mccfr_traverse(
                next_perspective,
                trainer_bot,
                updating_player_id,
                action if perspective.am_i_leader() else None
            )
            node_value += strategy[action] * action_values[action]

        # Update regrets
        for action in valid_actions:
            regret = action_values[action] - node_value
            trainer_bot.update_regret(info_set_key, action, regret)

        # Update strategy sum for average strategy computation
        trainer_bot.update_strategy_sum(info_set_key, valid_actions)

        return node_value

    else:
        # Opponent's turn => sample one action according to current strategy
        strategy = trainer_bot.get_current_strategy(info_set_key, valid_actions)
        sampled_action = sample_action(strategy)

        # Get next state after sampled action
        next_perspective = perspective.get_engine().play_one_trick(
            perspective.get_state_in_phase_two(),
            trainer_bot,
            trainer_bot
        )

        # No regret update for opponent
        return external_sampling_mccfr_traverse(
            next_perspective,
            trainer_bot,
            updating_player_id,
            sampled_action if perspective.am_i_leader() else None
        )


def train_mccfr(
    num_iterations: int,
    bot1: MCCFRBot,
    bot2: MCCFRBot,
    engine: Any  # SchnapsenGamePlayEngine
) -> None:
    """
    Run a self-play training loop for num_iterations iterations.
    Alternates which player is being updated on each iteration.
    """
    for it in range(num_iterations):
        # Create a fresh game state
        state = engine.get_random_phase_two_state(random.Random())
        
        # Decide which player to update this iteration
        updating_player_id = it % 2
        
        if updating_player_id == 0:
            # Update bot1's strategy
            perspective = state.get_perspective(0)
            external_sampling_mccfr_traverse(perspective, bot1, updating_player_id)
        else:
            # Update bot2's strategy
            perspective = state.get_perspective(1)
            external_sampling_mccfr_traverse(perspective, bot2, updating_player_id)

        if (it + 1) % 1000 == 0:
            print(f"Iteration {it+1} / {num_iterations} complete.")


# =======================
# Example Usage (CLI)
# =======================
if __name__ == "__main__":
    import sys
    from schnapsen.game import SchnapsenGamePlayEngine

    # Parse arguments
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: mccfr_bot.py <train|play> [options]")
        sys.exit(1)

    mode = args[0]

    if mode == "train":
        num_iters = int(args[1]) if len(args) > 1 else 10000
        bot1 = MCCFRBot("MCCFR_Bot1")
        bot2 = MCCFRBot("MCCFR_Bot2")
        engine = SchnapsenGamePlayEngine()

        print(f"Starting training for {num_iters} iterations...")
        train_mccfr(num_iters, bot1, bot2, engine)

        # Save models
        bot1.save_model("mccfr_bot1.pkl")
        bot2.save_model("mccfr_bot2.pkl")
        print("Training complete. Models saved.")

    elif mode == "play":
        if len(args) < 3:
            print("Usage: mccfr_bot.py play <model_bot1.pkl> <model_bot2.pkl>")
            sys.exit(1)
        model1_path = args[1]
        model2_path = args[2]

        bot1 = MCCFRBot("MCCFR_Bot1")
        bot2 = MCCFRBot("MCCFR_Bot2")
        bot1.load_model(model1_path)
        bot2.load_model(model2_path)

        engine = SchnapsenGamePlayEngine()
        winner, points, score = engine.play_game(bot1, bot2, random.Random())
        print(f"Winner: {winner} with {points} points and score {score}")

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1) 