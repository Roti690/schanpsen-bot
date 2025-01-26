from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase, SchnapsenDeckGenerator, SchnapsenGamePlayEngine
from typing import Optional
import random
import pathlib
import pandas as pd

class MCTSbot(Bot):
    """Monte Carlo Tree Search bot implementation."""

    def __init__(self, num_samples: int, depth: int, rand: random.Random, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.num_samples = num_samples
        self.depth = depth
        self.rand = rand
        self.engine = SchnapsenGamePlayEngine()

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        valid_moves = perspective.valid_moves()
        self.rand.shuffle(valid_moves)

        best_score = float('-inf')
        best_move = valid_moves[0]

        for move in valid_moves:
            score = 0
            for _ in range(self.num_samples):
                score += self.__evaluate(perspective.get_state_in_phase_two(), self.engine, leader_move, move)
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def __evaluate(self, gamestate, engine, leader_move: Optional[Move], my_move: Move) -> float:
        """Evaluate a move by simulating random play to a certain depth."""
        current_state = gamestate
        depth = self.depth
        score = 0.0

        while depth > 0 and not engine.is_game_over(current_state):
            # Apply my move
            if leader_move is None:
                # I am leader
                next_state = engine.play_one_trick(current_state, self, self)
            else:
                # I am follower
                next_state = engine.play_one_trick(current_state, self, self)

            # Random play for remaining depth
            while depth > 0 and not engine.is_game_over(next_state):
                valid_moves = engine.get_valid_moves(next_state)
                random_move = self.rand.choice(valid_moves)
                next_state = engine.play_one_trick(next_state, self, self)
                depth -= 1

            # Score the final state
            if engine.is_game_over(next_state):
                winner = engine.get_winner(next_state)
                if winner is current_state.leader:
                    score += 1.0
                else:
                    score -= 1.0

            depth -= 1

        return score 