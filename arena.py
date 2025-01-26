import sys
import pathlib
import random
import numpy as np
from typing import List, Tuple, Dict
from schnapsen.game import SchnapsenGamePlayEngine, Bot
from schnapsen.bots import RandBot, RdeepBot

# Add necessary paths
sys.path.append("./models")
sys.path.append("./data_gen")

from ML_bot import MLPlayingBot
from DeepCFR import RegretNetwork, StrategyNetwork, DeepCFRBot
from mccfr_bot import MCCFRBot

class Arena:
    """Arena for running bot tournaments"""
    
    def __init__(self, engine: SchnapsenGamePlayEngine = None):
        self.engine = engine or SchnapsenGamePlayEngine()
        self.rng = np.random.RandomState()
    
    def load_ml_bot(self, model_path: str, name: str = "MLBot") -> MLPlayingBot:
        """Load an ML bot from a saved model"""
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found at: {model_path}")
        return MLPlayingBot(model_path, name=name)
    
    def load_deepcfr_bot(self, input_size: int, name: str = "DeepCFRBot") -> DeepCFRBot:
        """Create a DeepCFR bot with given input size"""
        regret_net = RegretNetwork(input_size=input_size, action_size=1)
        strategy_net = StrategyNetwork(input_size=input_size, action_size=1)
        return DeepCFRBot(regret_net, strategy_net, name=name)
    
    def load_mccfr_bot(self, model_path: str, name: str = "MCCFRBot") -> MCCFRBot:
        """Load an MCCFR bot from a saved model"""
        bot = MCCFRBot(name=name)
        bot.load_model(model_path)
        return bot
    
    def create_rdeep_bot(self, num_samples: int = 8, depth: int = 4, name: str = "RdeepBot") -> RdeepBot:
        """Create an RdeepBot with given parameters"""
        return RdeepBot(num_samples=num_samples, depth=depth, rand=self.rng, name=name)
    
    def create_rand_bot(self, name: str = "RandBot") -> RandBot:
        """Create a random bot"""
        return RandBot(self.rng, name=name)
    
    def play_match(self, bot1: Bot, bot2: Bot, num_games: int = 10) -> dict:
        """Play a match between two bots and return the results."""
        wins = {str(bot1): 0, str(bot2): 0}
        for _ in range(num_games):
            engine = SchnapsenGamePlayEngine()
            winner, _, _ = engine.play_game(bot1, bot2, self.rng)
            wins[str(winner)] += 1
        return wins
    
    def run_tournament(self, bots: List[Bot], num_games: int = 10) -> dict:
        """Run a round-robin tournament between all bots."""
        # Initialize results dictionary with wins and losses
        results = {str(bot): {"wins": 0, "losses": 0} for bot in bots}
        
        # Run matches
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                match_results = self.play_match(bots[i], bots[j], num_games)
                bot1_wins = match_results[str(bots[i])]
                bot2_wins = match_results[str(bots[j])]
                
                # Update wins and losses
                results[str(bots[i])]["wins"] += bot1_wins
                results[str(bots[i])]["losses"] += bot2_wins
                results[str(bots[j])]["wins"] += bot2_wins
                results[str(bots[j])]["losses"] += bot1_wins
        
        return results
    
    def print_tournament_results(self, results: dict):
        """Print tournament results in a readable format."""
        print("\nTournament Results:")
        print("-" * 40)
        print("Bot".ljust(20) + "Wins".ljust(10) + "Losses".ljust(10))
        print("-" * 40)
        
        # Sort bots by wins
        sorted_bots = sorted(results.keys(), key=lambda x: results[x]["wins"], reverse=True)
        
        for bot in sorted_bots:
            stats = results[bot]
            print(f"{str(bot)[:20].ljust(20)}{str(stats['wins']).ljust(10)}{str(stats['losses']).ljust(10)}")

def main():
    """Run a sample tournament"""
    arena = Arena()
    
    # Create bots
    rdeep = arena.create_rdeep_bot(name="RDeep-8-4")
    rand = arena.create_rand_bot(name="Random")
    
    try:
        # Try to load trained bots if available
        ml_bot = arena.load_ml_bot("models/model_latest.pt", name="MLBot")
        mccfr_bot = arena.load_mccfr_bot("models/mccfr_bot1.pkl", name="MCCFR")
        
        # Get input size from ML bot for DeepCFR
        input_size = ml_bot.model[0].in_features
        deepcfr_bot = arena.load_deepcfr_bot(input_size, name="DeepCFR")
        
        bots = [rdeep, rand, ml_bot, mccfr_bot, deepcfr_bot]
    except Exception as e:
        print(f"Could not load all bots: {e}")
        print("Running tournament with basic bots only")
        bots = [rdeep, rand]
    
    # Run tournament
    results = arena.run_tournament(bots, num_games=50)
    arena.print_tournament_results(results)

if __name__ == "__main__":
    main() 