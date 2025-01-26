import sys
import random
import pathlib
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.bots import RdeepBot

# Add necessary paths
sys.path.append("./data_gen")
sys.path.append("./models")

from data_generation import MLDataBot

def main():
    # Initialize game engine and random number generator
    engine = SchnapsenGamePlayEngine()
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Create base bot for data generation
    base_bot = RdeepBot(num_samples=8, depth=4, rand=rng)
    
    # Setup replay memory directory and file
    replay_memory_dir = pathlib.Path("ML_replay_memories")
    replay_memory_dir.mkdir(parents=True, exist_ok=True)
    replay_memory_file = replay_memory_dir / "replay_memory.csv"
    
    # Create data collection bot
    data_bot = MLDataBot(base_bot, replay_memory_file)
    
    # Generate training data
    print("Generating training data...")
    num_games = 100
    for i in range(num_games):
        if i % 10 == 0:
            print(f"Playing game {i}/{num_games}")
        # Use different random seed for each game
        game_rng = random.Random(i)
        engine.play_game(data_bot, base_bot, game_rng)
    
    print("Data generation complete.")

if __name__ == "__main__":
    main()
