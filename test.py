import sys
import pathlib
import random
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.bots import RandBot, RdeepBot
import pandas as pd

# Add necessary paths
sys.path.append("./models")
sys.path.append("./data_gen")

from data_generation import create_replay_memory_dataset, get_state_feature_vector, get_move_feature_vector
from ML_bot import train_model, train_DL_model, MLPlayingBot
from DeepCFR import RegretNetwork, StrategyNetwork, DeepCFRBot
from mccfr_bot import MCCFRBot, train_mccfr
from arena import Arena

def get_input_size():
    """Calculate the input size by getting a sample state and move vector"""
    # Read the first row from the test data to get feature dimensions
    data_file = "ML_replay_memories/test_replay_memory.csv"
    data = pd.read_csv(data_file)
    
    # Convert string representations of lists to actual lists
    state_vector = eval(data['state_vector'].iloc[0])
    move_vector = eval(data['leader_move_vec'].iloc[0])
    
    return len(state_vector) + len(move_vector)

def test_data_generation():
    print("Testing data generation...")
    rng = random.Random(42)
    bot1 = RdeepBot(num_samples=4, depth=4, rand=rng)
    bot2 = RdeepBot(num_samples=4, depth=4, rand=rng)
    
    create_replay_memory_dataset(
        bot1=bot1,
        bot2=bot2,
        num_of_games=100,  # Smaller number for testing
        replay_memory_dir="ML_replay_memories",
        replay_memory_filename="test_replay_memory.csv",
        parallel=False,
        overwrite=True
    )
    
    # Verify file exists
    assert pathlib.Path("ML_replay_memories/test_replay_memory.csv").exists()
    print("Data generation test passed!")

def test_ml_training():
    print("Testing ML model training...")
    data_file = "ML_replay_memories/test_replay_memory.csv"
    model_path = train_DL_model(
        data_file=data_file,
        output_model_path="models",
        input_dim=get_input_size(),
        hidden_dim=64,
        batch_size=32,
        epochs=5,  # Smaller number for testing
        lr=0.001
    )
    
    # Verify model file exists
    assert pathlib.Path(model_path).exists()
    print("ML training test passed!")

def test_mccfr_training():
    print("Testing MCCFR training...")
    engine = SchnapsenGamePlayEngine()
    
    # Create two MCCFR bots for training
    bot1 = MCCFRBot("MCCFRBot1")
    bot2 = MCCFRBot("MCCFRBot2")
    
    # Train the bots
    print("Training MCCFR bots...")
    train_mccfr(
        num_iterations=100,  # Small number for testing
        bot1=bot1,
        bot2=bot2,
        engine=engine
    )
    
    # Save trained models
    bot1.save_model("models/mccfr_bot1.pkl")
    bot2.save_model("models/mccfr_bot2.pkl")
    
    # Test if models were saved
    assert pathlib.Path("models/mccfr_bot1.pkl").exists()
    assert pathlib.Path("models/mccfr_bot2.pkl").exists()
    
    # Test gameplay
    rng = random.Random(42)
    rand_bot = RandBot(rng)
    
    # Play a test game
    winner_id, points, score = engine.play_game(bot1, rand_bot, rng)
    print(f"Game completed. Winner: {winner_id}")
    
    print("MCCFR training test passed!")

def test_bot_gameplay():
    print("Testing bot gameplay...")
    engine = SchnapsenGamePlayEngine()
    rng = random.Random(42)
    
    # Test ML bot
    model_path = list(pathlib.Path("models").glob("model_*.pt"))[0]  # Get first model file
    ml_bot = MLPlayingBot(model_path)
    rand_bot = RandBot(rng)
    
    # Play a test game
    winner_id, points, score = engine.play_game(ml_bot, rand_bot, rng)
    print(f"Game completed. Winner: {winner_id}")
    
    # Test DeepCFR bot with correct input size
    input_size = get_input_size()
    regret_net = RegretNetwork(input_size=input_size, action_size=1)
    strategy_net = StrategyNetwork(input_size=input_size, action_size=1)
    cfr_bot = DeepCFRBot(regret_net, strategy_net)
    
    # Play a test game
    winner_id, points, score = engine.play_game(cfr_bot, rand_bot, rng)
    print(f"Game completed. Winner: {winner_id}")
    
    # Test MCCFR bot
    mccfr_bot = MCCFRBot("MCCFRBot")
    mccfr_bot.load_model("models/mccfr_bot1.pkl")
    
    # Play a test game
    winner_id, points, score = engine.play_game(mccfr_bot, rand_bot, rng)
    print(f"Game completed. Winner: {winner_id}")
    
    print("Bot gameplay test passed!")

def test_arena():
    print("Testing arena...")
    arena = Arena()
    
    # Create basic bots for testing
    rdeep = arena.create_rdeep_bot(name="RDeep-4-2", num_samples=4, depth=2)
    rand = arena.create_rand_bot(name="Random")
    
    # Test match between two bots
    print("Testing single match...")
    match_results = arena.play_match(rdeep, rand, num_games=10)
    assert len(match_results) == 2
    assert sum(match_results.values()) == 10
    print("Single match test passed!")
    
    # Test tournament with basic bots
    print("Testing tournament...")
    tournament_results = arena.run_tournament([rdeep, rand], num_games=10)
    assert len(tournament_results) == 2
    assert all(len(r) == 2 for r in tournament_results.values())
    print("Tournament test passed!")
    
    # Test loading trained bots if available
    try:
        print("Testing with trained bots...")
        model_path = list(pathlib.Path("models").glob("model_*.pt"))[0]
        ml_bot = arena.load_ml_bot(model_path, name="MLBot")
        mccfr_bot = arena.load_mccfr_bot("models/mccfr_bot1.pkl", name="MCCFR")
        
        # Get input size from ML bot for DeepCFR
        input_size = ml_bot.model[0].in_features
        deepcfr_bot = arena.load_deepcfr_bot(input_size, name="DeepCFR")
        
        # Run mini tournament with all bots
        all_bots = [rdeep, rand, ml_bot, mccfr_bot, deepcfr_bot]
        results = arena.run_tournament(all_bots, num_games=10)
        arena.print_tournament_results(results)
        print("Full tournament test passed!")
    except Exception as e:
        print(f"Skipping trained bot tests: {e}")
    
    print("Arena test passed!")

def main():
    # Run all tests
    try:
        test_data_generation()
        test_ml_training()
        test_mccfr_training()
        test_bot_gameplay()
        test_arena()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
