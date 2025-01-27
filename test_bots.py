import sys
import pathlib
import random
import time
from datetime import datetime
import pandas as pd
from typing import List, Tuple, Dict, Any
import json
import multiprocessing as mp
from itertools import combinations
import numpy as np

from schnapsen.game import SchnapsenGamePlayEngine, Bot
from schnapsen.bots import RdeepBot, RandBot
from data_gen.data_generation import MCTSbot
from models.mccfr_bot import MCCFRBot
from models.DeepCFR import DeepCFRBot, RegretNetwork, StrategyNetwork

def play_single_game(args) -> Dict[str, float]:
    """Play a single game between two bots."""
    bot1, bot2, seed = args
    engine = SchnapsenGamePlayEngine()
    rng = random.Random(seed)
    winner, points, score = engine.play_game(bot1, bot2, rng)
    
    result = {
        "winner": 1 if winner == bot1 else 0,
        "points_bot1": points if winner == bot1 else 0,
        "points_bot2": points if winner == bot2 else 0,
        "total_points": points
    }
    return result

def run_match_parallel(bot1: Bot, bot2: Bot, num_games: int = 100, seed: int = 42, num_processes: int = None) -> Dict[str, float]:
    """Run a match between two bots using multiple CPU cores."""
    # For bots that don't support multiprocessing, fall back to sequential execution
    if (isinstance(bot1, (MCCFRBot, DeepCFRBot)) or 
        isinstance(bot2, (MCCFRBot, DeepCFRBot))):
        bot_type = type(bot1).__name__ if isinstance(bot1, (MCCFRBot, DeepCFRBot)) else type(bot2).__name__
        print(f"{bot_type} detected - running sequentially for compatibility")
        results = [play_single_game((bot1, bot2, seed + i)) for i in range(num_games)]
    else:
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Create a list of game arguments
        game_args = [(bot1, bot2, seed + i) for i in range(num_games)]
        
        # Run games in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(play_single_game, game_args)
    
    # Aggregate results
    bot1_wins = sum(r["winner"] for r in results)
    bot1_points = sum(r["points_bot1"] for r in results)
    bot2_points = sum(r["points_bot2"] for r in results)
    total_moves = sum(r["total_points"] for r in results) // 10  # Rough estimate
    
    return {
        "win_rate": bot1_wins / num_games,
        "avg_points_bot1": bot1_points / num_games,
        "avg_points_bot2": bot2_points / num_games,
        "avg_game_length": total_moves / num_games
    }

def create_bot_configs() -> List[Dict[str, Any]]:
    """Create different bot configurations to test."""
    configs = []
    
    # MCTS configurations (prioritized) - reduced parameter space
    replay_memory_dir = pathlib.Path("ML_replay_memories")
    replay_memory_dir.mkdir(parents=True, exist_ok=True)
    
    # Reduced parameter combinations for faster testing
    mcts_params = [
        (4, 16),  # Fast baseline
        (8, 32),  # Medium complexity
        (12, 32)  # High depth, medium samples
    ]
    
    for depth, samples in mcts_params:
        configs.append({
            "bot_type": "MCTSbot",
            "params": {
                "replay_memory_file_path": replay_memory_dir / f"mcts_d{depth}_s{samples}.csv",
                "num_samples": samples,
                "depth": depth,
            }
        })
    
    # MCCFRBot configurations - reduced iterations
    for iterations in [1000, 5000]:  # Removed 10000 to save time
        configs.append({
            "bot_type": "MCCFRBot",
            "params": {
                "training_iterations": iterations,
                "model_path": f"models/mccfr_bot_{iterations}.pkl"
            }
        })
    
    # DeepCFR configurations - reduced sizes
    input_size = 173
    action_size = 20
    
    for hidden_size in [64, 128]:  # Removed 256 to save time
        configs.append({
            "bot_type": "DeepCFRBot",
            "params": {
                "regret_net": RegretNetwork(input_size, action_size, hidden_size=hidden_size),
                "strategy_net": StrategyNetwork(input_size, action_size, hidden_size=hidden_size)
            }
        })
    
    # RdeepBot configurations - minimal testing
    configs.append({
        "bot_type": "RdeepBot",
        "params": {
            "num_samples": 8,
            "depth": 4
        }
    })
    
    # Add RandBot as baseline
    configs.append({
        "bot_type": "RandBot",
        "params": {}
    })
    
    return configs

def create_bot(config: Dict[str, Any], seed: int = 42) -> Bot:
    """Create a bot instance from configuration."""
    rng = random.Random(seed)
    
    if config["bot_type"] == "RdeepBot":
        return RdeepBot(**config["params"], rand=rng)
    elif config["bot_type"] == "MCTSbot":
        return MCTSbot(**config["params"], rand=rng)
    elif config["bot_type"] == "MCCFRBot":
        bot = MCCFRBot()
        # Train the bot if needed
        if not pathlib.Path(config["params"]["model_path"]).exists():
            print(f"Training MCCFRBot for {config['params']['training_iterations']} iterations...")
            engine = SchnapsenGamePlayEngine()
            for _ in range(config["params"]["training_iterations"]):
                engine.play_game(bot, RandBot(rng), rng)
            bot.save_model(config["params"]["model_path"])
        else:
            bot.load_model(config["params"]["model_path"])
        return bot
    elif config["bot_type"] == "DeepCFRBot":
        return DeepCFRBot(**config["params"])
    elif config["bot_type"] == "RandBot":
        return RandBot(rng)
    else:
        raise ValueError(f"Unknown bot type: {config['bot_type']}")

def run_tournament_parallel(configs: List[Dict[str, Any]], num_games: int = 100, seed: int = 42, num_processes: int = None, time_limit_minutes: int = 30) -> pd.DataFrame:
    """Run a tournament between all bot configurations using multiple CPU cores."""
    results = []
    total_comparisons = len(configs) * (len(configs) - 1) // 2
    comparison_count = 0
    
    start_time = time.time()
    time_limit_seconds = time_limit_minutes * 60
    
    # Get all unique pairs of configurations
    config_pairs = list(combinations(enumerate(configs), 2))
    
    # Estimate time per comparison from first few games
    if len(config_pairs) > 0:
        print("Running time estimation...")
        i, config1 = config_pairs[0][0]
        j, config2 = config_pairs[0][1]
        bot1 = create_bot(config1, seed)
        bot2 = create_bot(config2, seed)
        
        # Run a small batch to estimate time
        test_games = 10
        test_start = time.time()
        _ = run_match_parallel(bot1, bot2, test_games, seed, num_processes)
        test_time = time.time() - test_start
        
        # Estimate time per comparison
        estimated_time_per_comparison = (test_time / test_games) * num_games
        
        # Calculate how many comparisons we can do within the time limit
        max_comparisons = int(time_limit_seconds / estimated_time_per_comparison)
        
        print(f"Estimated time per comparison: {estimated_time_per_comparison:.1f} seconds")
        print(f"Can complete approximately {max_comparisons} comparisons in {time_limit_minutes} minutes")
        
        # Limit the number of comparisons
        config_pairs = config_pairs[:max_comparisons]
        print(f"Limited to {len(config_pairs)} comparisons to meet time constraint")
    
    for (i, config1), (j, config2) in config_pairs:
        # Check if we're still within time limit
        if time.time() - start_time > time_limit_seconds:
            print("\nTime limit reached. Stopping tournament.")
            break
            
        comparison_count += 1
        print(f"\nComparison {comparison_count}/{len(config_pairs)}")
        print(f"Testing {config1['bot_type']} vs {config2['bot_type']}")
        
        bot1 = create_bot(config1, seed)
        bot2 = create_bot(config2, seed)
        
        match_stats = run_match_parallel(bot1, bot2, num_games, seed, num_processes)
        
        results.append({
            "bot1_type": config1["bot_type"],
            "bot1_params": str(config1["params"]),
            "bot2_type": config2["bot_type"],
            "bot2_params": str(config2["params"]),
            **match_stats
        })
        
        # Print progress
        elapsed_time = time.time() - start_time
        remaining_time = time_limit_seconds - elapsed_time
        
        print(f"Win rate for {config1['bot_type']}: {match_stats['win_rate']:.2%}")
        print(f"Average game length: {match_stats['avg_game_length']:.1f} rounds")
        print(f"Time remaining: {remaining_time/60:.1f} minutes")
    
    return pd.DataFrame(results)

def generate_report(results_df: pd.DataFrame, output_file: str):
    """Generate a detailed report from the tournament results."""
    with open(output_file, 'w') as f:
        f.write("# Bot Tournament Results Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall Statistics
        f.write("## Overall Statistics\n\n")
        f.write("### Average Win Rates by Bot Type\n\n")
        
        bot_types = set(results_df['bot1_type'].unique()) | set(results_df['bot2_type'].unique())
        
        for bot_type in bot_types:
            # Calculate win rate when bot was bot1
            bot1_games = results_df[results_df['bot1_type'] == bot_type]
            bot1_wins = bot1_games['win_rate'].mean() if not bot1_games.empty else 0
            
            # Calculate win rate when bot was bot2
            bot2_games = results_df[results_df['bot2_type'] == bot_type]
            bot2_wins = 1 - bot2_games['win_rate'].mean() if not bot2_games.empty else 0
            
            # Calculate overall win rate
            total_games = len(bot1_games) + len(bot2_games)
            if total_games > 0:
                overall_win_rate = (bot1_wins * len(bot1_games) + bot2_wins * len(bot2_games)) / total_games
                f.write(f"- {bot_type}: {overall_win_rate:.2%}\n")
        
        # Detailed Matchup Results
        f.write("\n## Detailed Matchup Results\n\n")
        f.write("| Bot 1 | Parameters | Bot 2 | Parameters | Win Rate | Avg Game Length | Avg Points Bot 1 | Avg Points Bot 2 |\n")
        f.write("|--------|------------|--------|------------|-----------|----------------|-----------------|----------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['bot1_type']} | {row['bot1_params']} | {row['bot2_type']} | {row['bot2_params']} | {row['win_rate']:.2%} | {row['avg_game_length']:.1f} | {row['avg_points_bot1']:.1f} | {row['avg_points_bot2']:.1f} |\n")
        
        # MCTS Analysis
        f.write("\n## MCTS Parameter Analysis\n\n")
        mcts_results = results_df[
            ((results_df['bot1_type'] == 'MCTSbot') & (results_df['bot2_type'] == 'RandBot')) |
            ((results_df['bot2_type'] == 'MCTSbot') & (results_df['bot1_type'] == 'RandBot'))
        ]
        
        if not mcts_results.empty:
            f.write("### MCTS Performance vs RandBot by Parameters\n\n")
            f.write("| Depth | Samples | Win Rate | Avg Game Length |\n")
            f.write("|--------|----------|-----------|----------------|\n")
            
            for depth in [4, 8, 12]:
                for samples in [16, 32, 64]:
                    filtered = mcts_results[
                        (mcts_results['bot1_params'].str.contains(f"'depth': {depth}") & 
                         mcts_results['bot1_params'].str.contains(f"'num_samples': {samples}")) |
                        (mcts_results['bot2_params'].str.contains(f"'depth': {depth}") & 
                         mcts_results['bot2_params'].str.contains(f"'num_samples': {samples}"))
                    ]
                    
                    if not filtered.empty:
                        win_rate = filtered['win_rate'].mean() if filtered['bot1_type'].iloc[0] == 'MCTSbot' else 1 - filtered['win_rate'].mean()
                        avg_length = filtered['avg_game_length'].mean()
                        f.write(f"| {depth} | {samples} | {win_rate:.2%} | {avg_length:.1f} |\n")
        
        # MCCFR Analysis
        f.write("\n## MCCFR Analysis\n\n")
        mccfr_results = results_df[
            (results_df['bot1_type'] == 'MCCFRBot') | (results_df['bot2_type'] == 'MCCFRBot')
        ]
        
        if not mccfr_results.empty:
            f.write("### MCCFR Performance by Training Iterations\n\n")
            f.write("| Training Iterations | Win Rate vs RandBot | Avg Game Length |\n")
            f.write("|-------------------|-------------------|----------------|\n")
            
            for iterations in [1000, 5000]:
                filtered = mccfr_results[
                    (mccfr_results['bot1_params'].str.contains(f"'training_iterations': {iterations}")) |
                    (mccfr_results['bot2_params'].str.contains(f"'training_iterations': {iterations}"))
                ]
                
                if not filtered.empty:
                    vs_rand = filtered[
                        (filtered['bot1_type'] == 'RandBot') | (filtered['bot2_type'] == 'RandBot')
                    ]
                    if not vs_rand.empty:
                        win_rate = vs_rand['win_rate'].mean() if vs_rand['bot1_type'].iloc[0] == 'MCCFRBot' else 1 - vs_rand['win_rate'].mean()
                        avg_length = vs_rand['avg_game_length'].mean()
                        f.write(f"| {iterations} | {win_rate:.2%} | {avg_length:.1f} |\n")
        
        # DeepCFR Analysis
        f.write("\n## DeepCFR Analysis\n\n")
        deepcfr_results = results_df[
            (results_df['bot1_type'] == 'DeepCFRBot') | (results_df['bot2_type'] == 'DeepCFRBot')
        ]
        
        if not deepcfr_results.empty:
            f.write("### DeepCFR Performance by Network Size\n\n")
            f.write("| Hidden Size | Win Rate vs RandBot | Avg Game Length |\n")
            f.write("|-------------|-------------------|----------------|\n")
            
            for hidden_size in [64, 128]:
                filtered = deepcfr_results[
                    (deepcfr_results['bot1_params'].str.contains(f"hidden_size={hidden_size}")) |
                    (deepcfr_results['bot2_params'].str.contains(f"hidden_size={hidden_size}"))
                ]
                
                if not filtered.empty:
                    vs_rand = filtered[
                        (filtered['bot1_type'] == 'RandBot') | (filtered['bot2_type'] == 'RandBot')
                    ]
                    if not vs_rand.empty:
                        win_rate = vs_rand['win_rate'].mean() if vs_rand['bot1_type'].iloc[0] == 'DeepCFRBot' else 1 - vs_rand['win_rate'].mean()
                        avg_length = vs_rand['avg_game_length'].mean()
                        f.write(f"| {hidden_size} | {win_rate:.2%} | {avg_length:.1f} |\n")
        
        # Save raw data
        results_df.to_csv(output_file + ".csv", index=False)
        f.write("\n\nRaw data has been saved to " + output_file + ".csv")

def main():
    print("Starting bot tournament...")
    
    # Create configurations
    configs = create_bot_configs()
    print(f"Created {len(configs)} bot configurations")
    
    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Run tournament with 30-minute time limit
    results = run_tournament_parallel(
        configs, 
        num_games=50,  # Reduced number of games
        seed=42, 
        num_processes=num_cores,
        time_limit_minutes=30
    )
    
    # Generate report
    report_file = f"bot_tournament_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_report(results, report_file)
    
    print(f"\nTournament completed. Report generated: {report_file}")

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on different platforms
    mp.set_start_method('spawn', force=True)
    main() 