# Bot Tournament Results Report

Generated on: 2025-01-27 01:14:05

## Overall Statistics

### Average Win Rates by Bot Type

- DeepCFRBot: 7.00%
- RandBot: 13.33%
- MCCFRBot: 20.00%
- RdeepBot: 43.33%
- MCTSbot: 76.17%

## Detailed Matchup Results

| Bot 1 | Parameters | Bot 2 | Parameters | Win Rate | Avg Game Length | Avg Points Bot 1 | Avg Points Bot 2 |
|--------|------------|--------|------------|-----------|----------------|-----------------|----------------|
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | 32.00% | 0.1 | 0.4 | 1.0 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | 32.00% | 0.2 | 0.5 | 1.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | MCCFRBot | {'training_iterations': 1000, 'model_path': 'models/mccfr_bot_1000.pkl'} | 92.00% | 0.2 | 1.9 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | MCCFRBot | {'training_iterations': 5000, 'model_path': 'models/mccfr_bot_5000.pkl'} | 86.00% | 0.2 | 1.9 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 88.00% | 0.2 | 1.8 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 90.00% | 0.2 | 2.0 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | RdeepBot | {'num_samples': 8, 'depth': 4} | 54.00% | 0.1 | 0.8 | 0.7 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d4_s16.csv'), 'num_samples': 16, 'depth': 4} | RandBot | {} | 84.00% | 0.2 | 2.0 | 0.2 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | 38.00% | 0.1 | 0.6 | 1.0 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | MCCFRBot | {'training_iterations': 1000, 'model_path': 'models/mccfr_bot_1000.pkl'} | 94.00% | 0.2 | 2.0 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | MCCFRBot | {'training_iterations': 5000, 'model_path': 'models/mccfr_bot_5000.pkl'} | 92.00% | 0.2 | 2.0 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 94.00% | 0.2 | 1.9 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 94.00% | 0.2 | 2.0 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | RdeepBot | {'num_samples': 8, 'depth': 4} | 64.00% | 0.2 | 1.0 | 0.6 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d8_s32.csv'), 'num_samples': 32, 'depth': 8} | RandBot | {} | 96.00% | 0.2 | 2.0 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | MCCFRBot | {'training_iterations': 1000, 'model_path': 'models/mccfr_bot_1000.pkl'} | 86.00% | 0.2 | 1.8 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | MCCFRBot | {'training_iterations': 5000, 'model_path': 'models/mccfr_bot_5000.pkl'} | 90.00% | 0.2 | 1.9 | 0.1 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 96.00% | 0.2 | 2.1 | 0.0 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | DeepCFRBot | {'regret_net': RegretNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
  )
), 'strategy_net': StrategyNetwork(
  (fc): Sequential(
    (0): Linear(in_features=173, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=20, bias=True)
    (5): Softmax(dim=-1)
  )
)} | 96.00% | 0.2 | 2.0 | 0.0 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | RdeepBot | {'num_samples': 8, 'depth': 4} | 52.00% | 0.2 | 0.8 | 0.8 |
| MCTSbot | {'replay_memory_file_path': PosixPath('ML_replay_memories/mcts_d12_s32.csv'), 'num_samples': 32, 'depth': 12} | RandBot | {} | 80.00% | 0.2 | 1.7 | 0.2 |
| MCCFRBot | {'training_iterations': 1000, 'model_path': 'models/mccfr_bot_1000.pkl'} | MCCFRBot | {'training_iterations': 5000, 'model_path': 'models/mccfr_bot_5000.pkl'} | 56.00% | 0.1 | 0.9 | 0.6 |

## MCTS Parameter Analysis

### MCTS Performance vs RandBot by Parameters

| Depth | Samples | Win Rate | Avg Game Length |
|--------|----------|-----------|----------------|
| 4 | 16 | 84.00% | 0.2 |
| 8 | 32 | 96.00% | 0.2 |
| 12 | 32 | 80.00% | 0.2 |

## MCCFR Analysis

### MCCFR Performance by Training Iterations

| Training Iterations | Win Rate vs RandBot | Avg Game Length |
|-------------------|-------------------|----------------|

## DeepCFR Analysis

### DeepCFR Performance by Network Size

| Hidden Size | Win Rate vs RandBot | Avg Game Length |
|-------------|-------------------|----------------|


Raw data has been saved to bot_tournament_report_20250127_011405.md.csv