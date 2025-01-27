import random
import torch
import numpy as np
from collections import defaultdict
import logging
import wandb
from accelerate import Accelerator
from .networks import RegretNet, StrategyNet

class DeepMCCFR:
    """
    Deep Monte Carlo Counterfactual Regret Minimization agent with baseline variance reduction.
    """
    def __init__(self, env, num_actions=3, episodes=2000, batch_size=32, lr=1e-3, use_wandb=True, distributed=True):
        self.env = env
        self.num_actions = num_actions
        self.episodes = episodes
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        
        # Setup distributed training if requested
        if distributed:
            self.setup_distributed()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dictionary-based regrets & strategy
        self.regret_table = defaultdict(lambda: np.zeros(num_actions, dtype=np.float32))
        self.strategy_table = defaultdict(lambda: np.ones(num_actions, dtype=np.float32)/num_actions)
        
        # Baseline table for variance reduction
        self.baseline_table = defaultdict(float)
        self.baseline_count = defaultdict(int)

        # Create neural nets (1 dimension for player ID + 4 environment features => 5)
        self.regret_net = RegretNet(input_dim=5, hidden_dim=64, num_actions=num_actions).to(self.device)
        self.strategy_net = StrategyNet(input_dim=5, hidden_dim=64, num_actions=num_actions).to(self.device)

        self.regret_opt = torch.optim.Adam(self.regret_net.parameters(), lr=lr)
        self.strategy_opt = torch.optim.Adam(self.strategy_net.parameters(), lr=lr)

        # Replay buffers
        self.regret_replay = []
        self.strategy_replay = []
        
        if self.use_wandb:
            wandb.init(project="deep-mccfr-schnapsen", config={
                "num_actions": num_actions,
                "episodes": episodes,
                "batch_size": batch_size,
                "learning_rate": lr,
                "distributed": distributed
            })
        
        logging.info(f"Initialized DeepMCCFR agent on device: {self.device}")

    def setup_distributed(self):
        """Setup distributed training across GPUs"""
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.regret_net, self.strategy_net = self.accelerator.prepare(
            self.regret_net, self.strategy_net
        )
        self.regret_opt, self.strategy_opt = self.accelerator.prepare(
            self.regret_opt, self.strategy_opt
        )

    def _make_features(self, info_set):
        player, feat_tuple = info_set
        feats = np.array(feat_tuple)
        return np.concatenate([[player], feats], axis=0)

    def update_baseline_table(self, info_set, payoff):
        current_avg = self.baseline_table[info_set]
        count = self.baseline_count[info_set]
        new_avg = (current_avg * count + payoff) / (count + 1)
        self.baseline_table[info_set] = new_avg
        self.baseline_count[info_set] = count + 1

    def sample_trajectory(self):
        state = self.env.get_initial_state()
        trajectory = []
        done = False
        while not done:
            current_player, _ = state
            info_set = self.env.get_info_set(state)
            strategy = self.strategy_table[info_set]
            action = np.random.choice(self.num_actions, p=strategy)
            
            next_state, reward, done = self.env.step(state, action)
            step_info = {
                'player': current_player,
                'info_set': info_set,
                'action': action,
                'reward': reward
            }
            trajectory.append(step_info)
            state = next_state

        # final payoff from perspective of last actor
        last_actor = trajectory[-1]['player']
        final_reward = trajectory[-1]['reward']
        payoff = {}
        payoff[last_actor] = final_reward
        payoff[1 - last_actor] = -final_reward

        for step in trajectory:
            step['final_payoff_for_player'] = payoff[step['player']]
        return trajectory

    def compute_and_update_regrets(self, trajectory):
        for step in trajectory:
            info_set = step['info_set']
            action = step['action']
            final_payoff = step['final_payoff_for_player']
            
            # Update baseline with the newly observed payoff
            self.update_baseline_table(info_set, final_payoff)
            baseline_value = self.baseline_table[info_set]
            
            # Compute advantage using baseline
            advantage = final_payoff - baseline_value
            
            old_regrets = self.regret_table[info_set]
            new_regrets = old_regrets.copy()
            new_regrets[action] += advantage
            self.regret_table[info_set] = new_regrets

    def regret_matching(self):
        for info_set, regrets in self.regret_table.items():
            clipped = np.maximum(regrets, 0.0)
            denom = clipped.sum()
            if denom > 1e-9:
                new_strategy = clipped / denom
            else:
                new_strategy = np.ones(self.num_actions)/self.num_actions
            self.strategy_table[info_set] = new_strategy

            # Save for strategy net training
            feat = self._make_features(info_set)
            self.strategy_replay.append((feat, new_strategy))

    def fill_regret_replay(self):
        for info_set, regrets in self.regret_table.items():
            feat = self._make_features(info_set)
            self.regret_replay.append((feat, regrets.copy()))

    def external_sampling_trajectory(self):
        """Sample trajectory using external sampling for opponent actions"""
        state = self.env.get_initial_state()
        trajectory = []
        done = False
        while not done:
            current_player, _ = state
            info_set = self.env.get_info_set(state)
            
            # Use strategy for current player, random for opponent
            if current_player == 0:  # Main player
                strategy = self.strategy_table[info_set]
                action = np.random.choice(self.num_actions, p=strategy)
            else:  # Opponent - uniform random
                action = np.random.choice(self.num_actions)
            
            next_state, reward, done = self.env.step(state, action)
            step_info = {
                'player': current_player,
                'info_set': info_set,
                'action': action,
                'reward': reward
            }
            trajectory.append(step_info)
            state = next_state

        last_actor = trajectory[-1]['player']
        final_reward = trajectory[-1]['reward']
        payoff = {}
        payoff[last_actor] = final_reward
        payoff[1 - last_actor] = -final_reward

        for step in trajectory:
            step['final_payoff_for_player'] = payoff[step['player']]
        return trajectory

    def compute_counterfactual_values(self, info_set, action):
        """Compute counterfactual values via partial re-traversals"""
        num_samples = 10  # Number of re-traversals
        values = []
        
        for _ in range(num_samples):
            # Start from the info set and simulate forward
            state = self.env.get_state_from_info_set(info_set)
            next_state, reward, done = self.env.step(state, action)
            
            if done:
                values.append(reward)
                continue
                
            # Continue with regular rollout
            while not done:
                current_player, _ = next_state
                next_info_set = self.env.get_info_set(next_state)
                strategy = self.strategy_table[next_info_set]
                next_action = np.random.choice(self.num_actions, p=strategy)
                next_state, reward, done = self.env.step(next_state, next_action)
            
            values.append(reward)
            
        return np.mean(values)

    def log_training_metrics(self, episode, metrics):
        """Log detailed training metrics"""
        if self.use_wandb:
            wandb.log({
                'episode': episode,
                'regret_loss': metrics['regret_loss'],
                'strategy_loss': metrics['strategy_loss'],
                'avg_payoff': metrics['avg_payoff'],
                'baseline_values': metrics['baseline_values'],
                'exploration_rate': metrics['exploration_rate']
            })
        
        logging.info(
            f"Episode {episode}/{self.episodes} - "
            f"Regret Loss: {metrics['regret_loss']:.4f}, "
            f"Strategy Loss: {metrics['strategy_loss']:.4f}, "
            f"Avg Payoff: {metrics['avg_payoff']:.4f}"
        )

    def run_training(self):
        for ep in range(1, self.episodes+1):
            # Use external sampling with 50% probability
            if random.random() < 0.5:
                traj = self.external_sampling_trajectory()
            else:
                traj = self.sample_trajectory()
                
            self.compute_and_update_regrets(traj)
            self.regret_matching()
            self.fill_regret_replay()
            
            metrics = self.train_networks()
            
            if ep % 200 == 0:
                avg_payoff = self.evaluate_selfplay(num_games=100)
                metrics['avg_payoff'] = avg_payoff
                metrics['exploration_rate'] = max(0.1, 1.0 - ep/self.episodes)
                self.log_training_metrics(ep, metrics)

    def train_networks(self):
        metrics = {'regret_loss': 0.0, 'strategy_loss': 0.0}
        
        # Train regret_net
        if len(self.regret_replay) >= self.batch_size:
            batch = random.sample(self.regret_replay, self.batch_size)
            feats, targets = [], []
            for (f, r) in batch:
                feats.append(f)
                targets.append(r)
            feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

            preds = self.regret_net(feats_t)
            loss_regret = ((preds - targets_t)**2).mean()
            metrics['regret_loss'] = loss_regret.item()

            self.regret_opt.zero_grad()
            if hasattr(self, 'accelerator'):
                self.accelerator.backward(loss_regret)
            else:
                loss_regret.backward()
            self.regret_opt.step()

        # Train strategy_net
        if len(self.strategy_replay) >= self.batch_size:
            batch = random.sample(self.strategy_replay, self.batch_size)
            feats, targets = [], []
            for (f, dist) in batch:
                feats.append(f)
                targets.append(dist)
            feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

            pred_probs = self.strategy_net(feats_t)
            loss_strat = ((pred_probs - targets_t)**2).mean()
            metrics['strategy_loss'] = loss_strat.item()

            self.strategy_opt.zero_grad()
            if hasattr(self, 'accelerator'):
                self.accelerator.backward(loss_strat)
            else:
                loss_strat.backward()
            self.strategy_opt.step()
            
        return metrics

    def evaluate_selfplay(self, num_games=500):
        total_p0 = 0.0
        for _ in range(num_games):
            traj = self.sample_trajectory()
            last_actor = traj[-1]['player']
            final_reward = traj[-1]['reward']
            payoff_p0 = final_reward if last_actor == 0 else -final_reward
            total_p0 += payoff_p0
        return total_p0 / num_games

    def save_model(self, path):
        torch.save({
            'regret_net_state_dict': self.regret_net.state_dict(),
            'strategy_net_state_dict': self.strategy_net.state_dict(),
            'regret_opt_state_dict': self.regret_opt.state_dict(),
            'strategy_opt_state_dict': self.strategy_opt.state_dict()
        }, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.regret_net.load_state_dict(checkpoint['regret_net_state_dict'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net_state_dict'])
        self.regret_opt.load_state_dict(checkpoint['regret_opt_state_dict'])
        self.strategy_opt.load_state_dict(checkpoint['strategy_opt_state_dict'])
        logging.info(f"Model loaded from {path}")

    def get_move(self, state_tensor, leader_move=None):
        """Get a move prediction for the given state."""
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            probs = self.strategy_net(state_tensor)
            action = torch.argmax(probs).item()
            probability = probs[action].item()
            return action, probability 