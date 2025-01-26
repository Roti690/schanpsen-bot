import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from schnapsen.game import Bot, PlayerPerspective, Move, GameState, SchnapsenGamePlayEngine
import random
from data_gen.data_generation import get_state_feature_vector, get_move_feature_vector

class RegretNetwork(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.net(x)

class StrategyNetwork(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepCFRBot(Bot):
    def __init__(self, regret_net: RegretNetwork, strategy_net: StrategyNetwork, name: Optional[str] = None):
        super().__init__(name)
        self.regret_net = regret_net
        self.strategy_net = strategy_net
        self.regret_net.eval()
        self.strategy_net.eval()
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        valid_moves = perspective.valid_moves()
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Get state features
        state_vector = get_state_feature_vector(perspective)
        state_tensor = torch.FloatTensor(state_vector)
        
        # Get strategy probabilities for each valid move
        move_probs = []
        for move in valid_moves:
            move_vector = get_move_feature_vector(move)
            move_tensor = torch.FloatTensor(move_vector)
            
            # Combine state and move features
            input_tensor = torch.cat([state_tensor, move_tensor])
            
            with torch.no_grad():
                prob = self.strategy_net(input_tensor.unsqueeze(0))
            move_probs.append(prob.item())
        
        # Normalize probabilities
        move_probs = np.array(move_probs)
        move_probs = move_probs / move_probs.sum()
        
        # Choose move based on strategy probabilities
        chosen_idx = np.random.choice(len(valid_moves), p=move_probs)
        return valid_moves[chosen_idx]

def load_txt_dataset(file_path: str, label_type: str = "regret") -> Tuple[np.ndarray, List[float]]:
    """Load and preprocess dataset from text file"""
    data = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                # Parse features
                features = [float(x) for x in parts[0].split()]
                
                # Parse label based on type
                if label_type == "regret":
                    label = float(parts[1])  # Regret value
                else:
                    label = float(parts[2])  # Strategy probability
                
                data.append(features)
                labels.append(label)
    
    return np.array(data), labels

def create_data_loader(features: np.ndarray, labels: List[float], batch_size: int, shuffle: bool = True):
    """Create PyTorch DataLoader from features and labels"""
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class DeepCFR:
    def __init__(self, input_size: int, action_size: int):
        self.regret_net = RegretNetwork(input_size, action_size)
        self.strategy_net = StrategyNetwork(input_size, action_size)
        
        self.regret_optimizer = optim.Adam(self.regret_net.parameters())
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters())
        
        self.regret_criterion = nn.MSELoss()
        self.strategy_criterion = nn.CrossEntropyLoss()
    
    def train_regret_network(self, data_loader, epochs: int):
        """Train the regret network"""
        self.regret_net.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, labels in data_loader:
                self.regret_optimizer.zero_grad()
                outputs = self.regret_net(features)
                loss = self.regret_criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.regret_optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Regret Network - Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}')
    
    def train_strategy_network(self, data_loader, epochs: int):
        """Train the strategy network"""
        self.strategy_net.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, labels in data_loader:
                self.strategy_optimizer.zero_grad()
                outputs = self.strategy_net(features)
                loss = self.strategy_criterion(outputs, labels)
                loss.backward()
                self.strategy_optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Strategy Network - Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}')

def apply_move_safely(game_state: GameState, move: Move) -> GameState:
    """Safely apply a move to a game state by creating a deep copy first"""
    new_state = game_state.clone()
    
    # Handle different move types
    if move.is_trump_exchange():
        # Handle trump exchange
        new_state.leader.hand.remove(move.jack)
        new_state.leader.hand.add(new_state.trump_card)
        new_state.trump_card = move.jack
    elif move.is_marriage():
        # Handle marriage moves
        if move.queen_card in new_state.leader.hand and move.king_card in new_state.leader.hand:
            # Both cards must be in hand for marriage
            pass  # Marriage doesn't remove cards
    else:
        # Handle regular moves
        if move.card in new_state.leader.hand:
            new_state.leader.hand.remove(move.card)
        elif move.card in new_state.follower.hand:
            new_state.follower.hand.remove(move.card)
    
    return new_state 