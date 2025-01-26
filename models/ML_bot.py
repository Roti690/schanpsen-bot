from schnapsen.game import Bot, PlayerPerspective, Move
from typing import Optional
import pathlib
import pandas as pd
import numpy as np
from data_gen.data_generation import get_state_feature_vector, get_move_feature_vector
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class MLPlayingBot(Bot):
    """Bot that uses a trained ML model to make decisions"""
    
    def __init__(self, model_location: pathlib.Path, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.model_location = model_location
        # Load the trained model
        self.model = torch.load(str(model_location))
        self.model.eval()  # Set to evaluation mode

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # Get valid moves
        valid_moves = perspective.valid_moves()
        
        if len(valid_moves) == 1:
            return valid_moves[0]
            
        # Get state features
        state_vector = get_state_feature_vector(perspective)
        state_tensor = torch.FloatTensor(state_vector)
        
        # Evaluate each valid move
        move_scores = []
        for move in valid_moves:
            move_vector = get_move_feature_vector(move)
            move_tensor = torch.FloatTensor(move_vector)
            
            # Combine state and move features
            input_tensor = torch.cat([state_tensor, move_tensor])
            
            # Get model prediction
            with torch.no_grad():
                score = self.model(input_tensor.unsqueeze(0))
            move_scores.append(score.item())
        
        # Choose the move with highest score
        best_move_idx = np.argmax(move_scores)
        return valid_moves[best_move_idx]

def train_model(model_type: str = "NN") -> None:
    """Train a new model using the replay memory dataset"""
    
    # Setup paths
    replay_memory_dir = "ML_replay_memories"
    replay_memory_filename = "replay_memory.csv"
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename
    
    model_dir = "ML_models"
    model_name = "simple_model"
    model_location = pathlib.Path(model_dir) / model_name
    
    # Check if replay memory exists
    if not replay_memory_location.exists():
        raise ValueError(f"Dataset was not found at: {replay_memory_location} !")
        
    # Load and preprocess data
    data = pd.read_csv(replay_memory_location)
    
    # Convert string representations of lists to actual lists
    data['state_vector'] = data['state_vector'].apply(eval)
    data['leader_move_vec'] = data['leader_move_vec'].apply(eval)
    data['follower_move_vec'] = data['follower_move_vec'].apply(eval)
    
    # Prepare features and labels
    X = []
    y = []
    
    for _, row in data.iterrows():
        state_vec = row['state_vector']
        move_vec = row['leader_move_vec']
        # Combine state and move vectors
        features = state_vec + move_vec
        X.append(features)
        y.append(1 if row['did_win_game'] else 0)
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Create model
    input_size = len(X[0])
    if model_type == "NN":
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    # Training parameters
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    batch_size = 32
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save model
    model_location.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_location)
    print(f"Model saved at: {model_location}")

def train_DL_model(data_file: str, output_model_path: str, input_dim: int, hidden_dim: int, batch_size: int = 32, epochs: int = 10, lr: float = 0.001):
    """Train a deep learning model"""
    
    # Load data
    data = pd.read_csv(data_file)
    
    # Convert string representations of lists to actual lists
    data['state_vector'] = data['state_vector'].apply(eval)
    data['leader_move_vec'] = data['leader_move_vec'].apply(eval)
    
    # Prepare features and labels
    X = []
    y = []
    
    for _, row in data.iterrows():
        state_vec = row['state_vector']
        move_vec = row['leader_move_vec']
        features = state_vec + move_vec
        X.append(features)
        y.append(1 if row['did_win_game'] else 0)
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Get actual input dimension from data
    actual_input_dim = X.shape[1]
    
    # Create model with correct input dimension
    model = nn.Sequential(
        nn.Linear(actual_input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim//2),
        nn.ReLU(),
        nn.Linear(hidden_dim//2, 1),
        nn.Sigmoid()
    )
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_model_path}/model_{timestamp}_epochs{epochs}_batch{batch_size}_lr{lr}.pt"
    torch.save(model, model_path)
    print(f"Model saved at: {model_path}")
    return model_path 