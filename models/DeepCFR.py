import torch
import torch.nn as nn
import torch.optim as optim
from schnapsen.game import Bot, PlayerPerspective, Move
from typing import Optional
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class RegretNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(RegretNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class StrategyNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(StrategyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class DeepCFR:
    def __init__(self, input_size, action_size):
        self.regret_net = RegretNetwork(input_size, action_size)
        self.strategy_net = StrategyNetwork(input_size, action_size)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=0.001)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.001)

    def train_regret_network(self, data_loader, epochs=10):
        regret_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for states, regrets in data_loader:
                self.regret_optimizer.zero_grad()

                predictions = self.regret_net(states)

                # Ensure shapes match
                if predictions.shape != regrets.shape:
                    regrets = regrets.view(-1, predictions.shape[1])

                loss = regret_loss_fn(predictions, regrets)
                loss.backward()
                self.regret_optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}, Regret Loss: {epoch_loss / len(data_loader):.4f}")


    def train_strategy_network(self, data_loader, epochs=10):
        strategy_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for states, strategies in data_loader:
                self.strategy_optimizer.zero_grad()

                # Forward pass
                predictions = self.strategy_net(states)

                # Ensure shapes match
                if predictions.shape != strategies.shape:
                    strategies = strategies.view(-1, predictions.shape[1])

                # Compute loss
                loss = strategy_loss_fn(predictions, strategies)
                loss.backward()
                self.strategy_optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}, Strategy Loss: {epoch_loss / len(data_loader):.4f}")



class DeepCFRBot(Bot):
    def __init__(self, regret_net, strategy_net, name="DeepCFRBot"):
        super().__init__(name)
        self.regret_net = regret_net
        self.strategy_net = strategy_net

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        state_vector = torch.tensor([perspective.get_state_vector()], dtype=torch.float32)
        strategy = self.strategy_net(state_vector).detach().numpy()[0]
        valid_moves = perspective.valid_moves()

        # Select a move based on the strategy probabilities
        chosen_move = random.choices(valid_moves, weights=strategy, k=1)[0]
        return chosen_move


class DeepCFRDataset(torch.utils.data.Dataset):
    def __init__(self, states, labels):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]



def load_txt_dataset(file_path, label_type="regret"):
    """
    Loads the dataset from a text file and separates features and labels.
    
    :param file_path: Path to the text file.
    :param label_type: Type of label to extract ("regret" or "strategy").
    :return: Tuple of (features, labels).
    """
    features = []
    labels = []

    with open(file_path, "r") as file:
        for line in file:
            # Split line into features and label
            if "||" in line:
                feature_part, label_part = line.split("||")
            else:
                raise ValueError("Line does not contain '||' separator.")

            # Parse features (comma-separated)
            feature_vector = [float(value) for value in feature_part.split(",")]
            features.append(feature_vector)

            # Parse label
            if label_type == "regret":
                label_vector = process_regret_label(label_part.strip())
            elif label_type == "strategy":
                label_vector = process_strategy_label(label_part.strip())
            else:
                raise ValueError("Invalid label_type. Choose 'regret' or 'strategy'.")
            labels.append(label_vector)

    return np.array(features), np.array(labels)

def process_regret_label(label):
    """
    Converts a label into a regret vector (example).
    :param label: Raw label data as a string.
    :return: Processed regret vector.
    """
    # Example: Split by commas or other delimiters
    return [float(x) for x in label.split(",")]


def process_strategy_label(label):
    """
    Converts a label into a strategy vector (example).
    :param label: Raw label data as a string.
    :return: Processed strategy vector.
    """
    # Example: Split by commas or other delimiters
    return [float(x) for x in label.split(",")]



class DeepCFRDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initializes the dataset.
        :param features: Array of feature vectors.
        :param labels: Array of label vectors (regrets or strategies).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



def create_data_loader(features, labels, batch_size=32, shuffle=True):
    """
    Creates a PyTorch DataLoader for the given features and labels.
    :param features: Array of feature vectors.
    :param labels: Array of label vectors.
    :param batch_size: Batch size for training.
    :param shuffle: Whether to shuffle the data.
    :return: DataLoader instance.
    """
    dataset = DeepCFRDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
