import torch
import torch.nn as nn
import torch.optim as optim
from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenDeckGenerator, GamePhase
from schnapsen.deck import Rank, Suit
from typing import Optional
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class RegretNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128):
        super(RegretNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class StrategyNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128):
        super(StrategyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size),
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
    def __init__(self, regret_net, strategy_net, device=None, name="DeepCFRBot"):
        super().__init__(name)
        self.regret_net = regret_net
        self.strategy_net = strategy_net
        self.device = device or torch.device("cpu")  # Default to CPU if no device is specified
        self.name = name

        # Move networks to the specified device
        self.regret_net.to(self.device)
        self.strategy_net.to(self.device)

    def __str__(self):
        return self.name

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move] = None) -> Move:
        """
        Decide the best move using the trained model.
        """
        # Get all valid moves
        valid_moves = perspective.valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        try:
            # Get the state vector
            state_vector = get_state_feature_vector(perspective)
            
            # Get leader's move vector
            leader_vector = get_move_feature_vector(leader_move)
            
            # Get move feature vectors for each valid move
            move_vectors = [get_move_feature_vector(move) for move in valid_moves]
            
            # Combine state and move features for each move
            input_data = [
                torch.tensor(state_vector + leader_vector + move_vector, dtype=torch.float32).to(self.device)
                for move_vector in move_vectors
            ]
            input_tensor = torch.stack(input_data)
            
            # Predict probabilities
            with torch.no_grad():
                probabilities = self.strategy_net(input_tensor).squeeze().cpu().numpy()
                
                # Ensure probabilities is 1D and same length as valid_moves
                if len(probabilities.shape) > 1:
                    probabilities = probabilities.mean(axis=0)
                if len(probabilities) != len(valid_moves):
                    probabilities = probabilities[:len(valid_moves)]
                
                # Ensure valid probability distribution
                probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
                probabilities_sum = probabilities.sum()
                if probabilities_sum > 0:
                    probabilities = probabilities / probabilities_sum
                else:
                    probabilities = np.ones_like(probabilities) / len(probabilities)
            
            # Select move with highest probability
            best_move_index = probabilities.argmax()
            return valid_moves[best_move_index]
            
        except Exception as e:
            print(f"Error in DeepCFRBot get_move: {str(e)}")
            # Fallback to random move if there's any error
            return random.choice(valid_moves)



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


def get_state_feature_vector(perspective: PlayerPerspective) -> list[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hoy encoding)
        - talon size (int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown)

        Important: This function should not include the move of this agent.
        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)
    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    player_score = perspective.get_my_score()
    # - points of this player (int)
    player_points = round(player_score.direct_points / 66, 1)
    # - pending points of this player (int)
    player_pending_points = round(player_score.pending_points / 40, 1)

    # add the features to the feature set
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]

    opponents_score = perspective.get_opponent_score()
    # - points of the opponent (int)
    opponents_points = round(opponents_score.direct_points / 66, 1)
    # - pending points of opponent (int)
    opponents_pending_points = round(opponents_score.pending_points / 40, 1)

    # add the features to the feature set
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]

    # - the trump suit (1-hot encoding)
    trump_suit = perspective.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
    # add this features to the feature set
    state_feature_list += trump_suit_one_hot

    # - phase of game (1-hot encoding)
    game_phase_encoded = [1, 0] if perspective.get_phase() == GamePhase.TWO else [0, 1]
    # add this features to the feature set
    state_feature_list += game_phase_encoded

    # - talon size (int)
    talon_size = perspective.get_talon_size() / 10
    # add this features to the feature set
    state_feature_list += [talon_size]

    # - if this player is leader (1-hot encoding)
    i_am_leader = [0, 1] if perspective.am_i_leader() else [1, 0]
    # add this features to the feature set
    state_feature_list += i_am_leader

    # gather all known deck information
    hand_cards = perspective.get_hand().cards
    trump_card = perspective.get_trump_card()
    won_cards = perspective.get_won_cards().get_cards()
    opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
    opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()
    # each card can either be i) on player's hand, ii) on player's won cards, iii) on opponent's hand, iv) on opponent's won cards
    # v) be the trump card or vi) in an unknown position -> either on the talon or on the opponent's hand
    # There are all different cases regarding card's knowledge, and we represent these 6 cases using one hot encoding vectors as seen bellow.

    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []

    for card in SchnapsenDeckGenerator().get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]
        # i) on player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]
        # ii) on player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]
        # iii) on opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]
        # iv) on opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]
        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]
        # vi) in an unknown position as it is invisible to this player. Thus, it is either on the talon or on the opponent's hand
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]
        # This list eventually develops to one long 1-dimensional numpy array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding
    # deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # add this features to the feature set
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

    return state_feature_list


def get_move_feature_vector(move: Optional[Move]) -> list[int]:
    """
        In case there isn't any move provided move to encode, we still need to create a "padding"-"meaningless" vector of the same size,
        filled with 0s, since the ML models need to receive input of the same dimensionality always.
        Otherwise, we create all the information of the move i) move type, ii) played card rank and iii) played card suit
        translate this information into one-hot vectors respectively, and concatenate these vectors into one move feature representation vector
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        move_type_one_hot_encoding: list[int]
        # in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1]
            card = move.queen_card
        #  in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0]
            card = move.jack
        #  in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0]
            card = move.card
        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding
        card_rank_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_rank(card.rank)
        card_suit_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_suit(card.suit)

    return move_type_one_hot_encoding_numpy_array + card_rank_one_hot_encoding_numpy_array + card_suit_one_hot_encoding_numpy_array


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> list[int]:
    """
    Translating the rank of a card into one hot vector encoding of size 13.
    """
    card_rank_one_hot: list[int]
    if card_rank == Rank.ACE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif card_rank == Rank.TWO:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif card_rank == Rank.THREE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif card_rank == Rank.FOUR:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif card_rank == Rank.FIVE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif card_rank == Rank.SIX:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif card_rank == Rank.SEVEN:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.EIGHT:
        card_rank_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.NINE:
        card_rank_one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.TEN:
        card_rank_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.JACK:
        card_rank_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.QUEEN:
        card_rank_one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.KING:
        card_rank_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise AssertionError("Provided card Rank does not exist!")
    return card_rank_one_hot


def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> list[int]:
    """
    Translating the suit of a card into one hot vector encoding of size 4.
    """
    card_suit_one_hot: list[int]
    if card_suit == Suit.HEARTS:
        card_suit_one_hot = [0, 0, 0, 1]
    elif card_suit == Suit.CLUBS:
        card_suit_one_hot = [0, 0, 1, 0]
    elif card_suit == Suit.SPADES:
        card_suit_one_hot = [0, 1, 0, 0]
    elif card_suit == Suit.DIAMONDS:
        card_suit_one_hot = [1, 0, 0, 0]
    else:
        raise ValueError("Suit of card was not found!")

    return card_suit_one_hot


def create_state_and_actions_vector_representation(perspective: PlayerPerspective, leader_move: Optional[Move],
                                                   follower_move: Optional[Move]) -> list[int]:
    """
    This function takes as input a PlayerPerspective variable, and the two moves of leader and follower,
    and returns a list of complete feature representation that contains all information
    """
    player_game_state_representation = get_state_feature_vector(perspective)
    leader_move_representation = get_move_feature_vector(leader_move)
    follower_move_representation = get_move_feature_vector(follower_move)
    print("State vector length:", len(player_game_state_representation)), len(leader_move_representation), len(follower_move_representation)
    return player_game_state_representation + leader_move_representation + follower_move_representation