import torch
import torch.nn as nn
from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase, SchnapsenDeckGenerator, RegularTrick, PartialTrick, Trick
from typing import Optional
from schnapsen.deck import Suit, Rank
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class DeepLearningBot(Bot):
    def __init__(self, model_path: str, input_size: int, hidden_size: int, name: Optional[str] = None):
        """
        Initialize the DeepLearningBot with a trained PyTorch model.

        :param model_path: Path to the trained PyTorch model.
        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layers.
        """
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_size, hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _build_model(self, input_size: int, hidden_size: int):
        """
        Define the neural network architecture.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in hidden layers.
        """
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        """
        Decide the best move using the trained model.

        :param perspective: The player's perspective of the game state.
        :param leader_move: The move made by the leader, if any.
        :return: The selected Move.
        """
        state_vector = self.get_state_feature_vector(perspective)
        valid_moves = perspective.valid_moves()
        move_vectors = [self.get_move_feature_vector(move) for move in valid_moves]


        # Combine state and move features for each move
        input_data = [
            torch.tensor(state_vector + move_vector, dtype=torch.float32).to(self.device)
            for move_vector in move_vectors
        ]
        input_tensor = torch.stack(input_data)

        # Predict probabilities
        with torch.no_grad():
            probabilities = self.model(input_tensor).squeeze().cpu().numpy()

        # Select the move with the highest probability
        best_move_index = probabilities.argmax()
        return valid_moves[best_move_index]

    def get_state_feature_vector(self, perspective: PlayerPerspective) -> list[int]:
        """
        Convert the game state into a feature vector.

        :param perspective: The player's perspective of the game state.
        :return: A feature vector representing the game state.
        """
        state_feature_list = []

        # Add points and pending points (4 features)
        state_feature_list += [
            perspective.get_my_score().direct_points,
            perspective.get_my_score().pending_points,
            perspective.get_opponent_score().direct_points,
            perspective.get_opponent_score().pending_points,
        ]

        # Add trump suit encoding (4 features)
        trump_suit = perspective.get_trump_suit()
        state_feature_list += self.get_one_hot_encoding_of_card_suit_method(trump_suit)

        # Add game phase encoding (2 features)
        game_phase_encoded = [1, 0] if perspective.get_phase() == perspective.get_phase().TWO else [0, 1]
        state_feature_list += game_phase_encoded

        # Add talon size and leader status (2 features)
        state_feature_list += [
            perspective.get_talon_size(),
            int(perspective.am_i_leader())
        ]

        # Add deck knowledge (120 features)
        hand_cards = perspective.get_hand().cards
        trump_card = perspective.get_trump_card()
        won_cards = perspective.get_won_cards().get_cards()
        opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
        opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()

        deck_knowledge_in_consecutive_one_hot_encodings = []
        for card in SchnapsenDeckGenerator().get_initial_deck():
            if card in hand_cards:
                card_knowledge = [0, 0, 0, 0, 0, 1]
            elif card in won_cards:
                card_knowledge = [0, 0, 0, 0, 1, 0]
            elif card in opponent_known_cards:
                card_knowledge = [0, 0, 0, 1, 0, 0]
            elif card in opponent_won_cards:
                card_knowledge = [0, 0, 1, 0, 0, 0]
            elif card == trump_card:
                card_knowledge = [0, 1, 0, 0, 0, 0]
            else:
                card_knowledge = [1, 0, 0, 0, 0, 0]
            deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge

        state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

        # Add leader's move (20 features)
        if perspective.am_i_leader():
            leader_move = Trick.leader_move()
            state_feature_list += self.get_move_feature_vector(leader_move)
        else:
            state_feature_list += [0] * 20  # Placeholder for no leader move

        # Add follower's move (20 features)
        if not perspective.am_i_leader():
            follower_move = round_trick.follower_move
            state_feature_list += self.get_move_feature_vector(follower_move)
        else:
            state_feature_list += [0] * 20  # Placeholder for no follower move

        return state_feature_list

    def get_move_feature_vector(self, move: Optional[Move]) -> list[int]:
        """
        Convert a move into a feature vector.

        :param move: The move to convert.
        :return: A feature vector representing the move.
        """
        if move is None:
            return [0] * 20  # Placeholder for no move

        move_vector = []
        if move.is_marriage():
            move_vector += [1, 0, 0]
        elif move.is_trump_exchange():
            move_vector += [0, 1, 0]
        else:
            move_vector += [0, 0, 1]

        card = move.card if not move.is_marriage() else move.queen_card
        move_vector += self.get_one_hot_encoding_of_rank(card.rank)
        move_vector += self.get_one_hot_encoding_of_suit(card.suit)

        return move_vector
    

    def get_one_hot_encoding_of_card_suit_method(self, card_suit: Suit) -> list[int]:
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
    
    def get_one_hot_encoding_of_card_rank_method(self, card_rank: Rank) -> list[int]:
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


class SchnapsenNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SchnapsenNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)

# Load data from file
def load_data(file_path):
    features, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            feature_str, label_str = line.strip().split(" || ")
            features.append([float(x) for x in feature_str.split(",")])
            labels.append(float(label_str))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Training script
def train_DL_model(data_path, model_path, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):
    # Load data
    features, labels = load_data(data_path)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Reshape labels to match output dimensions
    y_train = y_train.view(-1, 1)
    y_val = y_val.view(-1, 1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SchnapsenNet(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).view(-1, 1)  # Ensure output shape [batch_size, 1]
            loss = criterion(outputs, batch_labels)  # Ensure labels match [batch_size, 1]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features).view(-1, 1)  # Ensure output shape [batch_size, 1]
                loss = criterion(outputs, batch_labels)  # Ensure labels match [batch_size, 1]
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # Save the trained model
    # Save the model
    torch.save(model.model.state_dict(), model_path)  # Save only the Sequential model


    print(f"Model saved to {model_path}")