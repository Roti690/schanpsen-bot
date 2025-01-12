import torch
import torch.nn as nn
from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase, SchnapsenDeckGenerator, RegularTrick, PartialTrick, Trick
from typing import Optional
from schnapsen.deck import Suit, Rank
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


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
        leader_vector = self.get_move_feature_vector(leader_move)
        move_vectors = [self.get_move_feature_vector(move) for move in valid_moves]


        # Combine state and move features for each move
        input_data = [
            torch.tensor(state_vector+ leader_vector + move_vector, dtype=torch.float32).to(self.device)
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
        player_points = player_score.direct_points
        # - pending points of this player (int)
        player_pending_points = player_score.pending_points

        # add the features to the feature set
        state_feature_list += [player_points]
        state_feature_list += [player_pending_points]

        opponents_score = perspective.get_opponent_score()
        # - points of the opponent (int)
        opponents_points = opponents_score.direct_points
        # - pending points of opponent (int)
        opponents_pending_points = opponents_score.pending_points

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
        talon_size = perspective.get_talon_size()
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

    def get_move_feature_vector(self, move: Optional[Move]) -> list[int]:
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}_epochs{epochs}_batch{batch_size}_lr{lr}.pt"
    model_path = os.path.join(model_path, model_filename)

    torch.save(model.model.state_dict(), model_path)  # Save only the Sequential model


    print(f"Model saved to {model_path}")