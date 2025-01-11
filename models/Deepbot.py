from schnapsen.game import Bot, PlayerPerspective, Move, GameState, GamePlayEngine
import random
from typing import Optional


class DeepBot(Bot):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)