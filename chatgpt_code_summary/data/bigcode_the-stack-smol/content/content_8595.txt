from .player import Player

class Cell:
    EMPTY = ' '

    def __init__(self, value = EMPTY):
        self.value = value

    def __eq__(self, other):
        return other is not None and self.value == other.value

    def __str__(self) -> str:
        return str(self.value)

    def assign(self, player: Player):
        self.value = player.symbol