import random
from player import Player
from hand import Hand

class CPU(Player):
    def __init__(self, name: str):
        super().__init__(name)
        self.hand = Hand()

    def discard(self):
        if(self.hand == None or len(self.hand) <= 0):
            raise RuntimeError('No cards to discard')
        return self.hand.pop(random.randrange(len(self.hand)))

    def play(self, currentPlayPointLimit):
        print('{0}\'s Hand: {1}'.format(self.name, str(self.playHand)))
        if(self.playHand == None or len(self.playHand) <= 0):
            raise RuntimeError('No play hand was created or it is empty')
        playableCardIndexes = []
        for i, card in enumerate(self.playHand):
            if(card.valuePoints <= currentPlayPointLimit):
                playableCardIndexes.append(i)
        cardToPlayIndex = playableCardIndexes[random.randrange(len(playableCardIndexes))]
        return self.playHand.pop(cardToPlayIndex)