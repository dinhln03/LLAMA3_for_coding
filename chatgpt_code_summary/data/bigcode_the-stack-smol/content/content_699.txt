import random


class Card:

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.suit} {self.rank}: {BlackJack.values[self.rank]}"


class Hand:
    def __init__(self):
        self.cards = []  # start with empty list
        self.value = 0
        self.aces = 0

    def adjust_for_ace(self):
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1

    def add_card(self, card):
        self.cards.append(card)
        self.value += BlackJack.values[card.rank]

        if card.rank == 'Ace':
            self.aces += 1

    def __str__(self):
        return f"Current Hand:{self.cards}\nCurrent Value:{self.value}\nCurrent Aces:{self.aces}\n"


class Deck:

    def __init__(self, card_game):

        self.game = card_game

        # create deck with all 52 cards
        self.cards = list()
        for suit in self.game.suits:
            for rank in self.game.ranks:
                self.cards.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()

    def __str__(self):
        return f"{[x for x in self.cards]}"


class Chips:
    def __init__(self, total=100):
        self.total = total
        self.bet = 0

    def win_bet(self):
        self.total += self.bet
        self.bet = 0

    def lose_bet(self):
        self.total -= self.bet
        self.bet = 0

    def make_bet(self, bet):
        if bet <= self.total:
            self.bet = bet
        else:
            raise ValueError(f"The bet ({bet}) exceeds available chips ({self.total})")

    def __str__(self):
        return f"Total: {self.total}\nCurrent Bet:{self.bet}\n"


class Player:
    def __init__(self, name):
        self.name = name
        self.wins = 0
        self.lost_games = 0
        self.chips = Chips()

    def __str__(self):
        return f"{self.name}:\n{self.wins} wins\n{self.lost_games} losses\nChips:{self.chips}\n"


class BlackJack:

    suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
    ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
    values = {'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10,
              'Jack': 10, 'Queen': 10, 'King': 10, 'Ace': 11}

    def __init__(self, player):
        self.player = player
        self.deck = Deck(self)
        self.playing = False

    def greeting(self):
        print("WELCOME TO BLACKJACK!")

    def take_bet(self):

        while True:
            try:
                # Ask the Player for their bet
                bet = int(input("Please put your bet: "))

                # Make sure that the Player's bet does not exceed their available chips
                self.player.chips.make_bet(bet)

                break
            except TypeError:
                print("Invalid input. Please try again")
            except ValueError as exc:
                print(f"{exc} Please try again")

    def hit(self, hand):
        cd = self.deck.deal_card()
        # print(f"Deal Card: {cd}")
        hand.add_card(cd)
        hand.adjust_for_ace()

    def hit_or_stand(self, hand):
        while True:
            print(f"{self.player.name}: current {hand.value}")
            action = input("Hit or Stand? Enter 'h' or 's': ")
            if action[0].lower() == 's':
                print("STAY\n")
                self.playing = False
            elif action[0].lower() == 'h':
                print("HIT\n")
                self.hit(hand)
            else:
                print(f"Sorry, I do not understand your choice '{action}'. Please try again")
                continue
            break

    def player_busts(self, p_hand, d_hand):
        print(f"[P={p_hand.value},D={d_hand.value}]: {self.player.name} BUSTED!")
        self.player.chips.lose_bet()
        self.player.lost_games += 1

    def player_wins(self, p_hand, d_hand):
        print(f"[P={p_hand.value},D={d_hand.value}]: {self.player.name} WINS! ")
        self.player.chips.win_bet()
        self.player.wins += 1

    def dealer_busts(self, p_hand, d_hand):
        print(f"[P={p_hand.value},D={d_hand.value}]: {self.player.name} WINS - Dealer BUSTED!")
        self.player.chips.win_bet()
        self.player.wins += 1

    def dealer_wins(self, p_hand, d_hand):
        print(f"[P={p_hand.value},D={d_hand.value}]: Dealer WINS")
        self.player.chips.lose_bet()
        self.player.lost_games += 1

    def push(self, p_hand, d_hand):
        print(f"[P={p_hand.value},D={d_hand.value}]: Dealer and {self.player.name} tie - PUSH!")

    def show_some(self, p_hand, d_hand):
        # Show only one of the Dealer's cards, the other remains hidden
        print(f"Dealer's card (one hidden): {d_hand.cards[0]}")

        # Show both of the Player's cards
        print(f"{self.player.name}'s Cards:")
        for card in p_hand.cards:
            print(card)
        print(f"total= {p_hand.value}")

    def show_all_cards(self, p_hand, d_hand):
        # Show both of the Player's cards
        print(f"{self.player.name}'s Cards:")
        for card in p_hand.cards:
            print(card)
        print(f"total= {p_hand.value}")

        # Show both of the Player's cards
        print(f"Dealer's Cards:")
        for card in d_hand.cards:
            print(card)
        print(f"total= {d_hand.value}")

    def play(self):
        """
        # 1. Create a deck of 52 cards
        # 2. Shuffle the deck
        # 3. Ask the Player for their bet
        # 4. Make sure that the Player's bet does not exceed their available chips
        # 5. Deal two cards to the Dealer and two cards to the Player
        # 6. Show only one of the Dealer's cards, the other remains hidden
        # 7. Show both of the Player's cards
        # 8. Ask the Player if they wish to Hit, and take another card
        # 9. If the Player's hand doesn't Bust (go over 21), ask if they'd like to Hit again.
        # 10. If a Player Stands, play the Dealer's hand.
        #     The dealer will always Hit until the Dealer's value meets or exceeds 17
        # 11. Determine the winner and adjust the Player's chips accordingly
        # 12. Ask the Player if they'd like to play again
        """
        print("--NEW GAME---")
        self.playing = True
        self.deck.shuffle()

        dealer_hand = Hand()
        player_hand = Hand()

        # Deal two cards to the Dealer and two cards to the Player
        player_hand.add_card(self.deck.deal_card())
        dealer_hand.add_card(self.deck.deal_card())
        player_hand.add_card(self.deck.deal_card())
        dealer_hand.add_card(self.deck.deal_card())

        self.take_bet()

        # show cards, but keep one dealer card hidden
        self.show_some(player_hand, dealer_hand)
        while self.playing:
            # Ask the Player if they wish to Hit, and take another card
            # If the Player's hand doesn't Bust (go over 21), ask if they'd like to Hit again.
            self.hit_or_stand(player_hand)

            self.show_some(player_hand, dealer_hand)

            if player_hand.value > 21:
                # player busts -  lost his bet
                self.player_busts(player_hand, dealer_hand)
                break

        # If Player has not busted
        if player_hand.value <= 21:

            # The dealer will always Hit until the Dealer's value meets or exceeds 17
            while dealer_hand.value < 17:
                self.hit(dealer_hand)

            # Determine for the winner - show all cards
            self.show_all_cards(player_hand, dealer_hand)

            # Determine the winner and adjust the Player's chips accordingly
            if dealer_hand.value > 21:
                self.dealer_busts(player_hand, dealer_hand)
            elif player_hand.value > dealer_hand.value:
                self.player_wins(player_hand, dealer_hand)
            elif player_hand.value < dealer_hand.value:
                self.dealer_wins(player_hand, dealer_hand)
            else:
                self.push(player_hand, dealer_hand)


if __name__ == "__main__":

    game_on = True

    # Play a new game of BlackJack with Player Daniela
    player = Player('Daniela')
    game = BlackJack(player)
    game.greeting()
    while game_on:
        game.play()
        print(f"GAME DONE.\nGame Stats:\n\n{player}")

        # Ask the Player if they'd like to play again
        if input("Would you like another game? y/n: ") != 'y':
            game_on = False
