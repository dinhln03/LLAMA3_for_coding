from art import logo_blackjack
from replit import clear
import random

def deal_card():
    """Return random card"""
    cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    card = random.choice(cards)
    return card

def calculate_score(cards):
    """Take a list of cards and return the score"""
    if sum(cards) == 21 and len(cards) == 2:
        return 0
    if 11 in cards and sum(cards) > 21:
        cards.remove(11)
        cards.append(1)
    return sum(cards)

def compare(current_score_of_user, current_score_of_computer):
    if current_score_of_user > 21 and current_score_of_computer > 21:
        return "You went over. You lose"
    if current_score_of_user == current_score_of_computer:
        return "DRAW"
    elif current_score_of_computer == 0:
        return "You lose. Opponent has a blackjack"
    elif current_score_of_user == 0:
        return "You win with blackjack"
    elif current_score_of_user > 21:
        return "You went over. You lose"
    elif current_score_of_computer > 21:
        return "Opponent went over. You win"
    elif current_score_of_user > current_score_of_computer:
        return "You win"
    else:
        return "You lose"

def play_game():
    print(logo_blackjack)

    user_cards = []
    computer_cards = []
    is_game_over = False

    for i in range(2):
        user_cards.append(deal_card())
        computer_cards.append(deal_card())
    while not is_game_over:
        current_score_of_user = calculate_score(user_cards)
        current_score_of_computer = calculate_score(computer_cards)
        print(f"Your cards: {user_cards} and current score of yours: {current_score_of_user}")
        print(f"Computer's first card: [{computer_cards[0]}]")
        if current_score_of_user == 0 or current_score_of_computer == 0 or current_score_of_user > 21:
            is_game_over = True
        else:
            want_card = input("To get another card type 'y', to pass type 'n': ")
            if want_card == "y":
                user_cards.append(deal_card())
            else:
                is_game_over = True

    while current_score_of_computer != 0 and current_score_of_computer < 17:
        computer_cards.append(deal_card())
        current_score_of_computer = calculate_score(computer_cards)

    print(f"Your final hand: {user_cards} and final score: {current_score_of_user}")
    print(f"Computer's final hand: {computer_cards}, final score: {current_score_of_computer}")
    print(compare(current_score_of_user, current_score_of_computer))

while input("Do you want to play a game of blackjack? Type 'y' or 'n': ") == "y":
    clear()
    play_game()
