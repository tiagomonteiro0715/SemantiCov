import random

# from replit import clear


logo = """
.------.            _     _            _    _            _    
|A_  _ |.          | |   | |          | |  (_)          | |   
|( \/ ).-----.     | |__ | | __ _  ___| | ___  __ _  ___| | __
| \  /|K /\  |     | '_ \| |/ _` |/ __| |/ / |/ _` |/ __| |/ /
|  \/ | /  \ |     | |_) | | (_| | (__|   <| | (_| | (__|   < 
`-----| \  / |     |_.__/|_|\__,_|\___|_|\_\ |\__,_|\___|_|\_\\
      |  \/ K|                            _/ |                
      `------'                           |__/           
"""


def deal_card():
    """To choose a random card"""
    cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    return random.choice(cards)


def calculation_total(hand):
    if type(hand) != list:
        raise TypeError("calculation_total expects only list!")
    if any(type(item) is not int for item in hand):
        raise TypeError("The list you've passed to calculation_total should contain only integers!")

    if sum(hand) == 21 and len(hand) == 2:
        return 0
    if 11 in hand and sum(hand) > 21:
        hand.remove(11)
        hand.append(1)
    return sum(hand)


def compare(player_hand, computer_hand):
    if type(player_hand) != int or type(computer_hand) != int:
        raise TypeError("Compare function expects only integer values!")
    if player_hand == computer_hand:
        return "It's a draw"
    elif player_hand == 0:
        return "You win with a BlackJack"
    elif computer_hand == 0:
        return "Your opponent wins with a BlackJack"
    elif player_hand > 21:
        return "You went over. You lose!"
    elif computer_hand > 21:
        return "You win! Opponent went over."
    elif player_hand > computer_hand:
        return "You win"
    else:
        return "You lose"


def game_play(get_user_input=input):
    print(logo)

    player_hand = []
    computer_hand = []

    for _ in range(2):
        player_hand.append(deal_card())
        computer_hand.append(deal_card())

    is_game_over = False
    while not is_game_over:
        total_player = calculation_total(player_hand)
        total_computer = calculation_total(computer_hand)
        print(f"    Your hand: {player_hand} Total score: {total_player}.")
        print(f"    Computer's first card: {computer_hand[0]}")

        if total_player == 0 or total_computer == 0 or total_player > 21:
            is_game_over = True
        else:
            if get_user_input("Do you want to take another card? Type 'y' or 'n'.") == 'y':
                player_hand.append(deal_card())
            else:
                is_game_over = True

    while total_computer <= 17 and total_computer != 0:
        computer_hand.append(deal_card())
        total_computer = calculation_total(computer_hand)

    print(f"    Your final hand is: {player_hand} Total score: {total_player}")
    print(f"    Computer's final score: {computer_hand} Total score: {total_computer}")
    print(compare(total_player, total_computer))


if __name__ == "__main__":
    while input("Do you want to play a game of BlackJack? Type 'y' or 'n'. ") == 'y':
        # clear() - used in replit
        game_play()
