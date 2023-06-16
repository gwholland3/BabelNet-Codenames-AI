import random

from babelnet_bots.babelnet_bots import BabelNetSpymaster, BabelNetFieldOperative


WORDLIST_FILEPATH = 'wordlists/codenames_words.txt'


def main():
    # Retrieve list of words to draw from
    with open(WORDLIST_FILEPATH) as f:
        words = f.read().splitlines()

    # Generate a new, random board
    game_words, blue_words, red_words, bystanders, assassin = generate_new_board(words)

    print("Initializing bots...")
    spymaster_bot = BabelNetSpymaster(game_words)
    field_operative_bot = None

    # print_game_state(blue_words, red_words, bystanders, assassin)

    guessed_words = []
    lose = False
    while blue_words and not lose:
        print_board(game_words, guessed_words)
        if spymaster_bot:
            print("Generating clue...")
            clue, n_target_words = spymaster_bot.give_clue(set(blue_words), set(red_words), set(bystanders), assassin)
            print(f"Spymaster bot gives clue: {clue}, {n_target_words}")
            input("Press ENTER to continue")
        else:
            clue = input("Clue: ")
            n_target_words = input("Number of Guesses: ")
        for i in range(n_target_words+1):
            if field_operative_bot:
                print("Generating guess...")
                guess = field_operative_bot.make_guess(red_words+blue_words+bystanders+[assassin], clue)
                print(f"Field Operative bot makes guess: {guess}")
                input("Press ENTER to continue")
            else:
                guess = input(f"Guess {i}: ")
            if guess == '_pass':
                print("Skipping guess")
                break
            guessed_words.append(guess)
            if guess in red_words:
                print("You guessed the opponent team's word!")
                red_words.remove(guess)
                break
            if guess in bystanders:
                print("You guessed a bystander")
                bystanders.remove(guess)
                break
            if guess == assassin:
                print("You guessed the assassin, you lose!")
                lose = True
                break
            if guess in blue_words:
                print("Correct guess")
                blue_words.remove(guess)
            if not blue_words:
                break
    if not lose:
        print("You guessed all the blue words, you win!")
    

def generate_new_board(words):
    game_words = random.sample(words, 25)

    # TODO: Give red or blue one extra word and civs one fewer
    blue_words = game_words[:8]
    red_words = game_words[8:16]
    bystanders = game_words[16:24]
    assassin = game_words[24]

    # blue_words = ['center', 'crash', 'mail', 'poison', 'pole', 'shop', 'square', 'strike']
    # red_words = ['capital', 'eagle', 'model', 'mount', 'plate', 'spy', 'tag', 'trunk']
    # bystanders = ['apple', 'cap', 'crown', 'hollywood', 'pie', 'rome', 'ruler', 'spike']
    # assassin = 'screen'
    # game_words = blue_words + red_words + bystanders + [assassin]

    # Reorder the words on the board so you can't tell which color they are
    random.shuffle(game_words)

    return game_words, blue_words, red_words, bystanders, assassin


def print_game_state(blue_words, red_words, bystanders, assassin):
    print("=========================================================================================")
    print("BLUE WORDS: " + ', '.join(blue_words))
    print("RED WORDS: " + ', '.join(red_words))
    print("BYSTANDERS: " + ', '.join(bystanders))
    print("ASSASSIN: " + assassin)


def print_board(game_words, guessed_words):
    print()
    print('_'*76)
    for i, word in enumerate(game_words):
        if word in guessed_words:
            print(f"| {strike(word)+' '*(12-len(word))} ", end='')
        else:
            print(f"| {word:<12} ", end='')
        if i % 5 == 4:
            print("|")
            print('_'*76)
    print()


def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result


if __name__ == '__main__':
    main()

