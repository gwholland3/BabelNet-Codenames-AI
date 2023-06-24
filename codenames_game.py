import random

from babelnet_bots.babelnet_bots import BabelNetSpymaster, BabelNetFieldOperative


# Text file of newline-separated words to be randomly sampled from when generating a board
WORDLIST_FILEPATH = 'wordlists/test_words.txt'


def main():
    """
    This is a primitive framework for running Codenames in the terminal, to help
    with development and testing
    """

    # Retrieve list of words to draw from
    with open(WORDLIST_FILEPATH) as f:
        words = [word for word in f.read().splitlines() if word]  # Split file contents by newline, remove empty lines

    # Generate a new, random board
    game_words, blue_words, red_words, bystanders, assassin = generate_new_board(words)

    print("Initializing bots...")
    # Set the bots (`None` means a human player)
    spymaster_bot = BabelNetSpymaster(game_words)
    field_operative_bot = None

    # This prints the identity of each word (the key that the spymaster has access to in Codenames)
    print_game_state(blue_words, red_words, bystanders, assassin)

    # Main game loop
    guessed_words = []
    done = False
    while not done:
        print_board(game_words, guessed_words)

        # Retrieve clue
        if spymaster_bot:
            print("Generating clue...")
            clue, n_target_words = spymaster_bot.give_clue(set(blue_words), set(red_words), set(bystanders), assassin)
            print(f"Spymaster bot gives clue: {clue}, {n_target_words}")
            input("Press ENTER to continue")
        else:
            clue = input("Clue: ")
            n_target_words = input("Number of Guesses: ")

        # Field operative gets 1 more guess than the number of target words specified by the spymaster
        for i in range(n_target_words+1):
            # Retrieve guess
            if field_operative_bot:
                print("Generating guess...")
                guess = field_operative_bot.make_guess(red_words+blue_words+bystanders+[assassin], clue)
                print(f"Field Operative bot makes guess: {guess}")
                input("Press ENTER to continue")
            else:
                guess = input(f"Guess {i}: ")

            # Evaluate guess
            if guess == '_pass':
                print("Skipping guess")
                break
            guessed_words.append(guess)

            if guess in red_words:
                print("You guessed the opponent team's word!")
                red_words.remove(guess)
                if not red_words:
                    print("You guessed all the red words, you lose!")
                    done = True
                break  # End turn

            if guess in bystanders:
                print("You guessed a bystander")
                bystanders.remove(guess)
                break  # End turn

            if guess == assassin:
                print("You guessed the assassin, you lose!")
                done = True
                break

            if guess in blue_words:
                print("Correct guess")
                blue_words.remove(guess)
                if not blue_words:
                    print("You guessed all the blue words, you win!")
                    done = True
                    break


def generate_new_board(words):
    game_words = random.sample(words, 25)

    # TODO: Give red or blue one extra word and bystanders one fewer
    blue_words = game_words[:8]
    red_words = game_words[8:16]
    bystanders = game_words[16:24]
    assassin = game_words[24]

    # Reorder the words on the board, so you can't tell which color they are
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
