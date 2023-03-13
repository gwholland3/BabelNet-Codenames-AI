import random
import argparse

from babelnet_bots import BabelNetSpymaster


WORDLIST_FILEPATH = 'data/wordlists/test_words.txt'


def main(args):
    # Retrieve all Codenames words
    with open(WORDLIST_FILEPATH) as f:
        words = f.read().splitlines()

    game_words, blue_words, red_words, bystanders, assassin = generate_new_board(words)

    spymaster_bot = BabelNetSpymaster(game_words)
    field_operative_bot = None

    print_game_state(blue_words, red_words, bystanders, assassin)

    guessed_words = []
    lose = False
    while blue_words and not lose:
        print_board(game_words, guessed_words)
        if spymaster_bot:
            clue, n_target_words = spymaster_bot.give_clue(set(blue_words), set(red_words), set(bystanders), assassin)
            print(f"Spymaster bot gives clue: {clue} {n_target_words}")
            input("Press ENTER to continue")
        else:
            clue = input("Clue: ")
            n_target_words = input("Number of Guesses: ")
        for i in range(n_target_words+1):
            if field_operative_bot:
                guess = field_operative_bot.make_guess(red_words+blue_words+bystanders+[assassin], clue)
                print(f"Field Operative bot makes guess: {guess}")
                input("Press ENTER to continue")
            else:
                guess = input(f"Guess {i}: ")
            if guess == '_pass':
                break
            guessed_words.append(guess)
            if guess in red_words:
                red_words.remove(guess)
                break
            if guess in bystanders:
                bystanders.remove(guess)
                break
            if guess == assassin:
                print("You guessed the assassin, you lose!")
                lose = True
                break
            if guess in blue_words:
                blue_words.remove(guess)
    if not lose:
        print("You guessed all the blue words, you win!")
    
    # Draw graphs for all words
    # for word in game_words:
    #     spymaster_bot.draw_graph(spymaster_bot.graphs[word], word+"_all", get_labels=True)


def generate_new_board(words):
    game_words = random.sample(words, 25)

    # TODO: Give red or blue one extra word and civs one fewer
    red_words = game_words[:8]
    blue_words = game_words[8:16]
    bystanders = game_words[16:24]
    assassin = game_words[24]

    # Reorder the words on the board so you can't tell which color they are
    random.shuffle(game_words)

    return game_words, red_words, blue_words, bystanders, assassin


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
    # TODO: Remove unnecessary args
    parser = argparse.ArgumentParser(description='Play a game of codenames.')
    parser.add_argument('--verbose', action='store_true',
                        help='print out verbose information'),
    parser.add_argument('--visualize', action='store_true',
                        help='visualize the choice of clues with graphs')
    parser.add_argument('--split-multi-word', default=True)
    parser.add_argument('--disable-verb-split', default=True)
    parser.add_argument('--length-exp-scaling', type=int, default=None,
                        help='Rescale lengths using exponent')
    parser.add_argument('--no-heuristics', action='store_true',
                        help='Remove heuristics such as IDF and dict2vec')
    parser.add_argument('--kim-scoring-function', dest='use_kim_scoring_function', action='store_true',
                        help='use the kim 2019 et. al. scoring function'),
    args = parser.parse_args()

    main(args)

