from abc import ABC, abstractmethod


class Spymaster(ABC):

    @abstractmethod
    def give_clue(self, team_words, opp_words, bystanders, assassin):
        """
        Takes in the remaining unguessed words on the board, grouped by their category/color.
        Returns a tuple of the clue word followed by an integer number of guesses.
        """
        pass


class FieldOperative(ABC):

    @abstractmethod
    def make_guess(self, words, clue):
        """
        Takes in the remaining unguessed words on the board.
        Returns a guess word.
        """
        pass

