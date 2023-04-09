import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import mean
from random import choice, sample, shuffle
from tqdm import tqdm
from util import read_words, filter_possible_words, get_feedback


def initialize_agent(allowed, possible):
    """
    Initializes the WordleAgent that game.py uses to play Wordle.
    """
    # return RandomAgent(allowed, possible)
    return ExpectedAgent(allowed, possible)


class WordleAgent(ABC):

    def __init__(self, allowed, possible):
        self.allowed = allowed
        self.possible = possible

    @abstractmethod
    def first_guess(self):
        """Makes the first guess of a Wordle game.

        A WordleGame will call this method to get the agent's first guess of the game.
        This is an implicit signal to the agent that a new game has begun. Subsequent
        guess requests during the same game will use the .next_guess method.

        Returns
        -------
        str
            The first guess of a game of Wordle
        """

        ...

    @abstractmethod
    def next_guess(self):
        """Makes the next guess of an in-progress Wordle game.

        A WordleGame will call this method to get the agent's next guess of an
        in-progress game.

        Returns
        -------
        str
            The next guess of the agent, during an in-progress game of Wordle
        """
        ...

    @abstractmethod
    def report_feedback(self, guess, feedback):
        """Provides feedback to the agent after a guess.

        After the agent makes a guess, a WordleGame calls this method to deliver
        feedback to the agent about the guess. No return value is expected from the
        method call.

        Feedback takes the form of a list of colors, corresponding to the letters
        of the guess:
        - "green" means the guessed letter is in the target word, and in the specified position
        - "yellow" means the guessed letter is in the target word, but not in the specified position
        - "gray" means the guessed letter is not in the target word

        For instance, if the WordleGame calls:
            agent.report_feedback("HOUSE", ["gray", "green", "gray", "gray", "yellow"])
        Then the agent can infer that:
            - the target word has the letter "O" in position 2 (counting from 1)
            - the target word contains the letter "E", but not in position 5
            - the target word does not contain letters "H", "U", or "S"
        An example target word that fits this feedback is "FOYER".

        There are some important special cases when the guess contains the same letter in
        multiple positions. Suppose the letter X appears M times in the guess and N times
        in the target:
            - the K appearances of X in a correct position will be "GREEN"
            - if M <= N, then all other appearances of X will be "YELLOW"
            - if M > N, then N-K of the other appearances of X (selected arbitrarily) will
            be "YELLOW". The remaining appearances of X will be "GRAY"

        Parameters
        ----------
        guess : str
            The guess made by the agent
        feedback : list[str]
            A list of colors (expressed as strings "green", "yellow", "gray") corresponding
            to the letters in the guess
        """
        ...


class RandomAgent(WordleAgent):
    """A WordleAgent that guesses (randomly) from among words that satisfy the accumulated feedback."""
    # abt 4.07 times, 109 games/sec

    def __init__(self, allowed, possible):
        super().__init__(allowed, possible)
        self.pool = self.possible

    def first_guess(self):
        self.pool = self.possible
        shuffle(self.pool)
        return self.next_guess()

    def next_guess(self):
        shuffle(self.pool)
        return self.pool[0]

    def report_feedback(self, guess, feedback):
        self.pool = filter_possible_words(guess, feedback, self.pool)

class ExpectedAgent(WordleAgent):
    """A WordleAgent that guesses the next word word based on a given word's expected utility."""

    def __init__(self, allowed, possible):
        super().__init__(allowed, possible)
        self.pool = self.possible
        self.allowed_pool = self.allowed

    def first_guess(self):
        """Makes the first guess of a Wordle game based on the frequency score of each letter.

        This function stores the frequency of each letter's apperance and its position(index) in a dictionary and
        calculate the score of each word by summing up the score of each letter.   

        Returns
        -------
        str
            The first guess of a game of Wordle which has the highest frequency score
        """
        
        self.pool = self.possible
        score = 0
        cur_best_score = 0
        best_word = ""
        freq_dict = {}

        for word in self.pool:
            for i in range(5):
                if (i, word[i]) in freq_dict:
                    freq_dict[(i, word[i])] += 1
                else:
                    freq_dict[(i, word[i])] = 1

        for word in self.pool:
            score = 0
            for i in range(5):
                score += freq_dict.get((i, word[i]))
            if score > cur_best_score:
                cur_best_score = score
                best_word = word
        return best_word
        
    
    def next_guess(self):     
        """Makes the next guess of an in-progress Wordle game based on the expected utility of each possible word
        
        This function calculates the expected utlity (expected number of remaining words) after a guess and 
        return the word with the lowest utility.

        Returns
        -------
        str
            The next guess of the agent, during an in-progress game of Wordle
        """
        # recursive solution
        best_utility = 1000000000
        utility = 0
        best_word = ""

        for cur_guess in self.pool:
           
            utility = self.recursiveUtility(self.pool, -1, self.pool, cur_guess)
            if utility < best_utility:
                best_utility = utility
                best_word = cur_guess
        return best_word

                
    def recursiveUtility(self, pool, depth, prev_pool, guess):
        """Recursively calculates the expected utility for a given candidate word. 
        
        Inside the method, it calls itself three times; once with pool in case feedback color of the current letter is green, 
        once with pool in case feedback color of the current letter is yellow, and once in case the color is gray

        Parameters
        ----------
        pool : list[str]
            Current list of possible words at the current node of the decision tree
        depth: int
            The index of a word that we are currently at, or the current depth of the decision tree. 
        prev_pool: list[str]
            The list of possible words at the previous depth (or at the current node's parent)
        guess: str
            The candidate word whose utility we are calculating

        Returns
        -------
        float
            The expected utility of a guess
        """
        # two base cases
        if len(pool) <= 1:
            return len(pool) / len(prev_pool)
        if depth == 4:
            return len(pool)
        
        prev_pool = pool
        updated_pools = self.updatePool(pool, guess, depth + 1)

        return len(pool) / len(prev_pool) * (self.recursiveUtility(updated_pools[0], depth + 1, prev_pool, guess) + self.recursiveUtility(updated_pools[1], depth + 1, prev_pool, guess) + self.recursiveUtility(updated_pools[2], depth + 1, prev_pool, guess))


    def updatePool(self, cur_pool, guess, index):
        """ Partitions the pool of possible words based on the three possible feedbacks (green, yellow, grey) for each letter in the guess word.
            
            Parameters 
            ----------
            cur_pool : list[str]
                The current list of possible words.
            guess : str
                The candidate word.
            index : int
                The index of the letter we are currently on.

            Returns
            ----------
            list[str], list[str], list[str]
                Lists of possible words based on each feedback, in the order of "green", "yellow", and "gray". 
        """
        green_pool = []
        yellow_pool = []
        gray_pool = []

        for word in cur_pool:
            cur_letter = guess[index]
            if cur_letter == word[index]:
                green_pool.append(word)
            elif cur_letter in word:
                yellow_pool.append(word)
            else:
                gray_pool.append(word)
        return green_pool, yellow_pool, gray_pool

    def report_feedback(self, guess, feedback):
        self.pool = filter_possible_words(guess, feedback, self.pool)
        

    

    
