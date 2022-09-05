"""This module implements a ngram language model."""

# import standard python modules
import random
from typing import Union, List, Tuple, Iterator
import pickle
import logging
from pathlib import Path
from itertools import tee
from collections import defaultdict

# import download packages
import numpy as np

random.seed(1798)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

Ngram = Tuple[str]

class NGramLanguageModel:
    """
    This class implements a ngram language model with
    add-delta smoothing.

    Parameters
    ----------
    - pad_utterance: bool
        Whether or not pad utterances by adding fake tokens\
        in the beginning and ending of each utterance.
    - ngram_size: int
        The size of the ngrams.
    - smooth: flot, int
        The value for smoothing. Default=1e-3
    """
    def __init__(self,
                    pad_utterances: bool=True,
                    ngram_size: int=3,
                    smooth: Union[float, int]=1e-3):
        self.pad_utterances = pad_utterances
        self.ngram_size = ngram_size
        self.smooth = smooth
        self.ngram_counter = defaultdict(lambda: defaultdict(int))
        self.denominator_smoother = None

    def get_ngrams(self, splitted_utterance: List[str]) -> Iterator[Ngram] :
        """
        Return the ngrams of a given utterance.

        Parameters
        ----------
        - splitted_utterance: list of strings
            The utterance from which to extract the ngrams.

        Return
        -------
        - list:
            List of ngrams extracted from the utterance.
        """
        if self.pad_utterances :
            # add '<' for start token padding and '>' for\
            # end token padding
            splitted_utterance = (["<"] * (self.ngram_size - 1)) \
                + splitted_utterance + ([">"] * (self.ngram_size - 1))
        iterables = tee(splitted_utterance, self.ngram_size)
        for number_of_shifts, iterator in enumerate(iterables) :
            for _ in range(number_of_shifts) :
                next(iterator, None)
        # If pad_utterance is false, this returned iterable will be empty\
        # if the length of the utterance is smaller than the ngram size.
        return zip(*iterables)

    def estimate(self, train_file: str) -> None:
        """
        Estimate the language model from a raw text file.

        Parameters
        ----------
        - train_file: str
            The path of the file containing train sentences\
            with one sentence per line.
        """
        LOGGER.info("Training the model...")
        with open(train_file, mode="r", encoding="utf-8") as sentences_file:
            vocabulary = set()
            for utterance in sentences_file :
                utterance = utterance.strip().split(" ")
                for ngram in self.get_ngrams(utterance):
                    *context_tokens, next_token = ngram
                    self.ngram_counter[tuple(context_tokens)][next_token] += 1
                    vocabulary.add(next_token)
            # will be used to smooth the probability distribution
            # by adding the 'smooth' value to each token in the vocabulary
            self.denominator_smoother = len(vocabulary) * self.smooth
        LOGGER.info("Model trained.")

    def save_model(self, out_dirname: str, out_filename: str) -> None:
        """
        Save the estimated parameters and the hyperparameters of\
        the language model in a JSON file.

        Parameters
        ----------
        - out_dirname: str
            The directory where the model will be stored.
        - out_filename: str
            The filename of the model.
        """
        LOGGER.info(f"Saving the model to {out_dirname} as {out_filename}...")
        # back to standard python dictionnary
        self.ngram_counter = {ngram : dict(counts) for ngram, counts \
                                in self.ngram_counter.items()}
        out_directory = Path(out_dirname)
        out_directory.mkdir(parents=True, exist_ok=True)
        out_file = out_directory / f"{out_filename}.pkl"
        with open(out_file, "wb") as model_to_save:
            pickle.dump(self, model_to_save)
        LOGGER.info("Modle saved.")
    
    def load_model(self, path: str) -> None:
        """
        Load a stored language model in a pickle file.

        Parameters
        ----------
        - path: str
            Path to where the language model is stored in a pickle file.
        """
        LOGGER.info(f"Loading the model from {path}...")
        with open(path, "rb") as model_to_load:
            loaded_model = pickle.load(model_to_load)
            self.pad_utterances = loaded_model.pad_utterances
            self.ngram_size = loaded_model.ngram_size
            self.smooth = loaded_model.smooth
            self.ngram_counter = loaded_model.ngram_counter
            self.denominator_smoother = loaded_model.denominator_smoother
            del loaded_model
        LOGGER.info("Modele loaded.")

    def ngram_probability(self, ngram: Ngram) -> float:
        """
        Assign a probability of a given ngram by using\
        the estimated counts of the ngram language model.

        Paramerers
        ----------
        - ngram: Tuple of str
            The ngram for which you want to assign a probability.

        Return
        ------
        - float:
            The assigned probability to the given ngram.
        """
        *left_context, next_token = ngram
        left_context = tuple(left_context)
        left_context_seen = self.ngram_counter.get(left_context, False)
        if not left_context_seen:
            # unknown left_context, return smoothed probability, that is a
            # very small probability instead of returning 0 probability
            return self.smooth / self.denominator_smoother
        denominator = sum(left_context_seen.values()) + self.denominator_smoother
        # add also the smooth to the numerator, so all sums up to one.
        numerator = self.ngram_counter[left_context].get(next_token, 0.0) + self.smooth
        return numerator / denominator

    def assign_logprob(self, utterance: str) -> float:
        """
        This function will assign a normalised log proabability
        of a given utterance.

        Parameters
        ----------
        - utterance: str
            The utterance for which to compute

        Return
        ------
        - flot:
            The log probability of the utterance.
        """
        ngrams_of_the_utterance = list(self.get_ngrams(utterance.split(" ")))
        if not ngrams_of_the_utterance:
            # This condition can holds only in the case pad_utterances\
            # is set to False.
            return False
        ngram_values = np.array([self.ngram_probability(ngram)
                                    for ngram in ngrams_of_the_utterance])
        return np.sum(np.log(ngram_values)) / len(ngrams_of_the_utterance)