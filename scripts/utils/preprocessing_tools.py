"""This module implements many functions for preprocessing
    raw text corpora or sentences."""

import string
import pylangacq
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

SEPARATOR = Separator(phone='$', word='@')
BACKEND = EspeakBackend(language="en-us", language_switch="remove-utterance")

def clean_utterance(utterance: str) -> str:
    """
    Use pylangacq cleaner function to clean an utterancee.

    Parameters
    ----------
    utterance: str
        The utterance to clean.
    
    Return
    ------
    - str:
        The cleaned utterance.
    """
    utterance = pylangacq.chat._clean_utterance(utterance=utterance)
    # These symbols are used for separating multiword expressions
    utterance = utterance.replace("_", " ")
    utterance = utterance.replace("+", " ")
    return utterance

def remove_ponctuations(utterance: str) -> str:
    """
    Remove ponctuations from a given utterance.

    Parameters
    ----------
    - utterance: str
        The utterance from which the punctuations will be removed.

    Returns
    -------
    str:
        The utterance without punctuations.
    """
    return utterance.translate(str.maketrans('', '', string.punctuation))

def phonemes_tokenization(utterance: str) -> str:
    """
    Tokenize a phonemized utterance in phonemes by\
    using the phonemizer's separators.

    Parameters
    ----------
    - utterance: str
        the utterance to tokenize.
    
    Return
    ------
    - str:
        The tokenized utterance
    """
    utterance = utterance.replace("$", " ")
    return utterance.replace("@", " ")

def phonemization(utterance: str) -> str:
    """
    Phonemize a given utterance.

    Parameters
    ----------
    - utterance: str
        The utterance to phonemize.
    
    Return
    ------
    The phonemized utterance.
    """
    return BACKEND.phonemize([utterance], separator=SEPARATOR, strip=True)[0]

def preprocess(utterance: str,
                phonemize: bool=True) -> str:
    """
    Preprocess a given utterance by calling multiple functions.

    Parameters
    ----------
    - phonemize: bool
        Whether phonemize the utterance or not.
    - words: bool
        Whether tokenize the utterance in words or not.
    
    Return
    ------
    - str:
        The cleaned utterance.
    """
    utterance = utterance.strip().lower()
    utterance = remove_ponctuations(utterance)
    utterance = clean_utterance(utterance)
    if phonemize :
        utterance = phonemization(utterance)
        return phonemes_tokenization(utterance)
    return utterance