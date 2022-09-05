"""This module will compute accuracies on the different tasks."""

import random
import pandas as pd
from typing import Dict
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from ...utils.preprocessing_tools import preprocess
from ngram_lm import NGramLanguageModel

random.seed(1798)

def run_tasks(tasks_folder: str,
                ngram_lm: NGramLanguageModel,
                phonemize: bool) -> Dict[str, float] :
    """
    Run the tasks on ngram language model.

    Parameters
    ----------
    - tasks_folder: str
        The folder containing the task csvs
    - ngram_lm: NGramLanguageModel
        An instance of a trained language model.
    - phonemize: bool
        Whether phonemize or not the utterance
    
    Return
    ------
    - dict:
        Dictionnaty mapping tasks and their accuracy.
    """
    task_csvs = list(Path(tasks_folder).glob("*.csv"))
    total_tasks = len(task_csvs)
    task_scores = {}
    for task in tqdm(task_csvs, total=total_tasks) :
        task_name = task.stem
        total_sentences = 0.0
        good_classifications = 0.0
        task_csv = pd.read_csv(task, sep="\t")
        for real_sentence, fake_sentence in zip(task_csv.iloc[:, 0], task_csv.iloc[:, 1]):
            preprocessed_real_sentence = preprocess(real_sentence, phonemize)
            preprocessed_fake_sentence = preprocess(fake_sentence, phonemize)
            real_sentence_logprob = ngram_lm.assign_logprob(preprocessed_real_sentence)
            fake_sentence_logprob = ngram_lm.assign_logprob(preprocessed_fake_sentence)
            if not real_sentence_logprob or not fake_sentence_logprob:
                # This condition can holds if pad_utterances is set to False
                # in the ngram language model and the sentence is smaller than\
                # the ngram size.
                continue
            good_classifications += int(real_sentence_logprob > fake_sentence_logprob)
            total_sentences += 1
        task_scores[task_name] = good_classifications / total_sentences
    return task_scores

def main(args) -> None:
    """
    Run the model on the tasks and report the results\
    in a csv file
    """
    out_directory = Path("results/tasks_results")
    out_directory.mkdir(exist_ok=True, parents=True)
    ngram_lm = NGramLanguageModel()
    ngram_lm.load_model(args.ngram_model)
    result_tasks = run_tasks(args.tasks_folder,
                                ngram_lm,
                                args.phonemize,
                                args.tokenize_in_words)
    
    pd.DataFrame(result_tasks.items(), columns=['Task', 'Accuracy']).to_csv(
            out_directory / f"{args.out_filename}.csv",
            index=False
            )

if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("--tasks_folder",
                    type=str,
                    help="The folder containing the tasks",
                    required=True)
    parser.add_argument("--ngram_model",
                        type=str,
                        help="The trained ngram language model.",
                        required=True)

    parser.add_argument('--phonemize', action='store_true')
    parser.add_argument('--no-phonemize', dest='phonemize', action='store_false')
    parser.add_argument("--out_filename",
                        type=str,
                        help="The filename of the output file",
                        required=True)
    main(parser.parse_args())
