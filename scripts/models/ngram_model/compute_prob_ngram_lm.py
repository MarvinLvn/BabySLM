# """This module will compute accuracies on the different tasks."""

# import random
# import pandas as pd
# from typing import Dict
# from pathlib import Path
# from argparse import ArgumentParser
# from tqdm import tqdm

# from ...utils.preprocessing_tools import preprocess
# from ngram_lm import NGramLanguageModel

# random.seed(1798)

# def run_tasks(tasks_folder: str,
#                 ngram_lm: NGramLanguageModel,
#                 phonemize: bool) -> Dict[str, float] :
#     """
#     Run the tasks on ngram language model.

#     Parameters
#     ----------
#     - tasks_folder: str
#         The folder containing the task csvs
#     - ngram_lm: NGramLanguageModel
#         An instance of a trained language model.
#     - phonemize: bool
#         Whether phonemize or not the utterance
    
#     Return
#     ------
#     - dict:
#         Dictionnaty mapping tasks and their accuracy.
#     """
#     task_csvs = list(Path(tasks_folder).glob("*.csv"))
#     total_tasks = len(task_csvs)
#     task_scores = {}
#     for task in tqdm(task_csvs, total=total_tasks) :
#         task_name = task.stem
#         total_sentences = 0.0
#         good_classifications = 0.0
#         task_csv = pd.read_csv(task, sep="\t")
#         for real_sentence, fake_sentence in zip(task_csv.iloc[:, 0], task_csv.iloc[:, 1]):
#             preprocessed_real_sentence = preprocess(real_sentence, phonemize)
#             preprocessed_fake_sentence = preprocess(fake_sentence, phonemize)
#             real_sentence_logprob = ngram_lm.assign_logprob(preprocessed_real_sentence)
#             fake_sentence_logprob = ngram_lm.assign_logprob(preprocessed_fake_sentence)
#             if not real_sentence_logprob or not fake_sentence_logprob:
#                 # This condition can holds if pad_utterances is set to False
#                 # in the ngram language model and the sentence is smaller than\
#                 # the ngram size.
#                 continue
#             good_classifications += int(real_sentence_logprob > fake_sentence_logprob)
#             total_sentences += 1
#         task_scores[task_name] = good_classifications / total_sentences
#     return task_scores

# def main(args) -> None:
#     """
#     Run the model on the tasks and report the results\
#     in a csv file
#     """
#     out_directory = Path("results/tasks_results")
#     out_directory.mkdir(exist_ok=True, parents=True)
#     ngram_lm = NGramLanguageModel()
#     ngram_lm.load_model(args.ngram_model)
#     result_tasks = run_tasks(args.tasks_folder,
#                                 ngram_lm,
#                                 args.phonemize,
#                                 args.tokenize_in_words)
    
#     pd.DataFrame(result_tasks.items(), columns=['Task', 'Accuracy']).to_csv(
#             out_directory / f"{args.out_filename}.csv",
#             index=False
#             )

# if __name__ == "__main__" :
#     parser = ArgumentParser()
#     parser.add_argument("--tasks_folder",
#                     type=str,
#                     help="The folder containing the tasks",
#                     required=True)
#     parser.add_argument("--ngram_model",
#                         type=str,
#                         help="The trained ngram language model.",
#                         required=True)

#     parser.add_argument('--phonemize', action='store_true')
#     parser.add_argument('--no-phonemize', dest='phonemize', action='store_false')
#     parser.add_argument("--out_filename",
#                         type=str,
#                         help="The filename of the output file",
#                         required=True)
#     main(parser.parse_args())
# import sys
# sys.path.append('.')
# from scripts.prob_utils.loaders import load_stimuli_text
# from pathlib import Path

# path = Path("/scratch2/ysy/BenchmarkLangAcq/data/model_evaluation/")
# loaded = load_stimuli_text(path, ["test"])
# print(loaded)

import sys
sys.path.append('.')
from scripts.prob_utils.loaders import load_stimuli_text
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable, Tuple
from time import time
from typing import Iterable
from ngram_lm import UnigramLM, NGramLM

class TextNgramProbExtractor:
    """TODO"""

    def __init__(self, model_path, model_type, remove_word_spaces=True):
        self.remove_word_spaces = remove_word_spaces
        self.models = {
            "unigram" : UnigramLM,
            "ngram" : NGramLM
        }
        self.load(model_path, model_type)

    
    def load(self, model_path, model_type) -> None:
        """TODO"""
        if model_type not in self.models :
            raise ValueError("Model type have to be among ['unigram', 'ngram']")
        self.model = self.models[model_type]()
        self.model.load(model_path)
        self.loaded = True
    
    def preprocessing(self, example) -> str:
        """TODO"""
        if self.remove_word_spaces:
            return example.replace(' <SEP> ', ' ')

    def extract_all(self, data) -> Tuple[Iterable, Iterable]:
        start_time = time()
        seq_names = data['filename']
        transcriptions = data['transcription']
        probabilities = []
        for transcription in transcriptions:
            preprocessed = self.preprocessing(transcription)
            logprob = self.model.assign_logprob(preprocessed)
            probabilities.append(logprob)
        print(f"Done computing probabilities in %.2f s." % (time() - start_time))
        return seq_names, probabilities
    
    def write_probabilities(self, seq_names, probabilities, out_file):
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, 'w') as f:
            for filename, prob in zip(seq_names, probabilities):
                f.write(f'{filename} {prob}\n')
        print(f'Writing pseudo-probabilities to {out_file}')

def parseArgs():
    parser = ArgumentParser()  
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path of the dev or test set of the ZeroSpeech corpus '
                             '(should end with a lexical or syntactic folder)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained language model.')
    parser.add_argument('--model_type', type=str, choices=['unigram', 'ngram'], default='unigram',
                        help="The type of model, unigram or ngram (ngram size > 1)")
    parser.add_argument('--debug', action='store_true',
                        help="If True, will only process 10 sequences.")
    parser.add_argument('--mode', type=str, choices=['dev', 'test', 'both'], default='dev',
                        help="What set to extract: dev, test or both (default: dev)")
    parser.add_argument('--text', action='store_true',
                        help="If activated, will load text instead of audio (for text-based language models).")
    parser.add_argument('--phonemize', action='store_true',
                        help="If activated, will phonemize the input (will store cache "
                             "file containing phonemized input in the data folder).")
    parser.add_argument('--remove_word_spaces', action='store_true',
                        help="If activated will remove word spaces (only useful for syntactic evaluation)."
                             "Should be activated for models that haven't received spaces during training.")
    return parser.parse_args()
def main():
    args = parseArgs()
    args.input_path = Path(args.input_path)
    args.model_path = Path(args.model_path)
    if args.phonemize:
        if args.input_path == 'lexical':
            raise ValueError('--phonemize flag can''t be activated when evaluating on lexicon.')
    out_path = args.input_path.stem
    if args.mode == 'both':
        args.mode = ['dev', 'test']
    else:
        args.mode = [args.mode]
    type_task = args.input_path.stem
    type_data = args.input_path.parent.stem

    # Load stimuli (text, or audio)
    if args.text:
        stimuli = load_stimuli_text(args.input_path, args.mode, args.debug, args.phonemize)
    else:
        raise ValueError("Not implemented yet.")

    # Extract pseudo-prob
    prob_extractor = TextNgramProbExtractor(model_path=args.model_path,
                                            model_type=args.model_type,
                                            remove_word_spaces=args.remove_word_spaces)

    for data, data_name in zip(stimuli, args.mode):
        seq_names, probabilities = prob_extractor.extract_all(data)
        out_file = args.model_path.parent / type_data / 'tmp' / type_task / f'{data_name}.txt'
        prob_extractor.write_probabilities(seq_names, probabilities, out_file)

if __name__ == "__main__":
    main()