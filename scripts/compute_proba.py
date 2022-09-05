from pathlib import Path
from os.path import exists, join, basename, dirname, abspath
import sys
import argparse
from prob_utils.loaders import load_stimuli_text
from prob_utils.probability_extractors import TextLstmProbExtractor
import json


def write_args(args, example_input, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    args = vars(args)
    args['example_input'] = example_input
    args['input_path'] = str(args['input_path'])
    args['model_path'] = str(args['model_path'])
    with open(out_file, 'w') as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    print(f'Writing args to {out_file}')


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Compute pseudo log-probabilities.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path of the dev or test set of the ZeroSpeech corpus '
                             '(should end with a lexical or syntactic folder)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained language model.')
    parser.add_argument('--dict_path', type=str, help='The path to the folder where to find dict.txt used to train the '
                                                      'language model (if not specified, will look for it in '
                                                      'data-bin/ in the model directory)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='The number of sentences to be in each batch (default: 128)')
    parser.add_argument('--pooling', type=str, default='mean',
                        help="Type of pooling done on the features to calculate "
                             "the pseudo log-proba. 'sum' or 'mean'.")
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
    parser.add_argument('--bpe_encode', action='store_true',
                        help="If activated, will BPE encode input.")
    parser.add_argument('--bos_eos', action='store_true',
                        help="If activated, will add <BOS> and <EOS> TOKENS.")
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)
    args.input_path = Path(args.input_path)
    args.model_path = Path(args.model_path)
    if args.model_path.suffix != '.pt':
        raise ValueError('--model_path should point to a checkpoint file (.pt)')
    if args.phonemize:
        if args.input_path == 'lexical':
            raise ValueError('--phonemize flag can''t be activated when evaluating on lexicon.')
    out_path = args.input_path.stem
    if args.dict_path is not None:
        args.dict_path = Path(args.dict_path)
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
    prob_extractor = TextLstmProbExtractor(model_path=args.model_path, dict_path=args.dict_path,
                                           out_path=out_path, batch_size=args.batch_size, bpe_encode=args.bpe_encode,
                                           bos_eos=args.bos_eos, pooling=args.pooling,
                                           remove_word_spaces=args.remove_word_spaces)
    for data, data_name in zip(stimuli, args.mode):
        seq_names, probabilities = prob_extractor.extract_all(data)
        out_file = args.model_path.parent / type_data / 'tmp' / type_task / f'{data_name}.txt'
        prob_extractor.write_probabilities(seq_names, probabilities, out_file)
        args_file = args.model_path.parent / type_data / 'tmp' / type_task / f'args_{data_name}.txt'
        write_args(args, prob_extractor.get_example_input, args_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)