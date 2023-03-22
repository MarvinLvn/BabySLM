import argparse
from pathlib import Path
import pandas as pd
import sys
import json
from utils.phone_to_letter import letterize
from utils.probe_babyberta import load_model, prob_extractor_babyberta
from prob_utils.loaders import load_stimuli_text


# Ex:
# python scripts/extract_prob_babyberta.py --model babyberta1 --task syntactic --mode both --out results_babyberta --input_path /private/home/marvinlvn/DATA/CPC_data/test/child_zerospeech/syntactic
# python scripts/metrics/compute_syntactic.py -o results_babyberta/child_zerospeech/syntactic -g /private/home/marvinlvn/DATA/CPC_data/test/child_zerospeech -p results_babyberta/child_zerospeech/tmp -k test --is_text
def write_probabilities(seq_names, probabilities, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, 'w') as f:
        for filename, prob in zip(seq_names, probabilities):
            f.write(f'{filename} {prob}\n')
    print(f'Writing pseudo-probabilities to {out_file}')

def write_args(args, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    args = vars(args)
    args['input_path'] = str(args['input_path'])
    args['model'] = str(args['model'])
    args['out_file'] = str(args['out'])
    with open(out_file, 'w') as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    print(f'Writing args to {out_file}')

def main(argv):
    parser = argparse.ArgumentParser(description='WIP')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path of the dev or test set of the ZeroSpeech corpus '
                             '(should end with a lexical or syntactic folder)')
    parser.add_argument('--mode', type=str, choices=['dev', 'test', 'both'], default='dev',
                        help="What set to extract: dev, test or both (default: dev)")
    parser.add_argument('--model', type=str, required=True, choices=['babyberta1', 'babyberta2', 'babyberta3'],
                        help='The model to use to extract probabilities.')
    parser.add_argument('--task', type=str, required=False, choices=['syntactic', 'lexical'], default='lexical',
                        help='Extract probabilities on the lexical or syntactic task (default to lexical).')
    parser.add_argument('--out', type=str, required=True, help='Where to output the probabilities.')
    args = parser.parse_args(argv)
    args.input_path = Path(args.input_path)
    args.model = args.model.replace('babyberta', 'BabyBERTa-')
    type_task = args.input_path.stem
    type_data = args.input_path.parent.stem
    if args.mode == 'both':
        args.mode = ['dev', 'test']
    else:
        args.mode = [args.mode]

    # Load data
    stimuli = load_stimuli_text(args.input_path, args.mode)
    # Load model
    model = load_model(args.model)

    # Compute proba
    for data, data_name in zip(stimuli, args.mode):
        seq_names, probabilities = prob_extractor_babyberta(model, data)
        out_file = Path(args.out) / type_data / 'tmp' / type_task / f'{data_name}.txt'
        write_probabilities(seq_names, probabilities, out_file)
        args_file = Path(args.out) / type_data / 'tmp' / type_task / f'args_{data_name}.txt'
        write_args(args, args_file)



if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)