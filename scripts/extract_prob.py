import argparse
from pathlib import Path
import pandas as pd
import sys
from utils.phone_to_letter import letterize
from utils.probe_babyberta import load_model, babyberta_probing


def load_text_lexical(test_path):
    data = pd.read_csv(test_path, sep='\t')
    data = data[['word', 'fake_word_pho']]
    data['fake_word'] = data['fake_word_pho'].apply(lambda word: letterize(word))
    data.columns = ['real', 'fake_pho', 'fake']
    return data


def load_text_syntactic(test_path):
    data = pd.read_csv(test_path, sep='\t', header=None)
    data.columns = ['real', 'fake']
    return data


def save_results(results, acc_score, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    score = pd.DataFrame({'accuracy': [acc_score]})
    score.to_csv(str(out_path / 'accuracy.csv'), index=False)
    results.to_csv(str(out_path / 'results_pairs.csv'), index=False)


def main(argv):
    parser = argparse.ArgumentParser(description='WIP')
    parser.add_argument('--model', type=str, required=True, choices=['babyberta1', 'babyberta2', 'babyberta3'],
                        help='The model to use to extract probabilities.')
    parser.add_argument('--task', type=str, required=False, choices=['syntactic', 'lexical'], default='lexical',
                        help='Extract probabilities on the lexical or syntactic task (default to lexical).')
    parser.add_argument('--subtask', type=str, required=False, choices=['pos_order', 'anaphor_gender_agreement',
                                                                        'anaphor_number_agreement',
                                                                        'determiner_noun_agreement',
                                                                        'noun_verb_agreement'], default='pos_order',
                        help='Extract probabilities on the lexical or syntactic task (default to lexical).')
    parser.add_argument('--out', type=str, required=False, default=None, help='Where to output the probabilities.')
    args = parser.parse_args(argv)
    args.model = args.model.replace('babyberta', 'BabyBERTa-')

    if args.task == 'lexical':
        test_path = Path('/private/home/marvinlvn/ChildDirectedLexicalTest/child_workspace/corpora/wuggy_pairs/corpus_1.csv')
    elif args.task == 'syntactic':
        test_path = Path('/private/home/marvinlvn/ChildDirectedSyntacticTest/data/tasks') / (args.subtask + '.csv')

    if args.out is None:
        root = Path(__file__).absolute().parent.parent
        args.out = root / 'results' / args.model / args.task / test_path.stem

    mode = 'text'
    if mode == 'text':
        load_stimuli = load_text_syntactic if args.task == 'syntactic' else load_text_lexical
        data = load_stimuli(test_path)
        model = load_model(args.model)
        results, acc_score = babyberta_probing(model, data)
        save_results(results, acc_score, args.out)
    elif mode == 'speech':
        raise ValueError("Not implemented yet.")


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)