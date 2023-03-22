import argparse
import os
import random
import sys
from pathlib import Path

random.seed(10)
MAX_TOKENS = 4000


def sort_files_spkr_onset(files):
    def get_key(x):
        stem = x.stem
        return '_'.join(stem.split('_')[:6]), float(stem.split('_')[-2])
    return sorted(files, key=lambda x: get_key(x))


def read_folder(input_path, ext='.txt', no_tab_replace=False):
    data = []
    txt_files = input_path.glob('**/*%s' % ext)
    txt_files = [txt_file for txt_file in txt_files if not txt_file.stem.startswith('fairseq')]
    txt_files = sort_files_spkr_onset(txt_files)

    for txt_file in txt_files:
        with open(txt_file, 'r') as fin:
            line = fin.read()
            if not no_tab_replace:
                # replace word separator by special token <SEP>
                line = line.replace('\t', ' <SEP> ').replace('xxx', '<UNK>')
            splitted = line[:-1].split()
            if len(splitted) >= MAX_TOKENS:
                # Split list into fixed sized chunks
                line = [' '.join(splitted[i:i + MAX_TOKENS]) + '\n' for i in range(0, len(splitted), MAX_TOKENS)]
                data.extend(line)
            else:
                data.append(line)
    return data


def read_file(input_path):
    data = []
    with open(input_path, 'r') as fin:
        for line in fin.readlines():
            splitted = line[:-1].split(' ')
            if len(splitted) >= MAX_TOKENS:
                # Split list into fixed sized chunks
                line = [' '.join(splitted[i:i + MAX_TOKENS]) + '\n' for i in range(0, len(splitted), MAX_TOKENS)]
                data.extend(line)
            else:
                data.append(line)
    return data


def main(argv):
    # Args parser
    parser = argparse.ArgumentParser(description='Given a training set, split it across 3 files containing'
                                                 'the train, dev, and test set.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input training set.')
    parser.add_argument('--val_prop', type=float, default=0.1,
                        help='Proportion of the validation set.')
    parser.add_argument('--test_prop', type=float, default=0,
                        help='Proportion of the test set.')
    parser.add_argument('--prefix', type=str, default='fairseq',
                        help='Prefix to add to the basename of created files.')
    parser.add_argument('--ext', type=str, default='.txt',
                        help='File extension')
    parser.add_argument('--no_tab_replace', action='store_true',
                        help='If activated, will not replace tabulation by <SEP>.')
    args = parser.parse_args(argv)
    args.input_path = Path(args.input_path)

    # Not convenient for you user, but this way we ensure you don't mess up something!
    if args.input_path.stem not in ['phones', 'phones_with_space', 'sentences', 'sentences_bpe',
                                    'sentences_bpe_eos_bos'] and args.input_path.suffix != '.txt':
        raise ValueError("Argument --input_path should point to a folder named "
                         "[phones, phones_with_space, sentences, sentences_bpe] "
                         "OR it should point to a .txt file (for quantized units).\n"
                         "See Documentation about the training set preparation.")

    is_file = args.input_path.suffix == '.txt'

    # Read data
    if not is_file:
        data = read_folder(args.input_path, args.ext, args.no_tab_replace)
        output_folder = args.input_path
    else:
        data = read_file(args.input_path)
        output_folder = args.input_path.parent

    # Split data
    size_train = int((1-args.val_prop - args.test_prop) * len(data))
    size_val = int(args.val_prop * len(data))
    data_train, data_val, data_test = data[:size_train], data[size_train:size_train+size_val], data[size_train+size_val:]
    data = {'train': data_train,
            'val': data_val,
            'test': data_test}

    for key, data in data.items():
        output_file = output_folder / f'{args.prefix}_{key}.txt'
        if len(data) != 0:
            with open(output_file, 'w') as fin:
                for line in data:
                    fin.write(line)
    print("Done splitting input file into train/dev/test sets.")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

