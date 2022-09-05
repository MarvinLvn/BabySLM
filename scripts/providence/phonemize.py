import argparse
import os
import sys
from pathlib import Path

from paraphone.tasks.phonemize import CMUEnglishPhonemizer, CelexPhonemizer, PhonemizerWrapper
from paraphone.workspace import Workspace
from tqdm import tqdm


def load_phonemizers(workspace):
    return [CMUEnglishPhonemizer(workspace),
            CelexPhonemizer(workspace),
            PhonemizerWrapper(workspace, lang="en-us")]


def phonemize_sentence(sentence, sentence_path, phonemizers):
    # phonemizer doesn't like end of sentences
    # remove punctuations
    old_sentence = sentence
    sentence = sentence.replace('.', '').replace('^', '').replace('!', '') \
        .replace('?', '').replace('"', '').replace('„', '')
    # remove double white spaces
    sentence = ' '.join(sentence.split())
    # remove leading zeros of words
    sentence = ' '.join([w[1:] if w[0] == '0' else w for w in sentence.split(' ')])
    # remove carriage return
    sentence = sentence.strip()
    # print(sentence, end=' ')
    phonemized_sentence = []

    # loop through words
    for word in sentence.split(' '):

        # trying to phonemize with each phonemizer, in their order
        for phonemizer in phonemizers:
            try:
                phonemized_word = phonemizer.phonemize(word)
            except KeyError:
                continue
            except ValueError as err:
                raise ValueError(f"Error in phonemize/fold for word {repr(word)} from file {sentence_path}: {err}")
            else:
                phonemized_sentence += ['\t'] + phonemized_word
                break
        else:
            raise ValueError(f"Couldn't phonemize word {repr(sentence)} from file {sentence_path}")

    # Remove leading tabulation
    phonemized_sentence = phonemized_sentence[1:]
    # print("->", repr(' '.join(phonemized_sentence)))
    return phonemized_sentence, sentence, old_sentence


def read_sentence(sentence_path):
    with open(sentence_path, 'r') as f:
        sentence = f.read()
    return sentence


def read_and_phonemize_sentence(sentence_path, phonemizers):
    sentence = read_sentence(sentence_path)
    return phonemize_sentence(sentence, sentence_path, phonemizers)


def write_phonemized(phones, phonemes_file):
    phonemes_file.parent.mkdir(exist_ok=True, parents=True)
    with open(phonemes_file, 'w') as f:
        f.write(' '.join(phones).replace(' \t ', '\t')+'\n')


def fix_sentence(corrected_sentence, sentence_file):
    sentence_file.parent.mkdir(exist_ok=True, parents=True)
    with open(sentence_file, 'w') as f:
        f.write(corrected_sentence+'\n')


def main(argv):
    parser = argparse.ArgumentParser(description='Phonemize utterances using pronunciation dictionaries '
                                                 '(or phonemizer if word does not exist).')
    parser.add_argument('--sentences', type=str, required=False, help="Where the .txt files "
                                                                      "that need to be phonemized are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences')
    parser.add_argument('--out', type=str, required=False, help="Where the dataset will be stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/phonemes')
    parser.add_argument('--no_clean', action='store_true', help="If activated, will not clean the sentences")
    args = parser.parse_args(argv)
    args.sentences = Path(args.sentences)
    args.out = Path(args.out)

    workspace = Workspace(Path(os.path.dirname(os.path.realpath(__file__))).parent)
    phonemizers = load_phonemizers(workspace)
    sentence_files = args.sentences.glob('**/*.txt')
    for i, sentence_file in tqdm(enumerate(sentence_files)):
        phones, corrected_sentence, old_sentence = read_and_phonemize_sentence(sentence_file, phonemizers)

        # write phonemized sentences with space
        subpath = sentence_file.relative_to(args.sentences)
        phonemes_file = args.out.parent / (args.out.stem + '_with_space') / subpath
        write_phonemized(phones, phonemes_file)

        # write phonemized sentences with no space
        subpath = sentence_file.relative_to(args.sentences)
        phonemes_file = args.out / subpath
        phones = [p for p in phones if p != '\t']
        write_phonemized(phones, phonemes_file)

        # correct sentence
        if not args.no_clean and corrected_sentence != old_sentence.strip():
            fix_sentence(corrected_sentence, sentence_file)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)