import argparse
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import BertTokenizer

from phonemize import read_sentence


def write_tokenized_sentence(tokenized_sentence, out_path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as f:
        f.write(' '.join(['<BOS>']+tokenized_sentence+['<EOS>']))


def main(argv):
    parser = argparse.ArgumentParser(description='BPE encode the Providence corpus')
    parser.add_argument('--sentences', type=str, required=False,
                        help="Where the sentence files are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences')
    args = parser.parse_args(argv)
    args.sentences = Path(args.sentences)
    out = args.sentences.parent / (args.sentences.stem + '_bpe_eos_bos')
    if args.sentences.stem != 'sentences':
        raise ValueError("--annotation argument should end with a folder named sentences.")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_files = args.sentences.glob('**/*.txt')
    for sentence_file in tqdm(sentence_files):
        sentence = read_sentence(sentence_file)
        tokenized_sentence = tokenizer.tokenize(sentence)
        sub_path = sentence_file.relative_to(args.sentences)
        out_path = out / sub_path
        write_tokenized_sentence(tokenized_sentence, out_path)
    print("Done tokenizing.")

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)

# def write_tokenized_sentence(tokenized_sentence, out_path):
#     out_path.parent.mkdir(exist_ok=True, parents=True)
#     with open(out_path, 'w') as f:
#         f.write(' '.join(tokenized_sentence))
#
#
# def main(argv):
#     parser = argparse.ArgumentParser(description='BPE encode the Providence corpus')
#     args = parser.parse_args(argv)
#     args.sentences = Path("/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/training_sets")
#
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     sentence_files = list(args.sentences.glob('**/*.txt'))
#     sentence_files = [s for s in sentence_files if s.parent.stem == 'sentences']
#     for sentence_file in tqdm(sentence_files):
#         sentence = read_sentence(sentence_file)
#         tokenized_sentence = tokenizer.tokenize(sentence)
#         out_path = sentence_file.parent / 'sentences_bpe' / sentence_file.stem
#         print(sentence_file)
#         print(out_path)
#         exit()
#         write_tokenized_sentence(tokenized_sentence, out_path)
#     print("Done tokenizing.")
#
# if __name__ == "__main__":
#     # execute only if run as a script
#     args = sys.argv[1:]
#     main(args)