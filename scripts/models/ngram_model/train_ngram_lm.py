from ngram_lm import NGramLM, UnigramLM
from argparse import ArgumentParser


def main(args) -> None:
    """This function will train and save the ngram language model."""
    lm = NGramLM(pad_utterances=args.pad_utterances,
                    ngram_size=args.ngram_size,
                    smooth=args.smooth) if args.ngram_size > 1 else UnigramLM(args.smooth)
    lm.estimate(args.train_file)
    lm.save(args.out_directory, args.out_filename)

if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("--train_file",
                        type=str,
                        help="The directory containing the train files.",
                        required=True)
    parser.add_argument("--ngram_size",
                        type=int,
                        default=3,
                        help="The size of the the ngrams.",
                        required=False)
    parser.add_argument("--smooth",
                        type=float,
                        default=1e-3,
                        help="The value for smoothing the probability\
                            distribution of the language model.",
                        required=False)
    parser.add_argument('--pad_utterances',
                        help="Pad the utterances by adding fake tokens at the\
                            beginning and ending of each utterance.",
                        action='store_true')
    parser.add_argument('--no-pad_utterances',
                        dest='pad_utterances',
                        action='store_false')
    parser.add_argument("--out_directory",
                        type=str,
                        help="The directory where the model will be stored.",
                        required=True)
    parser.add_argument("--out_filename",
                        help="The filename for the model.",
                        required=True)
    main(parser.parse_args())
