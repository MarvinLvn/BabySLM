import argparse
import sys
from pathlib import Path
from utils.synthetizer import BaseCorporaSynthesisTask


def synthetize(input, output, credentials_path, test_mode=False):
    synthetizer = BaseCorporaSynthesisTask(no_confirmation=False)
    synthetizer.run(input, output, credentials_path, test_mode)


def main(argv):
    parser = argparse.ArgumentParser(description='Synthetize Providence.')
    parser.add_argument('--input', type=str, help='Path where to find the sentences that need to be synthetized',
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences')
    parser.add_argument('--out', type=str, help='Path where to store the synthetized audio',
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/audio_synthetized')
    parser.add_argument('--test', action='store_true',
                        help='if True, will generate only a few stimuli')
    parser.add_argument('--credentials_path', type=str, required=True,
                        help='Path to your Google TTS credentials')
    args = parser.parse_args(argv)
    args.input = Path(args.input)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    synthetize(args.input, args.out, args.credentials_path, args.test)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)