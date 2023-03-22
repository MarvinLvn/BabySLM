import argparse
import glob
import json
import os
import sys

import numpy as np


def main(argv):
    parser = argparse.ArgumentParser(description='This script takes as input the directory to the trained model '
                                                 'and returns the best epoch, selected on the validation accuracy.'
                                                 'In more details, it will average the loss obtained for each time '
                                                 'step (distance to the prediction) and will round the best '
                                                 'epoch to the nearest multiple of 5.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory containing the .json log files')
    parser.add_argument('--min', type=int, required=False, default=None,
                        help='Will select models whose epoch is more than --min')
    parser.add_argument('--max', type=int, required=False, default=None,
                        help='Will select models whose epoch is less than --max')
    parser.add_argument('--output-id', action='store_true',
                        help='output only the id of the best epoch')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.model_path):
        raise ValueError("Can't find %s" % args.model_path)

    checkpoint_logs = os.path.join(args.model_path, "checkpoint_logs.json")
    if not os.path.isfile(checkpoint_logs):
        raise ValueError("Can't find %s. Make sure you did train the model." % checkpoint_logs)

    with open(checkpoint_logs, 'rb') as fin:
        logs = json.load(fin)

    # Get checkpoint indices
    cp_idxs = glob.glob(os.path.join(args.model_path, "checkpoint*.pt"))
    cp_idxs = sorted([int(os.path.basename(e).replace('checkpoint_', '').replace('.pt', '')) for e in cp_idxs])

    if args.min is not None:
        cp_idxs = [e for e in cp_idxs if e >= args.min]

    if args.max is not None:
        cp_idxs = [e for e in cp_idxs if e <= args.max]

    if len(cp_idxs) == 0:
        raise ValueError(
            "Either no checkpoint between --min and --max can be found, either the model hasn't been trained. Please check.")
    acc_val = np.asarray(logs['locAcc_val'])
    acc_val = np.mean(acc_val, axis=1)
    cp_idxs = [e for e in cp_idxs if e <= len(acc_val)]
    acc_val = acc_val[cp_idxs]
    opt_idx = np.argmax(acc_val)
    opt_epoch = cp_idxs[opt_idx]

    if args.output_id:
        print(opt_epoch)
    else:
        print("Best epoch with an average accuracy of %f on the validation set is : %d" % (acc_val[opt_idx], opt_epoch))


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)