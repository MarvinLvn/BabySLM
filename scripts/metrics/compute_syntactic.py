"""Adapted from ZeroSpeech 2021"""

import argparse
import pathlib
import sys

import pandas


def load_data(gold_file, submission_file, is_text=False):
    """Returns the data required for evaluation as a pandas data frame

    Each line of the returned data frame contains a pair of (correct,
    incorrect) sentences and has the following columns: 'id', 'voice', 'type',
    'sentence', 'score sentence', 'non sentence', 'score non sentence'.

    Parameters
    ----------
    gold_file : path
        The gold file for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    data : pandas.DataFrame
        The data ready for evaluation

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    # ensures the two input files are here
    for input_file in (gold_file, submission_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f'file not found: {input_file}')

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(
        gold_file, header=0, index_col='filename')

    if is_text:
        voices = gold['voice'].unique()
        gold = gold[gold['voice'] == voices[0]]

    score = pandas.read_csv(
        submission_file, sep=' ', header=None,
        names=['filename', 'score'], usecols=[0, 1], index_col='filename')

    # ensures the filenames in gold and submission are the same
    if set(gold.index) != set(score.index):
        has_less_files = set(gold.index) - set(score.index)
        has_more_files = set(score.index) - set(gold.index)
        print("MismatchError:", file=sys.stderr)
        if len(has_more_files) > 0:
            print('submission has extra files', file=sys.stderr)
            print(f'extra files: {has_more_files}', file=sys.stderr)

        if len(has_less_files) > 0:
            print('submission is missing files', file=sys.stderr)
            print(f'missing files: {has_less_files}:', file=sys.stderr)

        sys.exit(1)

    if is_text:
        voices = gold['voice'].unique()
        gold = gold[gold['voice'] == voices[0]]
        # same number of lines is expected
        data = pandas.concat([gold, score], axis=1)
    else:
        # merge by filename
        data = pandas.merge(gold, score, on='filename')

    data.reset_index(drop=True, inplace=True)

    if len(data) != len(gold):
        raise ValueError("len(data) should be equal to len(gold). Aborting.")

    # going from a sentence per line to a pair (grammatical sentence, ungrammatical sentence) per line
    data = pandas.concat([
        data.loc[data['correct'] == 1].reset_index().rename(
            lambda x: 's_' + x, axis=1),
        data.loc[data['correct'] == 0].reset_index().rename(
            lambda x: 'ns_' + x, axis=1)], axis=1)

    data.drop(
        ['s_index', 'ns_index', 'ns_voice', 'ns_type', 'ns_subtype',
         's_correct', 'ns_correct', 'ns_id'],
        axis=1, inplace=True)

    data.rename(
        {'s_id': 'id',
         's_voice': 'voice',
         's_type': 'type',
         's_subtype': 'subtype',
         's_transcription': 'sentence',
         'ns_transcription': 'non sentence',
         's_score': 'score sentence',
         'ns_score': 'score non sentence'},
        axis=1, inplace=True)


    if data[['score sentence', 'score non sentence']].isna().sum().sum():
        print(data[data['score sentence'].isna()])
        print(data[data['score non sentence'].isna()])
        raise ValueError("Found some NaN in the predicted scores. Aborting.")
    return data


def evaluate_all(data):
    score = data.loc[:, ['score sentence', 'score non sentence']].to_numpy()
    data['score'] = (
            0.5 * (score[:, 0] == score[:, 1])
            + (score[:, 0] > score[:, 1]))
    return data.copy()


def evaluate_by_pair(data):
    """Returns a data frame with the computed scores by (grammatical sentence, ungrammatical sentence) pair

    Parameters
    ----------
    data : pandas.DataFrame
        The result of `load_data`

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (sentence, non sentence) pairs, the data frame has the
        columns: 'sentence', 'non sentence' 'type' and 'score'.

    """
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score sentence' and 'score non sentence' columns that become useless
    score = data.loc[:, ['score sentence', 'score non sentence']].to_numpy()
    data['score'] = (
            0.5 * (score[:, 0] == score[:, 1])
            + (score[:, 0] > score[:, 1]))
    data.drop(columns=['score sentence', 'score non sentence'], inplace=True)
    score = data.groupby(['type', 'subtype', 'id']).apply(lambda x: (
        x.iat[0, 2],  # type
        x.iat[0, 3],  # subtype
        x.iat[0, 4],  # sentence
        x.iat[0, 5],  # non sentence
        x['score'].mean()))
    score = pandas.DataFrame(score.to_list(), columns=['type', 'subtype', 'sentence', 'non sentence', 'score'])
    return score


def evaluate_by_type(by_pair):
    """Returns a data frame with mean scores by syntax error type

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_type : pandas.DataFrame
        The score collapsed on types, the data frame has the
        following columns: 'type', 'score'.

    """
    data = by_pair.score.groupby([by_pair['type']]).agg(
        n='count', score='mean', std='std').reset_index()
    return data


def evaluate(gold_file, submission_file, is_text=False):
    """Returns the score by sentences pair and by syntax type

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated pairs, the data frame has the columns:
        'sentence', 'non sentence' and 'score'.
    by_type : pandas.DataFrame
        The score collapsed on syntax errors types, the data frame has the
        following columns: 'type', 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    data = load_data(gold_file, submission_file, is_text)

    all_trials = evaluate_all(data)
    by_pair = evaluate_by_pair(data)
    by_type = evaluate_by_type(by_pair)
    by_pair.drop(['type', 'subtype'], axis=1, inplace=True)

    return all_trials, by_pair, by_type


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format='%.4f')
    print(f'  > Wrote {filename}')


def write_final(acc, filename):
    with open(filename, 'w') as fin:
        print(acc, file=fin)
    print(f'  > Wrote {filename}')


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path where to store the output files')
    parser.add_argument('-g', '--gold', type=str, required=True,
                        help='Path where the gold files lies.')
    parser.add_argument('-p', '--predicted', type=str, required=True,
                        help='Path where the pseudo-probabilities lie.')
    parser.add_argument('-k', '--kind', type=str, required=True, choices=['dev', 'test'],
                        help="Do we need to look for the dev, or the test files?")
    parser.add_argument('--task_name', type=str, default='lexical',
                        help="Name of folder where to look for gold and hypothesis files.")
    parser.add_argument('--is_text', action='store_true',
                        help="If activated, will only keep one voice (for text-based language models).")
    args = parser.parse_args(argv)

    kind = args.kind
    dataset = pathlib.Path(args.gold)
    submission = pathlib.Path(args.predicted)
    output = pathlib.Path(args.output)

    print(f'Evaluating syntactic {kind}...')
    gold_file = dataset / args.task_name / kind / 'gold.csv'
    submission_file = submission / args.task_name / f'{kind}.txt'

    all_trials, by_pair, by_type = evaluate(gold_file, submission_file, is_text=args.is_text)

    output.mkdir(exist_ok=True, parents=True)
    write_csv(
        all_trials, output / f'score_syntactic_{kind}_all_trials.csv')
    write_csv(
        by_pair, output / f'score_syntactic_{kind}_by_pair.csv')
    write_csv(
        by_type, output / f'score_syntactic_{kind}_by_type.csv')

    # write final score
    write_final(by_pair['score'].mean(), output / f'overall_accuracy_syntactic_{kind}.txt')


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)