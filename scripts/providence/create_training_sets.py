import argparse
import os
import sys
from pathlib import Path
import math

from tqdm import tqdm


def sort_files_spkr_onset(files):
    def get_key(x):
        stem = x.stem
        return '_'.join(stem.split('_')[:6]), float(stem.split('_')[-2])
    return sorted(files, key=lambda x: get_key(x))


def create_symlink_from_original_files(subpath, sentence1_file, sentence2_file, phones1_file, phones2_file,
                                       audio1_file, audio2_file, out, duration, idx_pack):
    # Destination files
    training_set_path = Path('%gh/%02d' % (duration, idx_pack))
    dst_sentence1_file = out / training_set_path / 'sentences' / subpath.with_suffix('.txt')
    dst_sentence2_file = out / training_set_path / 'sentences_bpe_eos_bos' / subpath.with_suffix('.txt')
    dst_phones1_file = out / training_set_path / 'phones' / subpath.with_suffix('.txt')
    dst_phones2_file = out / training_set_path / 'phones_with_space' / subpath.with_suffix('.txt')
    dst_audio1_file = out / training_set_path / 'audio' / subpath.with_suffix('.wav')
    dst_audio2_file = out / training_set_path / 'audio_synthetized' / subpath.with_suffix('.wav')

    # Create dirs
    dst_sentence1_file.parent.mkdir(parents=True, exist_ok=True)
    dst_sentence2_file.parent.mkdir(parents=True, exist_ok=True)
    dst_phones1_file.parent.mkdir(parents=True, exist_ok=True)
    dst_phones2_file.parent.mkdir(parents=True, exist_ok=True)
    dst_audio1_file.parent.mkdir(parents=True, exist_ok=True)
    dst_audio2_file.parent.mkdir(parents=True, exist_ok=True)

    # Symlink
    if not dst_sentence1_file.is_symlink():
        os.symlink(sentence1_file, dst_sentence1_file)
    if not dst_sentence2_file.is_symlink():
        os.symlink(sentence2_file, dst_sentence2_file)
    if not dst_phones1_file.is_symlink():
        os.symlink(phones1_file, dst_phones1_file)
    if not dst_phones2_file.is_symlink():
        os.symlink(phones2_file, dst_phones2_file)
    if not dst_audio1_file.is_symlink():
        os.symlink(audio1_file, dst_audio1_file)
    if not dst_audio2_file.is_symlink():
        os.symlink(audio2_file, dst_audio2_file)


def create_smallest_packs(sentences1_path, sentences2_path, phones1_path, phones2_path, audio1_path, audio2_path, out, dur_pack, nb_packs):
    # h to ms
    target_dur = dur_pack*60*60*1000

    curr_nb_packs = 0
    cum_dur_pack = 0

    audio1_files = audio1_path.glob('**/*.wav')
    audio1_files = sort_files_spkr_onset(audio1_files)

    print("Starting to create %d training sets of %.2f hours." % (nb_packs, dur_pack))
    for audio1_file in tqdm(audio1_files):
        subpath = audio1_file.relative_to(audio1_path)
        phones1_file = phones1_path / subpath.with_suffix('.txt')
        phones2_file = phones2_path / subpath.with_suffix('.txt')
        sentence1_file = sentences1_path / subpath.with_suffix('.txt')
        sentence2_file = sentences2_path / subpath.with_suffix('.txt')
        audio2_file = audio2_path / subpath.with_suffix('.wav')

        onset, offset = audio1_file.stem.split('_')[-2:]
        onset, offset = float(onset), float(offset)
        dur = offset - onset
        cum_dur_pack += dur
        create_symlink_from_original_files(subpath, sentence1_file, sentence2_file, phones1_file, phones2_file,
                                           audio1_file, audio2_file, out, dur_pack, curr_nb_packs)

        # We have gathered enough data to create a pack
        # Reset variables and create next pack
        if cum_dur_pack > target_dur:
            curr_nb_packs += 1
            cum_dur_pack = 0

        if curr_nb_packs == nb_packs:
            break


def recursive_merge(folder1, folder2, out_folder):
    for folder in [folder1, folder2]:
        for src_file in folder.glob('**/*'):
            if src_file.is_file():
                if src_file.stem.startswith('fairseq') or src_file.stem.startswith('_seqs_cache'):
                    continue
                subpath = src_file.relative_to(folder)
                dst_file = out_folder / subpath
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                if not dst_file.is_symlink():
                    os.symlink(src_file, dst_file)


def merge_all_packs(out_path, min_dur, max_dur):
    print("Start merging packs together! This might take a while...")
    prev_dur = min_dur
    curr_dur = 2*prev_dur
    while curr_dur <= max_dur:
        input_path = out_path / ('%gh' % prev_dur)
        input_nb_packs = int(max_dur // prev_dur)
        for idx_child_pack, idx_pack in enumerate(range(0, input_nb_packs, 2)):
            left_pack, right_pack = idx_pack, idx_pack+1
            # we merge left_pack and right_pack
            left_path = input_path / ('%02d' % left_pack)
            right_path = input_path / ('%02d' % right_pack)
            output_path = out_path / ('%gh' % curr_dur) / ('%02d' % idx_child_pack)
            recursive_merge(left_path, right_path, output_path)
        print("Done with packs of %d hours" % curr_dur)
        prev_dur = prev_dur*2
        curr_dur = prev_dur*2


def check_if_power_two(x):
    return math.ceil(math.log2(x)) == math.floor(math.log2(x))


def main(argv):
    parser = argparse.ArgumentParser(description='Create 30mn, 1h, 2h, 4h, 8h, 16h, 32h, 64h, 128h training sets.')
    parser.add_argument('--sentences1', type=str, required=False, help="Where the .txt files "
                                                                       "containing sentences are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences')
    parser.add_argument('--sentences2', type=str, required=False, help="Where the .txt files "
                                                                       "containing sentences are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences_bpe_eos_bos')
    parser.add_argument('--phones1', type=str, required=False, help="Where the .txt files containing phones"
                                                                    "with no space are stored",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/phonemes')
    parser.add_argument('--phones2', type=str, required=False, help="Where the .txt files containing phones"
                                                                    "with spaces are stored",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/phonemes_with_space')
    parser.add_argument('--audio1', type=str, required=False, help="Where the audio files are stored",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/audio')
    parser.add_argument('--audio2', type=str, required=False, help="Where the audio files are stored",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/audio_synthetized/en-US-Wavenet-I')
    parser.add_argument('--out', type=str, required=False, help="Where the training sets will be stored",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/training_sets')
    parser.add_argument('--min_dur', type=float, default=0.5, required=False,
                        help="Cumulated duration of the smallest training set in hours (default to 0.5h)."
                             "Note that --min_dur should be a power of two.")
    parser.add_argument('--max_dur', type=float, default=128, required=False,
                        help="Cumulated duration of the biggest training set in hours (default to 128h)."
                             "Note that --max_dur should be a power of two.")
    args = parser.parse_args(argv)
    assert check_if_power_two(args.max_dur)
    assert check_if_power_two(args.min_dur)
    assert args.min_dur < args.max_dur
    args.sentences1 = Path(args.sentences1)
    args.sentences2 = Path(args.sentences2)
    args.phones1 = Path(args.phones1)
    args.phones2 = Path(args.phones2)
    args.audio1 = Path(args.audio1)
    args.audio2 = Path(args.audio2)
    args.out = Path(args.out)

    nb_packs = int(args.max_dur // args.min_dur)
    # create_smallest_packs(args.sentences1, args.sentences2, args.phones1, args.phones2, args.audio1, args.audio2,
    #                       args.out, args.min_dur, nb_packs)
    merge_all_packs(args.out, args.min_dur, args.max_dur)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)