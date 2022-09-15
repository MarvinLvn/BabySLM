import argparse
import os
import sys
from pathlib import Path

from pyannote.core import Segment
from pyannote.database.util import load_rttm
from tqdm import tqdm

from utils.audio_extraction import cut_wave_file


def main(argv):
    parser = argparse.ArgumentParser(description='Compute the intersection of human-annotated speech segments'
                                                 'with segments identified as SPEECH by the voice type classifier.')
    parser.add_argument('--audio', type=str, required=False, help="Where the .wav files containing speech segments are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/audio')
    parser.add_argument('--annotation', type=str, required=False,
                        help="Where the sentence files are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned/sentences')
    parser.add_argument('--rttm', type=str, required=False, help="Where the .rttm files containing vtc annotations are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/annotations/vtc/raw')
    args = parser.parse_args(argv)
    args.audio = Path(args.audio)
    args.annotation = Path(args.annotation)
    args.rttm = Path(args.rttm)

    for rttm_file in tqdm(args.rttm.glob('**/*.rttm')):
        uri = rttm_file.stem
        if uri == 'Violet_030200':
            # dummy file that provokes bugs
            continue
        rttm = load_rttm(rttm_file)[uri]
        for i, wav_file in enumerate(args.audio.glob('**/%s_*.wav' % uri)):
            # get data
            stem = wav_file.stem
            onset, offset = stem.split('_')[-2:]
            onset, offset = float(onset), float(offset)
            # delete files whose dur. is 0
            if (offset - onset) / 1000 > 0.5:
                subpath = wav_file.relative_to(args.audio)
                txt_file = args.annotation / subpath.with_suffix('.txt')
                os.remove(wav_file)
                os.remove(txt_file)
                continue
            original_seg = Segment(onset/1000, offset/1000)
            rttm[original_seg] = 'original_seg'
            # compute new boundaries (keep only what has been classified as SPEECH by the vtc)
            overlap = rttm.get_overlap(labels=["SPEECH", "original_seg"])
            if len(overlap) != 0:
                new_onset, new_offset = int(overlap[0].start * 1000), int(overlap[-1].end * 1000)
                new_duration = new_offset - new_onset
                if new_onset != onset or new_offset != offset:
                    print("Cutting %d th seg:" % i)
                    # cut the audio file
                    new_basename = '_'.join(wav_file.stem.split('_')[:-2])
                    new_basename += '_' + str(new_onset) + '_' + str(new_offset)
                    new_wav_file = wav_file.parent / (new_basename + '.wav')
                    new_beg = new_onset - onset
                    print(wav_file.stem, 'to', new_basename)
                    cut_wave_file(wav_file, new_wav_file, new_beg, new_duration)
                    # remove the older file
                    os.remove(wav_file)

                    # rename the annotation file
                    subpath = wav_file.relative_to(args.audio)
                    txt_file = args.annotation / subpath.with_suffix('.txt')
                    new_txt_file = txt_file.parent / (new_basename + '.txt')
                    os.rename(txt_file, new_txt_file)
            rttm.__delitem__(original_seg)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)