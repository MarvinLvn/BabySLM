import argparse
import sys
from pathlib import Path

import pylangacq
from tqdm import tqdm

from utils.audio_extraction import cut_wave_file


def clean_sentence(tokens):
    sentence = ' '.join([t.word for t in tokens if t.word != 'CLITIC'
                         and t.word.replace('/', '') not in ['?', '!', '.', '...', ',', ';']])
    sentence = sentence.replace('_', ' ')
    return sentence.lower()


def write_annotation(sentence, out_file):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fout:
        fout.write(sentence+'\n')


def extract_segments(wav_file, annotation_file, out):
    annotation = pylangacq.read_chat(str(annotation_file))
    if len(annotation.utterances()) == 0:
        return 0, 0
    child_id = annotation.headers()[0]['Participants']['CHI']['name']
    date_rec = list(annotation.dates_of_recording())[0].strftime('%m_%d_%Y')

    participants = set(annotation.participants()) - set(['CHI', 'TOY', 'ENV', 'TO1', 'TO2', 'TO3', 'NON'])
    lost, cum_dur = 0, 0
    for participant in sorted(participants):
        for utterance in annotation.utterances(participants=participant):
            time_marks = utterance.time_marks
            if time_marks is not None:
                # Get onset, offset, and sentence
                onset, offset = time_marks
                sentence = clean_sentence(utterance.tokens)
                if len(sentence) == 0:
                    lost += 1
                    continue
                # Cut wav file and annotation
                basename = '_'.join([wav_file.stem, date_rec, participant, str(onset), str(offset)])
                out_annotation_file = out / 'sentences' / ('%s_%s' % (child_id, participant)) / (basename + '.txt')
                out_wav_file = out / 'audio' / ('%s_%s' % (child_id, participant)) / (basename + '.wav')
                dur = offset - onset
                cum_dur += dur
                cut_wave_file(wav_file, out_wav_file, onset, dur)
                write_annotation(sentence, out_annotation_file)
            else:
                lost += 1
    return lost, cum_dur/1000



def main(argv):
    parser = argparse.ArgumentParser(description='Extract speech utterances '
                                                 'pronounced by adult in the Providence corpus.')
    parser.add_argument('--audio', type=str, required=False, help="Where the .wav files are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/recordings/raw')
    parser.add_argument('--annotation', type=str, required=False, help="Where the .cha files are stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/annotations/cha/raw')
    parser.add_argument('--out', type=str, required=False, help="Where the dataset will be stored.",
                        default='/private/home/marvinlvn/DATA/CPC_data/train/providence/cleaned')
    args = parser.parse_args(argv)
    args.audio = Path(args.audio)
    args.annotation = Path(args.annotation)
    args.out = Path(args.out)

    wav_files = list(args.audio.glob('**/*.wav'))
    lost_sentences = 0
    cum_dur = 0
    for wav_file in tqdm(wav_files):
        subpath = wav_file.relative_to(args.audio)
        annotation_file = args.annotation / subpath.with_suffix('.cha')
        if not annotation_file.is_file():
            raise ValueError("I found this audio file: %s \n"
                             "But I can't find its annotation file: %s" % (wav_file, annotation_file))
        lost, dur = extract_segments(wav_file, annotation_file, args.out)
        lost_sentences += lost
        cum_dur += dur
    print("Lost %d sentences that were empty." % lost_sentences)
    print("Found %.2f s of speech" % (cum_dur))
    print("Found %.2f hours of speech" % (cum_dur / 3600))

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)