import sys
sys.path.append('.')

import pandas as pd
from scripts.providence.phonemize import phonemize_sentence, load_phonemizers
from paraphone.workspace import Workspace
from pathlib import Path
import os


def phonemize_input(data, stimuli_path, phonemizers, key):
    if not (stimuli_path.parent / 'gold_phonemize.csv').is_file():
        phonemized_sentences = [' '.join(phonemize_sentence(s, stimuli_path, phonemizers)[0]).replace(' \t ', ' <SEP> ') for s in data[key]]
        data[key] = phonemized_sentences
        data.to_csv(stimuli_path.parent / 'gold_phonemize.csv', sep=',', index=False)
    else:
        data = pd.read_csv(stimuli_path.parent / 'gold_phonemize.csv')
    return data


def load_stimuli_text(path, kinds, debug=False, phonemize=False):
    out = []
    if path.stem == 'lexical':
        key = 'phones'
    else:
        key = 'transcription'
    if phonemize:
        # Not the cleanest way to do it :/
        workspace = Workspace(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
        phonemizers = load_phonemizers(workspace)

    for kind in kinds:
        stimuli_path = path / kind / 'gold.csv'
        data = pd.read_csv(stimuli_path)
        if phonemize:
            data = phonemize_input(data, stimuli_path, phonemizers, key)
        if debug:
            data = data[:50]
        voices = data['voice'].unique()
        data = data[data['voice'] == voices[0]][[key, 'filename']]
        data.columns = ['transcription', 'filename']
        out.append(data)
    print('Stimuli loaded.')
    return out