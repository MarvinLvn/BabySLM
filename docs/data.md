## Download the Providence corpus

The original dataset is available on [Phonebank](https://gin.g-node.org/EL1000/providence).
However, a child-project version can be found [there](https://gin.g-node.org/EL1000/providence). 
This is where we'll start from.
You can gain access to the data following [these guidelines](https://gin.g-node.org/EL1000/EL1000).

## Required dependencies 

If you want to prepare the Providence corpus, you must install this conda env:

``````bash
# Install dependencies
conda env create -f data_prep.yml
conda activate provi
# Install paraphone: used to phonemized sentences
git clone ssh://git@gitlab.cognitive-ml.fr:1022/mlavechin/paraphone.git
cd paraphone
pip install -e .
``````

To force align the corpus, we'll need `abkhazia` whose installation instructions can be found [there](https://docs.cognitive-ml.fr/abkhazia/install.html).

Tips for cluster users for abkhazia installation: 
- `gfortran` (required dependency) comes with `module load gcc`
- `clang ++` (required dependency) comes with `module load llvm`

## Preparing the training sets

First, we need to extract speech segments along with their annotations:

```bash
python scripts/providence/extract_providence.py --audio ~/DATA/CPC_data/train/providence/recordings/raw \
  --annotation ~/DATA/CPC_data/train/providence/annotations/cha/raw \
  --out ~/DATA/CPC_data/train/providence/cleaned
```

As human-annotated utterances boundaries are a bit off, we'll correct them with what the vtc has been identified as SPEECH (we take the intersection of HUMAN & VTC).
WARNING: This script will directly recompute boundaries and will modify the files in the `sentences` and the `audio` folders. You might want to save them before running this command:

```bash
python scripts/providence/correct_boundaries.py --audio ~/DATA/CPC_data/train/providence/cleaned/audio \
  --annotation ~/DATA/CPC_data/train/providence/cleaned/sentences \
  --rttm ~/DATA/CPC_data/train/providence/annotations/vtc/raw
```

Then, we need to phonemize sentences:

```bash
python scripts/providence/phonemize_sentences.py --sentences ~/DATA/CPC_data/train/providence/cleaned/sentences \
  --out ~/DATA/CPC_data/train/providence/cleaned/phonemes
```

This will create new folders `phonemes` and `phonemes_with_space` that contains the phonemized version of the utterances without and with spaces, respectively.
It will also clean the `sentences` folder by removing punctuations and special characters such as ^. You can deactivate this behavior with the `--no_clean` flag.

We BPE-encode sentences (to later train a BPE-LSTM):

```bash
python scripts/providence/bpe_encode.py --sentences ~/DATA/CPC_data/train/providence/cleaned/sentences
```

If you want to synthetize sentences from Providence, you can run:
```bash
python scripts/providence/synthetize.py --credentials_path /path/to/credentials.json \
  
```

Then convert the .ogg to .wav files with:

```bash
for ogg in /path/to/providence/audio_synthetized/en-US-Wavenet-I/*/*.ogg; do 
  ffmpeg -i ${ogg} -acodec pcm_s16le -ac 1 -ar 16000 ${ogg%.*}.wav; 
done;
```

Last, we create 30mn, 1h, 2h, ... 128h training sets with the command:

```bash
python scripts/providence/create_training_sets.py --sentences1 ~/DATA/CPC_data/train/providence/cleaned/sentences \
  --sentences2 ~/DATA/CPC_data/train/providence/cleaned/sentences_bpe \
  --phones1 ~/DATA/CPC_data/train/providence/cleaned/phonemes \
  --phones2 ~/DATA/CPC_data/train/providence/cleaned/phonemes_with_space \
  --audio ~/DATA/CPC_data/train/providence/cleaned/audio \
  --out ~/DATA/CPC_data/train/providence/cleaned/training_sets
```

