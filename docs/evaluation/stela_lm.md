# Evaluate STELA: CPC + K-means + LSTM (speech-based)

### 1) Download pre-trained models

Once you followed [the instructions](../evaluation.md) to download the evaluation dataset.
You can download some of the pre-trained models used in the paper:

```bash
mkdir babyslm_models

# STELA trained on 1,024 hours of audiobooks (this is the one we'll use in the example below)
wget https://cognitive-ml.fr/downloads/baby-slm/models/STELA/stela_audiobooks_1024h.zip -P babyslm_models
unzip babyslm_models/stela_audiobooks_1024h.zip -d babyslm_models

# STELA trained on 1,024 hours of child-centered long-form recordings (optional)
wget https://cognitive-ml.fr/downloads/baby-slm/models/STELA/stela_seedlings_1024h.zip -P babyslm_models
unzip babyslm_models/stela_seedlings_1024h.zip -d babyslm_models

# STELA trained on 128 hours of naturalistic recordings of child-parent interactions (optional)
wget https://cognitive-ml.fr/downloads/baby-slm/models/STELA/stela_providence_128h.zip -P babyslm_models
unzip babyslm_models/stela_providence_128h.zip -d babyslm_models
```

To run this part of the code, we need to follow the [installation instructions specific to STELA](../installation/stela.md).

### 2) Evaluate it

Let us assume we want to evaluate the STELA model trained on 1,024h of audiobooks on the lexical development set.

/!\ Before running any of the command line below, you must replace the field `pathCheckpoint` in `cpc_small/checkpoint_args.json` and `kmeans50/args.json` with the actual path where the model is stored.
Otherwise, the scripts won't be able to load the model!

1) Discretize the input stimuli:

```bash
KMEANS_PATH=babyslm_models/stela_audiobooks_1024h/kmeans50/checkpoint_last.pt
BABYSLM_PATH=<DATA_LOCATION>/babyslm
OUTPUT_PATH=babyslm_models/stela_audiobooks_1024h/quantized/lexical/dev

python scripts/audio_lm/quantize_audio.py $KMEANS_PATH $BABYSLM_PATH $OUTPUT_PATH --file_extension .wav
```

2) Compute probabilities:

```bash
QUANTIZED_INPUT=babyslm_models/stela_audiobooks_1024h/features_lexical
PATH_LSTM=babyslm_models/stela_audiobooks_1024h/lstm/checkpoint_best.pt
DICT_PATH=babyslm_models/stela_audiobooks_1024h/lstm

python scripts/compute_proba.py --input_path $QUANTIZED_INPUT \
  --model_path $PATH_LSTM --mode dev --dict_path $DICT_PATH
```

3) Compute scores:

```bash
BABYSLM_PATH=<DATA_LOCATION>/babyslm
PROB_PATH=babyslm_models/stela_audiobooks_1024h/lstm
OUTPUT_PATH=babyslm_models/stela_audiobooks_1024h/scores

python scripts/metrics/compute_lexical.py -g $BABYSLM_PATH \
  -p $PROB_PATH \
  -o $OUTPUT_PATH \
  -k dev
```

You can check in `overall_accuracy_lexical_dev.txt` that the model obtains a lexical accuracy of 59.5% on the dev set.

This works similarly for the syntactic evaluation where you must replace `compute_lexical.py` by `compute_syntactic.py`.

This also works similarly for the test set, where you must replace occurrences of `dev` by `test`.