# Evaluate LSTM language models (phones or words)

### 1) Download pre-trained model

Once you followed [the instructions](../evaluation.md) to download the evaluation dataset.
You can download the pre-trained model used in the paper:

```bash
mkdir babyslm_models

# LSTM trained on words (BPE) extracted from the Providence corpus
wget https://cognitive-ml.fr/downloads/baby-slm/models/LSTM_words/lstm_providence_128h.zip -P babyslm_models
unzip babyslm_models/lstm_providence_128h.zip -d babyslm_models
```

### 2) Evaluate it

Let us assume we want to evaluate the model on the syntactic task.

1) Compute probabilities:

```bash
BABYSLM_PATH=<DATA_LOCATION>/babyslm/syntactic
LSTM_PATH=babyslm_models/lstm_providence_128h/checkpoint_best.pt
DICT_PATH=babyslm_models/lstm_providence_128h

python scripts/compute_proba.py --input_path $BABYSLM_PATH --model_path $LSTM_PATH --dict_path $DICT_PATH \
    --mode dev --text --bos_eos --bpe_encode
```

Here, the flag `--bos_eos` indicates that beginning of sentences and end of sentences tokens should be used.

The flag `--bpe_encode` indicates that input stimuli should be BPE encoded.

2) Compute scores:

```bash
BABYSLM_PATH=<DATA_LOCATION>/babyslm
PROB_PATH=babyslm_models/lstm_providence_128h/babyslm/tmp
OUTPUT_PATH=babyslm_models/lstm_providence_128h/scores

python scripts/metrics/compute_syntactic.py --gold $BABYSLM_PATH \
    --predicted $PROB_PATH \
    --output $OUTPUT_PATH \
    --kind dev --is_text
```

You can check in `overall_accuracy_syntactic_dev.txt` that the model obtains a syntactic accuracy of 65.3% on the dev set.
