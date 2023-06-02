# Evaluate BabyBERTa (text-based)

### 1) Download pre-trained model

Once you followed [the instructions](../evaluation.md) to download the evaluation dataset and the [installation instructions specific to BabyBERTA](../installation/babyberta.md)

You can download the pre-trained model used in the paper (originally downloaded from https://github.com/phueb/BabyBERTa):

```bash
mkdir babyslm_models

# LSTM trained on words (BPE) extracted from the Providence corpus
wget https://cognitive-ml.fr/downloads/baby-slm/models/BabyBERTa/BabyBERTa_AO-CHILDES.zip -P babyslm_models
unzip babyslm_models/BabyBERTa_AO-CHILDES.zip -d babyslm_models
```

### 2) Evaluate it

Let us assume we want to evaluate BabyBERTa on the syntactic task.

1) Compute probabilities:

```bash
BABYSLM_PATH=<DATA_LOCATION>/babyslm/syntactic
OUTPUT_PATH=score_babyberta
python scripts/extract_prob_babyberta.py --input_path $BABYSLM_PATH \
  --out $OUTPUT_PATH \
  --model babyberta1 \
  --task syntactic --mode dev 
```

2) Compute scores:

```bash
BABYSLM_PATH=<DATA_LOCATION>/babyslm
PROB_PATH=score_babyberta/babyslm/tmp
OUTPUT_PATH=score_babyberta

python scripts/metrics/compute_syntactic.py -g $BABYSLM_PATH \
  -p $PROB_PATH \
  -o $OUTPUT_PATH \
  -k dev --is_text
```

You can check in `overall_accuracy_syntactic_dev.txt` that BabyBERTa obtains a syntactic accuracy of 70.4% on the dev set.
