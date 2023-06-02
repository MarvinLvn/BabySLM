# Evaluate BabyBERTa (text-based)


Before running any of the command line below, you must follow [the instructions](../evaluation.md) to download the evaluation dataset and the [installation instructions specific to BabyBERTA](../installation/babyberta.md).
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
