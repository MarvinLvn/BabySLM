# Evaluate LSTM language models (phones or words)

Compute probabilities associated to each stimuli:

```bash
MODEL=/path/to/lstm_text/phones/128h/00
EVAL_SET=/path/to/child_zerospeech
python scripts/compute_proba.py --input_path $EVAL_SET/lexical --model_path $MODEL/checkpoint_best.pt \
    --mode [MODE] [OPTIONAL_FLAGS] --text 
```

where: 
- [MODE] can be dev or test, depending on whether you want to evaluate the model on the development or the test set.
- [OPTIONAL_FLAGS] can be `--phonemize` if you want to consider the phonetic transcription of the stimuli (for the syntactic task), it's not used for the lexical task that is always performed on the phonetic transcription of the stimuli.
- [OPTIONAL_FLAGS] can be `--remove_word_spaces` if you want to remove word spaces in the stimuli (for instance, if you trained a phone-based LSTM without considering spaces as a token). No effect on the lexical task. 
- `--text` is a flag that forces the script to use text stimuli - consider only one voice -. It is mandatory if you want to evaluate text-based models.
The command works the same for evaluating on the syntactic task (you'll just have to replace each occurrence of lexical by syntactic in the above command).

Compute score:

```bash

python scripts/metrics/compute_lexical.py --output $MODEL/child_zerospeech/lexical --gold $EVAL_SET \
    --predicted $MODEL/child_zerospeech/predicted_probabilities --kind <MODE> --is_text
```

where:
- `--kind` is dev or test.
- `--is_text` indicates if only one voice should be considered when computing the score (mandatory for text-based language models).
- `--predicted` points the the folder containing either dev.txt (if --kind == dev) or test.txt (if --kind == test) containing the probabilities returned by the model
