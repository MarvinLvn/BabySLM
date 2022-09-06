# Access to evaluation stimuli

Work in progress (Marvin)

# Evaluate your own models

To evaluate your own model, you can use `scripts/compute_lexical.py` or `scripts/compute_syntactic.py`. 
The script expects the following parameters:
- `--gold` path to the child_zerospeech dataset
- `--predicted` path to the folder where the file dev.txt or test.txt lies. This file should contain probabilities returned by the model (see below) 
- `--output` where to the store the output
- `--kind` either dev or test, depending on whether you want to compute the score on the dev or the test set.
- `--is_text` indicates if the model is a text-based (in which case the script will compute the score on only one voice) or audio-based (will consider all voices)

The file that you must provide to compute the lexical score or the syntactic score should look like:

```txt
impacts-en-US-Wavenet-B -3.6343493461608887
ɪ_m_p_ɛ_k_t_s-en-US-Wavenet-B -3.651639461517334
impacts-en-US-Wavenet-B -3.6343493461608887
ɪ_m_p_ʌ_s_t_s-en-US-Wavenet-B -3.6196532249450684
impacts-en-US-Wavenet-B -3.6343493461608887
```

where:
- the first column contain the filename of the stimuli consider
- the second column contain the probability returned by the model.

Please note that if you evaluate a text-based language model, there should be as many lines as there are word/nonwords for a single voice (let's say en-US-Wavenet-B if you consider the dev set).
However, if you evaluate an audio-based language model, there should be as many lines as there are word/nonwords for all the voices (2 voices for the dev set, 8 voices for the test set).
 

# Evaluate text-based models

### Evaluate LSTM language models

Compute probabilities associated to each stimuli

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

### Evaluate N-grams

Work in progress (Yaya)

# Evaluate audio-based language models

### Evaluate CPC + K-means + LSTM (speech-based)

Work in progress (Marvin)

# Evaluate BabyBERTa (text-based)

If you want to evaluate BabyBERTa, you must install the following conda environment:

```bash
# Install dependencies
conda env create -f babyberta_env.yml & conda activate babyberta
```

Work in progress (Marvin)


