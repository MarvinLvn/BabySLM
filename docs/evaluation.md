# Access to evaluation stimuli

Work in progress (Marvin)

# Evaluate your own model

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
- the first column contains the filename of the stimuli considered
- the second column contains the probability returned by the model.

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

1 - First, compute the probabilities

In order to compute the probabilities on the tasks corpora, run this command line:

```bash

sh scripts/models/ngram_model/compute_prob_for_all_ngram_models.sh -i <GOLDS_FOLDER> -m <TRAINED_MODELS> -t <MODEL_TYPE> -e <MODE> -o <OUTPUT_FOLDER>

```

Where:
- `-i  Folder storing the gold files for test or dev.`
- `-m  Path to where all the trained models are stored.`
- `-t  The type of ngram model, must be 'unigram' for unigram model and 'ngram' otherwise.`
- `-e  Evaluate the models on 'test' or 'dev'.`
- `-o  Path to where the computed probabilities will be stored.`

For example, if we assume that the evaluation corpora for lexical and syntactic tasks are stored in a `data/model_evaluation/` folder and that the trained ngram models are stored in a `trained_models/ngrams/` folder, then run this script for computing the probabilities on the test corpus for the lexical task for the trigram models :

```bash

sh scripts/models/ngram_model/compute_prob_for_all_ngram_models.sh -i data/model_evaluation/lexical -m trained_models/ngrams/trigrams -t ngram -e test -o results/trigrams

```

2 - Second, compute the scores

```bash
sh scripts/models/ngram_model/compute_scores_for_all_models.sh -i <GOLDS_FOLDER> -p <PREDICTED_PROBABILITIES> -k <MODE> -t <TASK>
```

Where:
- `-i  Path containing gold files.`
- `-p  Path containing files storing the predicted probabilities.`
- `-k  The type of evaluated dataset. Must be 'dev' or 'test'.`
- `-t  The task.`

For example, if we assume that the evaluation corpora for lexical and syntactic tasks are stored in a `data/model_evaluation/` folder and that the predicted probabilities by a, lets say, trigram model are stored in a `results/trigrams` folder, then run this script for computing the scores on the test corpus for the lexical:

```bash
sh scripts/models/ngram_model/compute_scores_for_all_models.sh -i data/model_evaluation/ -p results/unigrams/ -k dev -t lexical
```

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
