# Download the development and test sets

You can use the following command lines to download the dataset in the `<DATA_LOCATION>` folder.
Only the development sets will be necessary to run the example.

```
mkdir -p <DATA_LOCATION>/babyslm/{lexical,syntactic}

# Download dev
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip -P <DATA_LOCATION>/babyslm/lexical
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip -P <DATA_LOCATION>/babyslm/syntactic

unzip <DATA_LOCATION>/babyslm/lexical/dev.zip -d <DATA_LOCATION>/babyslm/lexical
unzip <DATA_LOCATION>/babyslm/syntactic/dev.zip -d <DATA_LOCATION>/babyslm/syntactic

# Download test
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip -P <DATA_LOCATION>/babyslm/lexical
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip -P <DATA_LOCATION>/babyslm/syntactic

unzip <DATA_LOCATION>/babyslm/lexical/test.zip -d <DATA_LOCATION>/babyslm/lexical
unzip <DATA_LOCATION>/babyslm/syntactic/test.zip -d <DATA_LOCATION>/babyslm/syntactic
```

Alternatively, you can click on the following links to download the evaluation stimuli

<center>

| Lexical evaluation                                                                            | Syntactic evaluation                                                                           |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [dev](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip) (728 MB)    | [dev](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip) (659 MB)   |
| [test](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip) (10.9 GB) | [test](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip) (7.7 GB) |

</center>

# Evaluate your own model

Let us run two examples!

### Lexical evaluation (spot-the-word)

In `example/librivox_1024h/lexical/dev.txt`, you'll find the probabilities computed on the dev set of the lexical task for a model trained on 1,024h of speech from Librivox.
The file looks like:

```txt
impacts-en-US-Wavenet-B -3.6343493461608887
ɪ_m_p_ɛ_k_t_s-en-US-Wavenet-B -3.651639461517334
impacts-en-US-Wavenet-B -3.6343493461608887
ɪ_m_p_ʌ_s_t_s-en-US-Wavenet-B -3.6196532249450684
impacts-en-US-Wavenet-B -3.6343493461608887
```

where:
- the first column contains the filename of the stimulus.
- the second column contains the probability returned by the model.

Assuming you are in the `BabySLM` folder, you can compute the lexical accuracy by running:

```bash
python scripts/metrics/compute_lexical.py --gold <DATA_LOCATION>/babyslm \
--predicted example/librivox_1024h \ 
--output example/librivox_1024h/lexical/dev_scores \
--kind dev
```

where:
- the `--gold` argument is the path to the evaluation dataset.
- the `--predicted` argument is the path to the folder containing the probabilities returned by the model (i.e., `dev.txt` or `test.txt`).
- the `--output` argument indicates where to store the output files.
- the `--kind` argument is either dev or test, depending on whether you want to compute the score on the dev or the test set.

The file `overall_accuracy_lexical_dev.txt` should contain the string `59.5` which is the lexical accuracy obtained by the model on the dev set.

### Syntactic evaluation (grammatical acceptability judgment)

Let's compute the syntactic score obtained by BabyBERTa this time! It works pretty similarly as above:
The `example/babyberta/syntactic/dev.txt` file looks like:

```bash
The_woman_chews_en-US-Wavenet-B -15.0255126953125
The_chews_woman_en-US-Wavenet-B -15.721282958984375
The_person_learns_en-US-Wavenet-B -15.446614265441895
The_learns_person_en-US-Wavenet-B -15.838199615478516
```
But this time, we only have stimulus from a single voice (`en-US-Wavenet-B`). This is because BabyBERTa is a text-based language model and running it on different voices would yield the exact same results.

We compute the syntactic accuracy by running:

```bash
python scripts/metrics/compute_syntactic.py --gold ~/Documents/babyslm \
--predicted example/babyberta \
--output example/babyberta/syntactic/dev_scores \
--kind dev \
--is_text
```

Here, we have an additional `--is_text` flag indicating that the model works with text (orthographic or phonetic form) and that only a single voice should be considered.
The file `overall_accuracy_syntactic_dev.txt` should contain the string `70.4` which is the syntactic accuracy obtained by BabyBERTa on the dev set.

### Final notes


















- `--is_text` indicates if the model is a text-based (in which case the script will compute the score on only one voice) or audio-based (will consider all voices)

Please note that if you evaluate a text-based language model, there should be as many lines as there are word/nonwords for a single voice (let's say en-US-Wavenet-B if you consider the dev set).
However, if you evaluate an audio-based language model, there should be as many lines as there are word/nonwords for all the voices (2 voices for the dev set, 8 voices for the test set).






# Evaluate models used in the paper

How to evaluate:
- [STELA (audio)](evaluation/stela_lm.md)
- [LSTM (phones or words)](evaluation/text_lstm_lm.md)
- [N-grams (phones or words)](evaluation/ngram_lm.md)
- [BabyBERTa (words)](evaluation/babyberta_lm.md)






