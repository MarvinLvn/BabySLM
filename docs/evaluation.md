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

# Download test (you can skip this for now)
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

### 1) Lexical evaluation (spot-the-word)

In `example/librivox_1024h/lexical/dev.txt`, you'll find the probabilities computed on the lexical dev set for a model trained on 1,024h of speech from Librivox.
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

You can check in `overall_accuracy_lexical_dev.txt` that the model obtains a lexical accuracy of 59.5% on the dev set.

### 2) Syntactic evaluation (grammatical acceptability judgment)

Let's compute the syntactic score obtained by BabyBERTa this time! 

It works pretty similarly as above. The `example/babyberta/syntactic/dev.txt` file looks like:

```bash
The_woman_chews_en-US-Wavenet-B -15.0255126953125
The_chews_woman_en-US-Wavenet-B -15.721282958984375
The_person_learns_en-US-Wavenet-B -15.446614265441895
The_learns_person_en-US-Wavenet-B -15.838199615478516
```
But here, we only have stimulus from a single voice (`en-US-Wavenet-B`). This is because BabyBERTa is a text-based language model and running it on different voices would yield the exact same results.

We compute the syntactic accuracy by running:

```bash
python scripts/metrics/compute_syntactic.py --gold ~/Documents/babyslm \
--predicted example/babyberta \
--output example/babyberta/syntactic/dev_scores \
--kind dev \
--is_text
```

Here, we have an additional `--is_text` flag indicating that the model works with text (orthographic or phonetic form) and that only a single voice should be considered.
You can check in `overall_accuracy_syntactic_dev.txt` that BabyBERTa obtains a syntactic accuracy of `70.4%` on the dev set.

### 3) Final notes

Your turn! You can extract probabilities using your own model and storing them following the patterns provided in the `example/babyberta` or `example/librivox_1024h` folders.
Might be helpful to check how we implemented this for [LSTM](scripts/compute_proba.py), [BabyBERTa](scripts/extract_prob_babyberta.py), and [STELA](scripts/compute_proba.py). 

### 4) Going further

How to download the pre-trained models used in the paper and evaluate them:
- [STELA (audio)](evaluation/stela_lm.md)
- [LSTM (phones or words)](evaluation/text_lstm_lm.md)
- [N-grams (phones or words)](evaluation/ngram_lm.md)
- [BabyBERTa (words)](evaluation/babyberta_lm.md)






