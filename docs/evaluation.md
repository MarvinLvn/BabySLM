# Download the development and test sets

You can use the following command lines to download the dataset in the `<DATA_LOCATION>` folder:

```
mkdir -p <DATA_LOCATION>/babyslm/{lexical,syntactic}
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip -P <DATA_LOCATION>/babyslm/lexical
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip -P <DATA_LOCATION>/babyslm/lexical
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip -P <DATA_LOCATION>/babyslm/syntactic
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip -P <DATA_LOCATION>/babyslm/syntactic
```

Alternatively, you can click on the following links to download the evaluation stimuli

<center>

| Lexical evaluation                                                                            | Syntactic evaluation                                                                           |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [dev](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip) (728 MB)    | [dev](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip) (659 MB)   |
| [test](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip) (10.9 GB) | [test](https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip) (7.7 GB) |

</center>

# Evaluate your own model

To evaluate your own model, you can use `scripts/compute_lexical.py` or `scripts/compute_syntactic.py`. 
The script expects the following parameters:
- `--gold` path to the evaluation dataset
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

# Evaluate models used in the paper

How to evaluate:
- [STELA (audio)](evaluation/stela_lm.md)
- [LSTM (phones or words)](evaluation/text_lstm_lm.md)
- [N-grams (phones or words)](evaluation/ngram_lm.md)
- [BabyBERTa (words)](evaluation/babyberta_lm.md)






