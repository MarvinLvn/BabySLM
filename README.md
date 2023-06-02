<p align="center" width="30%">
<img src="./docs/images/babyslm_logo.png"> 
</p>

# BabySLM: language-acquisition-friendly benchmark of self-supervised spoken language models

Welcome to this repository where you'll find all you need to evaluate your language model at:
1) the lexical level using a spot-the-word task (available in audio or phonetic form; see Table 1)
2) the syntactic level using a grammatical acceptability judgment task (available in audio, phonetic or orthographic form; see Table 2)

# Getting started

You'll probably want to start from there:

- [How to download the evaluation data? How to evaluate my own model?](docs/evaluation.md)

# Examples of stimuli

<center>

| Word   | Pseudo-word                                                 | Word   | Pseudo-word                                                 |
|--------|-------------------------------------------------------------|--------|-------------------------------------------------------------|
| hello  | lello <br> pello <br> sero <br> dello <br> sello <br>       | cookie | kootie <br> koonie <br> roodie <br> rootie <br> boonie <br> |

Table 1: Minimal pairs of real and pseudo-words used in the spot-the-word lexical task.
</center>

<center>

| Phenomenon                | Sentence example                                                      |
|---------------------------|-----------------------------------------------------------------------|
| Adjective-noun order      | ✓ The good mom. <br> ✗ The mom good.                                  |
| Noun-verb order           | ✓ The dragon says. <br> ✗ The says dragon.                            |
| Anaphor-gender agreement  | ✓ The dad cuts himself. <br> ✗ The dad cuts herself.                  |
| Anaphor-number agreement  | ✓The boys told themselves. <br> ✗ The boys told himself.              |
| Determiner-noun agreement | ✓ Each good sister. <br> ✗ Many good sister.                          |
| Noun-verb agreement       | ✓ The prince needs the princess. <br> ✗ The prince need the princess. |

Table 2: Minimal pairs of grammatical (✓)  and ungrammatical (✗) sentences used in the syntactic task.
</center>

# Reproduce the BabySLM benchmark

If you want to go further:

- [How to download the pre-trained models used in the paper and evaluate them?](docs/evaluation/entry_point.md)
- [How to retrain the models used in the paper?](docs/training/entry_point.md)
- [How to prepare the training sets used in the paper?](docs/data.md)
- [How to recreate the lexical evaluation?](https://github.com/MarvinLvn/ChildDirectedLexicalTest)
- [How to recreate the syntactic evaluation?](https://github.com/MarvinLvn/ChildDirectedSyntacticTest)

# How to cite?

```text
@inproceedings{lavechin2023baby,
title={BabySLM: language-acquisition-friendly benchmark of self-supervised spoken language models},
author={Lavechin, Marvin and Sy, Yaya and Titeux, Hadrien and Bland{\'o}n, Mar{\'\i}a Andrea Cruz and R{\"a}s{\"a}nen, Okko and Bredin, Herv{\'e} and Dupoux, Emmanuel and Cristia, Alejandrina},
year={2023},
booktitle = {Interspeech}
}
```

Additionnally, if you BabyBERTa, please cite:

```text
@inproceedings{huebner2021babyberta,
  title={BabyBERTa: Learning more grammar with small-scale child-directed language},
  author={Huebner, Philip A and Sulem, Elior and Cynthia, Fisher and Roth, Dan},
  booktitle={Proceedings of the 25th conference on computational natural language learning},
  pages={624--646},
  year={2021}
}
```

If you use the Providence corpus, please cite:

```text
@inproceedings{borschinger2013joint,
  title={A joint model of word segmentation and phonological variation for English word-final/t/-deletion},
  author={B{\"o}rschinger, Benjamin and Johnson, Mark and Demuth, Katherine},
  booktitle={Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1508--1516},
  year={2013}
}
```
If you use the LibriVox corpus, please cite:

```text
@article{kearns2014librivox,
  title={Librivox: Free public domain audiobooks},
  author={Kearns, Jodi},
  journal={Reference Reviews},
  volume={28},
  number={1},
  pages={7--8},
  year={2014},
  publisher={Emerald Group Publishing Limited}
}
```