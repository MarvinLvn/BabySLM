# Behavioral probing of language acquisition models at the phonetic, lexical and syntactic level

<p align="center" width="100%">
<img src="./docs/benchmarking_speech_acquisition_transp.png"> 
</p>

Welcome to this repository where you'll find all you need to evaluate your language acquisition model at:
1) the phonetic level using a machine ABX discrimination task (audio-based models)
2) the lexical level using a spot-the-word task (audio-based or phone-based models)
3) the syntactic level using a spot-the-grammatical-sentence task (audio-based, phone-based or word-based model)

Task 1) is a distance-based task: your model needs to return representations of some audio files (discrete or continuous)

Tasks 2) and 3) are probability-based tasks: your model must return the probability associated to a stimuli (audio, phonetic transcription, or word transcription). 
It's up to you to decide how you can extract a relevant probability from your model.

# Which tasks are used to evaluate the model ?

<details>
<summary><b>(Click to expand)</b></summary>

1) Phonetic evaluation, the machine sound ABX discrimination task:
   - The model receives three spoken triphones A=<em>**/bit/**</em>, B=<em>**/bat/**</em>, X=<em>**/bit/**</em>.
   - Representations are returned by the model: R_A, R_B, and R_X.
   - The model is considered to be right if d(R_A, R_X) < d(R_B,R_X) as A and B represent the same triphone
   - Can be computed within-speakers (A,B and X are pronunced by the same speaker), across-speakers (A,B pronounced by the same speaker but X pronunced by a different speaker), on read-speech or spontaneous-speech.
    
2) Lexical evaluation, spot-the-word task:
   - The model receives two stimuli A=<em>**rabbit**</em> and B=<em>**raddit**</em> that form a pair of (word, nonword)
   - The probability associated to each stimuli is computed: P_A and P_B
   - The model is considered to be right if P_A > P_B 
    
Work in progress (Marvin). Give some examples here    

3) Syntactic evaluation, spot-the-grammatical-sentence task:
   - The model receives two stimuli A=<em>**The nice prince**</em> and B=<em>**The prince nice**</em> that form a pair of (grammatical sentence, ungrammatical sentence)
   - The probability associated to each stimuli is computed: P_A and P_B
   - The model is considered to be right if P_A > P_B 
    
Work in progress (Marvin). Give some examples here    
</details>

# Getting started

You'll probably want to start from there:

- [How to download the data? How to evaluate my own model?](docs/evaluation.md)

But if you want to go further:

- [How to evaluate models from the paper?](docs/evaluation.md)
- [How to retrain the models used in the paper?](docs/training.md)
- [How to prepare the training sets used in the paper?](docs/data.md)
- [How to recreate the lexical evaluation?](https://github.com/MarvinLvn/ChildDirectedLexicalTest)
- [How to recreate the syntactic evaluation?](https://github.com/MarvinLvn/ChildDirectedSyntacticTest)