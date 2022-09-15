"""Analysing the results"""

from unittest import result
import pandas as pd
from ngram_lm import UnigramLM
from pathlib import Path

def unigrams_syntactic_order(gold_df, results_by_pair):
    tasks = ["adj_noun_order", "noun_verb_order"]
    tasks_order = gold_df.loc[((gold_df["subtype"].isin(tasks)) & (gold_df["correct"] == 0))]
    transcription_tasks_order = tasks_order["transcription"]
    lookup_transcription = list(transcription_tasks_order)
    tasks_order_results = results_by_pair.loc[((results_by_pair["non sentence"].isin(lookup_transcription)) & (results_by_pair["score"] != 0.5))]
    return tasks_order_results

def preprocessing(example) -> str:
    return example.replace(' <SEP> ', ' ')

def pairs(results_pairs) :
    for sentence, non_sentence in zip(results_pairs["sentence"], results_pairs["non sentence"]):
        sentence = preprocessing(sentence)
        non_sentence = preprocessing(non_sentence)
        yield sentence, non_sentence

def print_score_diffs(lm, preprocessed_pairs) :
    for sentence, non_sentence in preprocessed_pairs:
        sentence_score = lm.assign_logprob(sentence)
        non_sentence_score = lm.assign_logprob(non_sentence)
        if sentence_score != non_sentence_score:
            print("------------" * 7)
            print(f"sentence={sentence} | non sentence={non_sentence}")
            print(f"sentence score={sentence_score} | non sentence score ={non_sentence_score}")

def create_df_by_hours(results):
    all_df_results = []
    for results_type in results.rglob("*by_type.csv"):
        if "00" not in str(results_type):
            continue
        dataset = results_type.parents[0].stem
        model_size = results_type.parents[5].stem
        hour = str(results_type.parents[4])
        hour = hour.split("/")[-1]
        df = pd.read_csv(results_type)
        df["dataset"] = dataset
        df["model_size"] = model_size
        df["hour"] = hour
        df["hour_nb"] = float(hour[:-1])
        
        all_df_results.append(df)
    return pd.concat(all_df_results, ignore_index=True)


# gold_df = pd.read_csv("data/model_evaluation/syntactic/test/gold.csv")
# results_by_pair = pd.read_csv("results/unigrams/128h/00/scores/syntactic/test/score_syntactic_test_by_pair.csv")
# results_pairs = unigrams_syntactic_order(gold_df, results_by_pair)
# preprocessed_pairs = pairs(results_pairs)
# lm = UnigramLM()@
# lm.load("trained_models/ngrams/unigrams/128h/00.pkl")
# print_score_diffs(lm, preprocessed_pairs)

results_path = Path("results")
out = create_df_by_hours(results_path)
out.to_csv("plots/data/syntactic_by_type_dev_test_scores.csv")



