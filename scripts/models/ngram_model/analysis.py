"""Analysing the results"""

from os import ctermid
import re
import pandas as pd
from collections import defaultdict
from ngram_lm import NGramLM
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

def unknown_ngrams_lexical_task(csv_file, models_folder):
    df = pd.read_csv(csv_file)
    uncorrect = list(df.loc[df["correct"] == 0]["phones"])
    correct = list(df.loc[df["correct"] == 1]["phones"])
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    results_df = []
    for parameters in models_folder.rglob("*.pkl"):
        hours = float(str(parameters.parents[0]).split("/")[-1][:-1])
        model = parameters.parents[1].stem
        training_set = parameters.stem
        lm = NGramLM()
        lm.load(parameters)
        for word in uncorrect:
            for ngram in lm.get_ngrams(word.split(" ")):
                if lm.ngram_size == 1 and ngram not in lm.parameters:
                    counts[model][hours][training_set][0] += 1
                if lm.ngram_size > 1:
                    *ctx, next_token = ngram
                    ctx = tuple(ctx)
                    if ctx not in lm.parameters:
                        counts[model][hours][training_set][0] += 1
                    elif next_token not in lm.parameters[ctx]:
                        counts[model][hours][training_set][0] += 1

        for word in correct:
            for ngram in lm.get_ngrams(word.split(" ")):
                if lm.ngram_size == 1 and ngram not in lm.parameters:
                    counts[model][hours][training_set][1] += 1
                if lm.ngram_size > 1:
                    *ctx, next_token = ngram
                    ctx = tuple(ctx)
                    if ctx not in lm.parameters:
                        counts[model][hours][training_set][1] += 1
                    elif next_token not in lm.parameters[ctx]:
                        counts[model][hours][training_set][1] += 1
    for model in counts:
        for hours in counts[model]:
            for training_set in counts[model][hours]:
                for correctness, count in counts[model][hours][training_set].items():
                    results_df.append({
                         "Model" : model,
                         "Hours" : hours,
                         "Training_set" : training_set,
                         "Correct" : correctness,
                         "Count" : count
                    })
    df = pd.DataFrame(results_df)
    grouped = df.groupby(["Model", "Hours", "Correct"])
    aggregated = grouped.agg({'Count': ['mean', 'std']})
    df = aggregated.xs('Count', axis=1)
    df = df.rename(columns={"mean":"Count"})
    return df




def create_df_by_hours_syntactic(results):
    all_results = []
    for results_type in results.rglob("*by_type.csv"):
        dataset = results_type.parents[0].stem
        model_size = results_type.parents[5].stem
        hour = str(results_type.parents[4])
        hour = hour.split("/")[-1]
        hour = float(hour[:-1])
        training_set = results_type.parents[3].stem
        df = pd.read_csv(results_type)
        df = df.drop(columns=["std"])
        df["Dataset"] = dataset
        df["Model"] = model_size
        df["Hour"] = hour
        df["Training_set"] = training_set
        df["score"] = 100.0 * df.score

        all_results.append(df)

    df = pd.concat(all_results, ignore_index=True)
    grouped = df.groupby(["Dataset", "Model", "Hour", "type"])
    aggregated = grouped.agg({'score': ['mean', 'std']})
    df = aggregated.xs('score', axis=1)
    df = df.rename(columns={"mean":"Score"})
    return df


def create_df_by_hours_lexical(results):
    all_results = []
    for results in results.rglob("*overall_accuracy_lexical*"):
        dataset = results.parents[0].stem
        model_size = results.parents[5].stem
        hour = str(results.parents[4])
        hour = hour.split("/")[-1]
        training_set = results.parents[3].stem
        with open(results, "r") as overall_acc:
            score = next(overall_acc).strip()
        all_results.append({
                            "Dataset" : dataset,
                            "Model": model_size,
                            "Hour": float(hour[:-1]),
                            "score": float(score) * 100.0,
                            "training_set": training_set
                            })
    df = pd.DataFrame(all_results)
    grouped = df.groupby(["Dataset", "Model", "Hour"])
    aggregated = grouped.agg({'score': ['mean', 'std']})
    df = aggregated.xs('score', axis=1)
    df = df.rename(columns={"mean":"Score"})
    return df

# gold_df = pd.read_csv("data/model_evaluation/syntactic/test/gold.csv")
# results_by_pair = pd.read_csv("results/unigrams/128h/00/scores/syntactic/test/score_syntactic_test_by_pair.csv")
# results_pairs = unigrams_syntactic_order(gold_df, results_by_pair)
# preprocessed_pairs = pairs(results_pairs)
# lm = UnigramLM()
# lm.load("trained_models/ngrams/unigrams/128h/00.pkl")
# print_score_diffs(lm, preprocessed_pairs)

results_path = Path("results")
# out = create_df_by_hours_syntactic(results_path)
# out.to_csv("plots/data/agg_syn.csv")

# out = create_df_by_hours_lexical(results_path)
# out.to_csv("plots/data/agg_lex.csv")

out = unknown_ngrams_lexical_task("data/model_evaluation/lexical/test/gold.csv",
                                    Path("trained_models/ngrams/"))
out.to_csv("plots/data/unknown_lex.csv")




