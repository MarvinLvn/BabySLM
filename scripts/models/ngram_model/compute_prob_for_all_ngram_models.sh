#!/bin/bash

programname=$trainer_on_all_files
function usage {
    echo "usage: $programname [-h] [-i input_path] [-m models_path] [-t model_type] [-e mode] [-o output_path]"
    echo "  -h  display help."
    echo "  -i  Folder storing the gold files for test or dev."
    echo "  -m  Path where all the trained models are stored."
    echo "  -t  The type of ngram model, must be 'unigram' or 'ngram'."
    echo "  -e  Evaluate the models on 'test', 'dev' or 'both'."
    echo "  -o  Path to where the computed probabilities will be stored."
    exit 1
}

while getopts i:m:t:e:o: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
        m) models_path=${OPTARG};;
        t) model_type=${OPTARG};;
        e) mode=${OPTARG};;
        o) output_path=${OPTARG};;
    esac
done
case $1 in
    -h) usage; shift ;;
esac
shift
echo "======================== ARGUMENTS ========================="
echo "The path to the evaluation files: $input_path";
echo "Where the ngrams models are stored: $models_path";
echo "The type of the ngram models: $model_type";
echo "Evaluating them on: $mode";
echo "The outputs will be stored here: $output_path";

echo "================= COMPUTE THE PROBABILITIES ================"
for hours in $models_path/*; do
    for model in $hours/*.pkl; do
        model_name=$(basename -- "$model")
        model_stem="${model_name%.*}"
        output=$output_path/$(basename -- "$hours")/$model_stem
        echo $model
        python scripts/models/ngram_model/compute_prob_ngram_lm.py --input_path $input_path --output_path $output --model_path $model --model_type $model_type --mode $mode --text --remove_word_spaces
    done
done