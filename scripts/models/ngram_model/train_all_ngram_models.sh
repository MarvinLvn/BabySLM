#!/bin/bash

programname=$trainer_on_all_files
function usage {
    echo "usage: $programname [-h] [-i train_files_folder] [-n ngram_size] [-o output_path]"
    echo "  -h  display help."
    echo "  -i  The folder storing all training files. Must be of the for hours/training_sets"
    echo "  -n  The size of the ngrams."
    echo "  -o  The folder where all the trained models will be stored."
    exit 1
}

while getopts i:n:o: flag
do
    case "${flag}" in
        i) train_files_folder=${OPTARG};;
        n) ngram_size=${OPTARG};;
        o) output_path=${OPTARG};;
    esac
done
case $1 in
    -h) usage; shift ;;
esac
shift
echo "======================== ARGUMENTS ========================="
echo "input folder: $train_files_folder";
echo "ngram size: $ngram_size";
echo "output folder: $output_path";
mkdir -p $output_path

echo "================= RUNNING THE TRAININGS ================"
for hours in $train_files_folder/*; do
    for training_set in $hours/*; do
        language_model_name="$(basename "$training_set")"
        language_model_folder=$output_path/"$(basename "$hours")"
        for text_file in $training_set/*; do
            echo "Training on $text_file ..."
            python scripts/models/ngram_model/train_ngram_lm.py --train_file $text_file --ngram_size $ngram_size --pad_utterances --out_directory $language_model_folder --out_filename $language_model_name
        done
    done
done
