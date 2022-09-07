#!/bin/bash

programname=$training_files_creator
function usage {
    echo "usage: $programname [-h] [-i input_path] [-o output_path] [-v val_prop] [-t test_prop]"
    echo "  -h  display help."
    echo "  -i  folder storing all training sets."
    echo "  -o  folder where all the created files will be stored."
    echo "  -v  proportion for the validation set."
    echo "  -t  proportion for the test set."
    exit 1
}

while getopts i:o:v:t: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
        o) output_path=${OPTARG};;
        v) val_prop=${OPTARG};;
        t) test_prop=${OPTARG};;
    esac
done
case $1 in
    -h) usage; shift ;;
esac
shift
echo "======================== ARGUMENTS ========================="
echo "input folder: $input_path";
echo "output folder: $output_path";
echo "validation proportion: $val_prop";
echo "test proportion: $test_prop";
mkdir -p $output_path

echo "================= MAKING THE TRAINING FILES ================"
training_limit=31
for hours in $input_path/*; do
    for training_set in $( ls $hours | sort -n ); do
        if [ "$training_set" -gt "$training_limit" ]; then
            continue
        fi
        out_training_path=$output_path/"$(basename "$hours")"/$training_set
        mkdir -p $out_training_path
        echo "Making $out_training_path"
        python scripts/text_lm/split_train_val_test_lm.py --input_path $hours/$training_set/phones/ --val_prop $val_prop --test_prop $test_prop --output_path $out_training_path --prefix ngrams
    done
done
