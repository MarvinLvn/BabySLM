#!/bin/bash

programname=$trainer_on_all_files
function usage {
    echo "usage: $programname [-h] [-o output_path] [-g gold] [-p predicted] [-k kind]"
    echo "  -h  display help."
    echo "  -o  Path to where store the scores."
    echo "  -g  Path containing gold files."
    echo "  -p  Path containing files storing the predicted probabilities."
    echo "  -k  The type of evaluated dataset. Must be 'dev' or 'test'."
    exit 1
}

while getopts o:g:p:k: flag
do
    case "${flag}" in
        o) output_path=${OPTARG};;
        g) gold=${OPTARG};;
        p) predicted=${OPTARG};;
        k) kind=${OPTARG};;
    esac
done
case $1 in
    -h) usage; shift ;;
esac
shift
echo "======================== ARGUMENTS ========================="
echo "Path to where the scores will be stored: $output_path";
echo "Path containing the gold csv files: $gold";
echo "Path containing the predicted probabilities: $predicted";
echo "The evaluation dataset is: $kind";

echo "================= RUNNING THE TRAININGS ================"
for hours in $predicted/*; do
    for model in $hours/*; do
        python scripts/metrics/compute_syntactic.py -o $model/scores -g $gold -p $model/model_evaluation -k $kind --is_text
    done
done
