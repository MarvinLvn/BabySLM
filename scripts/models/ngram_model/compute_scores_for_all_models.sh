#!/bin/bash

programname=$trainer_on_all_files
function usage {
    echo "usage: $programname [-h] [-g gold] [-p predicted] [-k kind] [-t task]"
    echo "  -h  display help."
    echo "  -g  Path containing gold files."
    echo "  -p  Path containing files storing the predicted probabilities."
    echo "  -k  The type of evaluated dataset. Must be 'dev' or 'test'."
    echo "  -t  The task."
    exit 1
}

while getopts g:p:k:t: flag
do
    case "${flag}" in
        g) gold=${OPTARG};;
        p) predicted=${OPTARG};;
        k) kind=${OPTARG};;
        t) task=${OPTARG};;
    esac
done
case $1 in
    -h) usage; shift ;;
esac
shift
echo "======================== ARGUMENTS ========================="
echo "Path containing the gold csv files: $gold";
echo "Path containing the predicted probabilities: $predicted";
echo "The evaluation dataset is: $kind";
echo "The task: $task"; 

echo "================= COMPUTING THE SCORES ================"
for hours in $predicted/*; do
    for model in $hours/*; do
        python scripts/metrics/compute_$task.py -o $model/scores/$task/$kind -g $gold -p $model/model_evaluation -k $kind --is_text
    done
done