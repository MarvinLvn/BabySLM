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
    echo "======================================================"
    echo "input folder: $input_path";
    echo "output folder: $output_path";
    echo "validation proportion: $val_prop";
    echo "test proportion: $test_prop";
    mkdir -p $output_path
    echo "================= MAKING THE TRAINING FILES ================"
    # for filename in $train_files/*.one_sentence_per_line; do # train_files*.one_sentence_per_line
    #     $kenlm_folder/build/bin/lmplz --discount_fallback -o $ngram_size < $filename > $out_dirname/${filename##*/}.arpa
    # done
