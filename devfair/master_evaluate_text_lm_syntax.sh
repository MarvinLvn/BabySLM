#!/bin/bash
#SBATCH --partition=learnlab
#SBATCH --output=logs/eval/lstm_eval_test_syntax_%A_%a.out
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --array=1-318%318

source activate /private/home/marvinlvn/.conda/envs/provi
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p lstm_eval.txt)
./evaluate_text_lm_syntax.sh ${ARGS}
