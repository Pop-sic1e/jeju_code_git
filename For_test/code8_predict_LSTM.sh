#!/bin/bash

#SBATCH --job-name=code8_prediction_LSTM       # job name
#SBATCH --output=./expected_outputs/code8_bash_result/result/%x_%j.out       # result log path
#SBATCH --error=./expected_outputs/code8_bash_result/err/%x_%j.err           # error path
#SBATCH --time=1:00:00                   # time limit
#SBATCH --cpus-per-task=24


# run code
python -u code8_predict_LSTM.py