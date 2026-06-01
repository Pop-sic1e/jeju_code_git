#!/bin/bash

#SBATCH --job-name=code8_prediction_LSTM       # job name
#SBATCH --output=./bash_result/result/%x_%j.out       # result log path
#SBATCH --error=./bash_result/err/%x_%j.err           # error path
#SBATCH --time=23:56:00                   # time limit
#SBATCH --cpus-per-task=16


# run code
python -u code8_predict_LSTM.py