#!/bin/bash

#SBATCH --job-name=code8_LSTM_16_train    # job name
#SBATCH --output=./result/%x_%j.out       # result log path
#SBATCH --error=./err/%x_%j.err           # error log path
#SBATCH --time=00:20:00                   # maximum execution time
#SBATCH --gres=gpu:1                      # request 1 GPU
#SBATCH --cpus-per-task=8                 # number of CPU


# 파이썬 코드 실행
python -u code8_LSTM_16_train.py