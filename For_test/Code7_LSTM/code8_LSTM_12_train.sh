#!/bin/bash

#SBATCH --job-name=code8_LSTM_12_train    # job name
#SBATCH --output=./result/%x_%j.out       # result log path
#SBATCH --error=./err/%x_%j.err           # error log path
#SBATCH --time=23:56:00                   # maximum execution time: 1 day
#SBATCH --gres=gpu:1                      # request 1 GPU
#SBATCH --cpus-per-task=8                 # request 8 CPU


# 파이썬 코드 실행
python -u code8_LSTM_12_train.py