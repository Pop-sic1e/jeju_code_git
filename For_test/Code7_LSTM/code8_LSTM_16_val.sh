#!/bin/bash

#SBATCH --job-name=code8_LSTM_16_val      # job name
#SBATCH --output=./result/%x_%j.out       # result log path
#SBATCH --error=./err/%x_%j.err           # error log path
#SBATCH --time=00:20:00                   # maximum execution time
#SBATCH --cpus-per-task=16                # number of CPU


# 파이썬 코드 실행
export EVAL_WORKERS=8
export THREADS_PER_WORKER=2
python -u code8_LSTM_16_val.py