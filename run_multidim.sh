#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    mkdir -p results/step/dim/hidden200
    mkdir -p results/ramp/dim/hidden200
    python3 TrainRNNfromconfig.py --config_file ./stepconfig.ini --baseDirectory ./results/step/dim/hidden200/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
    python3 TrainRNNfromconfig.py --config_file ./rampconfig.ini --baseDirectory ./results/ramp/dim/hidden200/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
elif [ "$1" = "train_local" ]; then
    mkdir -p testing/step/dim
    mkdir -p testing/ramp/dim
    python3 TrainRNNfromconfig.py --config_file ./stepconfig.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./rampconfig.ini --time 2 --epochNum 200  --baseDirectory ./testing/ramp/dim/ --baseSaveFileName fixedHiddenTrainingResults
fi