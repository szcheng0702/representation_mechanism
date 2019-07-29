#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    mkdir -p results/newramp/corrNoise/hidden500
    # python3 TrainRNNfromconfig.py --config_file ./stepconfig.ini --baseDirectory ./results/step/corrNoise/hidden500/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
    python3 TrainRNNfromconfig.py --config_file ./config/corrnewramp.ini --baseDirectory ./results/newramp/corrNoise/hidden500/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
elif [ "$1" = "train_local" ]; then
    mkdir -p testing/newramp/corrNoise/hidden500
    # python3 TrainRNNfromconfig.py --config_file ./stepconfig.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/corrNoise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./config/corrnewramp.ini --time 1 --epochNum 50  --baseDirectory ./testing/newramp/corrNoise/hidden500/ --baseSaveFileName fixedHiddenTrainingResults
fi