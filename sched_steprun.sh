#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

mkdir -p results/step/dim/hidden200batch200_epo8000_singleinputsched
python3 TrainRNNfromconfig.py --config_file ./schedstep.ini --baseDirectory ./results/step/dim/hidden200batch200_epo8000_singleinputsched/ --baseSaveFileName fixedHiddenTrainingResults
