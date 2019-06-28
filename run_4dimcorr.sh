#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    mkdir -p results/step/dim/hidden300dim4_corr_noise
    mkdir -p results/ramp/dim/hidden300dim4_corr_noise
    python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 4
    python3 TrainRNNfromconfig.py --config_file ./corrramp.ini --baseDirectory ./results/ramp/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 4
elif [ "$1" = "train_local" ]; then
    mkdir -p testing/step/dim/hidden300dim4_corr_noise
    mkdir -p testing/ramp/dim/hidden300dim4_corr_noise
    python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./corrramp.ini --time 2 --epochNum 200  --baseDirectory ./testing/ramp/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrramp.ini --time 2 --epochNum 200 --baseDirectory ./testing/ramp/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrramp.ini --baseDirectory ./results/ramp/dim/hidden300dim4_corr_noise/ --baseSaveFileName fixedHiddenTrainingResults
fi