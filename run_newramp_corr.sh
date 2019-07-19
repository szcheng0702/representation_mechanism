#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    # mkdir -p results/step/dim/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p results/newramp/dim/perfecttest
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --baseDirectory ./results/step/dim/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
    python3 TrainRNNfromconfig.py --config_file ./corrnewramp.ini --baseDirectory ./results/newramp/dim/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/dim/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p testing/newramp/dim/perfecttest
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./corrnewramp.ini --time 2 --epochNum 200  --baseDirectory ./testing/newramp/dim/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrnewramp.ini --time 2 --epochNum 200 --baseDirectory ./testing/newramp/dim/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --baseDirectory ./results/step/dim/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrnewramp.ini --baseDirectory ./results/ramp_PRRandom/dim/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults
fi