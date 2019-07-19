#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    # mkdir -p results/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p results/newramp/perfecttest
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
    python3 TrainRNNfromconfig.py --config_file ./corrnewramp.ini --baseDirectory ./results/newramp/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 3
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p testing/newramp/perfecttest
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./corrnewramp.ini --time 2 --epochNum 200  --baseDirectory ./testing/newramp/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrnewramp.ini --time 2 --epochNum 200 --baseDirectory ./testing/newramp/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrnewramp.ini --baseDirectory ./results/ramp_PRRandom/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults
fi