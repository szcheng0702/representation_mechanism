#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    # mkdir -p results/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p results/ramp_PRRandom/dim/hidden500epo5000_corr
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 2
    python3 TrainRNNfromconfig.py --config_file ./config/corrnewramp.ini --baseDirectory ./results/ramp_PRRandom/dim/hidden500epo5000_corr/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 2
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p testing/ramp_PRRandom/dim/hidden500epo5000_corr
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./config/corrnewramp.ini --epochNum 200  --baseDirectory ./testing/ramp_PRRandom/dim/hidden500epo5000_corr/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./config/corrnewramp.ini --epochNum 200 --baseDirectory ./testing/newramp/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./config/corrnewramp.ini --baseDirectory ./results/ramp_PRRandom/perfecttest/ --baseSaveFileName fixedHiddenTrainingResults
fi