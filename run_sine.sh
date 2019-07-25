#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    # mkdir -p results/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p results/sine/dim/hidden500epo2000
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 2
    python3 TrainRNNfromconfig.py --config_file ./config/sineconfig.ini --baseDirectory ./results/sine/dim/hidden500epo2000/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 5
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/hidden400epo3000_corr_newramp_0.1noise
    mkdir -p testing/sine/dim/hidden500epo2000
    # python3 TrainRNNfromconfig.py --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./config/sineconfig.ini --epochNum 100  --baseDirectory ./testing/sine/dim/hidden500epo2000/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./config/sineconfig.ini --epochNum 100 --baseDirectory ./testing/sine/dim/hidden500epo2000/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./corrstep.ini --baseDirectory ./results/step/hidden400epo3000_corr_newramp_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./config/sineconfig.ini --baseDirectory ./results/sine/dim/hidden500epo2000/ --baseSaveFileName fixedHiddenTrainingResults
fi