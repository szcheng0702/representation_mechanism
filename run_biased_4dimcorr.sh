#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    mkdir -p results/step/dim/hidden300dim4_biasedcorr_0.1noise
    mkdir -p results/newramp/dim/hidden300dim4_biasedcorr_0.1noise
    cp ./biased_corrstep.ini results/step/dim/hidden300dim4_biasedcorr_0.1noise
    cp ./biased_corrnewramp results/newramp/dim/hidden300dim4_biasedcorr_0.1noise
    python3 TrainRNNfromconfig.py --config_file ./biased_corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 4
    python3 TrainRNNfromconfig.py --config_file ./biased_corrnewramp.ini --baseDirectory ./results/newramp/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 4
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/dim/hidden300dim4_biasedcorr_0.1noise
    mkdir -p testing/newramp/dim/hidden300dim4_biasedcorr_0.1noise
    # python3 TrainRNNfromconfig.py --config_file ./biased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./biased_corrnewramp.ini --time 2 --epochNum 200  --baseDirectory ./testing/newramp/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrnewramp.ini --time 2 --epochNum 200 --baseDirectory ./testing/newramp/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrnewramp.ini --baseDirectory ./results/newramp/dim/hidden300dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
fi