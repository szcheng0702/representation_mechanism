#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    mkdir -p results/step/dim/hidden300dim4_0.8biasedcorr_0.05noise
    mkdir -p results/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise
    cp ./0.05noisebiased_corrstep.ini results/step/dim/hidden300dim4_0.8biasedcorr_0.05noise
    cp ./0.05noisebiased_corrnewramp.ini results/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise
    python3 TrainRNNfromconfig.py --config_file ./0.05noisebiased_corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden --gpu_idx 3
    python3 TrainRNNfromconfig.py --config_file ./0.05noisebiased_corrnewramp.ini --baseDirectory ./results/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden --gpu_idx 3
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/dim/hidden300dim4_0.8biasedcorr_0.05noise
    mkdir -p testing/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise
    # python3 TrainRNNfromconfig.py --config_file ./0.05noisebiased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
    python3 TrainRNNfromconfig.py --config_file ./0.05noisebiased_corrnewramp.ini --time 2 --epochNum 200  --baseDirectory ./testing/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
elif [ "$1" = "test_local" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./0.05noisebiased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./0.05noisebiased_corrnewramp.ini --time 2 --epochNum 200 --baseDirectory ./testing/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
elif [ "$1" = "test" ]; then
	python3 TrainRNNfromconfig.py --mode 'test' --config_file ./0.05noisebiased_corrstep.ini --baseDirectory ./results/step/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./0.05noisebiased_corrnewramp.ini --baseDirectory ./results/newramp/dim/hidden300dim4_0.8biasedcorr_0.05noise/ --baseSaveFileName fixedHidden
fi