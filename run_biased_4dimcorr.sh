#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

if [ "$1" = "train" ]; then
    # mkdir -p results/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    mkdir -p results/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    # cp ./biased_corrstep.ini results/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    cp ./biased_corrramp_PRRandom.ini results/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    # python3 TrainRNNfromconfig.py --config_file ./biased_corrstep.ini --baseDirectory ./results/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 0
    python3 TrainRNNfromconfig.py --config_file ./biased_corrramp_PRRandom.ini --baseDirectory ./results/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults --gpu_idx 0
elif [ "$1" = "train_local" ]; then
    # mkdir -p testing/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    mkdir -p testing/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise
    # python3 TrainRNNfromconfig.py --config_file ./biased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --config_file ./biased_corrramp_PRRandom.ini --time 2 --epochNum 200  --baseDirectory ./testing/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test_local" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrstep.ini --time 2 --epochNum 200 --baseDirectory ./testing/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrramp_PRRandom.ini --time 2 --epochNum 200 --baseDirectory ./testing/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
elif [ "$1" = "test" ]; then
	# python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrstep.ini --baseDirectory ./results/step/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
    python3 TrainRNNfromconfig.py --mode 'test' --config_file ./biased_corrramp_PRRandom.ini --baseDirectory ./results/ramp_PRRandom/dim/hidden500epo5000dim4_biasedcorr_0.1noise/ --baseSaveFileName fixedHiddenTrainingResults
fi