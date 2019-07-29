import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
import shutil
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb
from config import Config#for input
import random

from RateUnitNetwork import RateUnitNetwork
import GenerateDiffDynamics

getMultiDimTestSet = GenerateDiffDynamics.TargetMultiDimTestSet
getRandomBatch=GenerateDiffDynamics.TargetBatch


def signal_arguments(config):
    delayToInput = random.choice(config.delayToInput)
    inputOnLength = random.choice(config.inputOnLength)
    timePoints = random.choice(config.timePoints)
    rampPeak=random.choice(config.rampPeak)

    return delayToInput,inputOnLength,timePoints,rampPeak

    
def LoadModel(saved_model_path):
    if saved_model_path:
        if os.path.isfile(saved_model_path):
            # args = argparse.Namespace()
            print("=> loading checkpoint '{}'".format(saved_model_path))
            checkpoint = torch.load(saved_model_path, map_location=lambda storage, loc: storage)
            
            hiddenUnitNum = checkpoint['hiddenUnitNum']
            inputSize = checkpoint['inputSize']
            outputSize = checkpoint['outputSize']
            dt = checkpoint['dt']
            numVariables = checkpoint['numVariables']
            noise_std = checkpoint['noise_std']
            # dropUnit = None
            # #dropUnit = checkpoint['dropUnit']

            network = RateUnitNetwork(
                inputSize, hiddenUnitNum, outputSize, dt, noise_std)
            network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model '{}'".format(saved_model_path))
            
    # self.network = network
    # print(network)
    return network

def RunMultiDimTestSet(network, config,args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(args.gpu_idx)
        print('Running on gpu idx: ')
        print(args.gpu_idx)

    device = torch.device('cuda' if args.cuda else 'cpu')

    network.eval() #add this line to make sure the network is in "evaluation" mode
    dimNum=args.outputSize
    perDimTestSetSize=args.perDimTestSetSize
    outputType=args.dynamics
    testSetSize=args.testSetSize
    delayToInput,inputOnLength,timePoints,rP=signal_arguments(config)



    # inputTensor, targetTensor = getMultiDimTestSet(perDimTestSetSize, dimNum, delayToInput, inputOnLength, timePoints,args.dt,outputType,rampPeak=None)
    inputTensor, targetTensor=getRandomBatch(testSetSize, dimNum, delayToInput, inputOnLength, timePoints,args.dt,outputType,rampPeak=rP)

    inputTensor = Variable(inputTensor).to(device)


    # testSetSize = perDimTestSetSize ** dimNum
    hiddenState = Variable(torch.randn(testSetSize, args.hiddenUnitNum)*1e-2).to(device)
    output = []
    with torch.no_grad():
        for i in range(0,timePoints):
            outputState, hiddenState = network(inputTensor[:,i,:], hiddenState)
            output.append(outputState)
        o = torch.cat(output,1)
        outputTensor = o.reshape(testSetSize,timePoints,args.outputSize)

    return outputTensor, inputTensor, targetTensor


