import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import sys
import os
import time
import math
import csv
import pdb

def StepTargetMultiDimTestSet(testSetSize, dimNum, delayToInput, inputOnLength, timePoints):
    useValVec = np.linspace(-1,1, testSetSize)
    multiValArray = np.tile(useValVec,(dimNum,1))
    useValArray = np.asarray(np.meshgrid(*multiValArray))
    testSetSize = useValVec.shape[0] ** dimNum
    useValArray = np.reshape(useValArray,(dimNum, testSetSize))
    
    inputTensor, targetTensor = \
        StepTargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints)

    for ii in range(1, testSetSize):
        curInputTensor, curTargetTensor = \
            StepTargetSingleSequence(np.array(useValArray[:,ii]), delayToInput, inputOnLength, timePoints)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)

    print(inputTensor.shape)
    return  inputTensor, targetTensor

def StepTargetBatch(batchSize, numDim, delayToInput, inputOnLength, timePoints):
    useValArray = np.random.uniform(-1,1,(numDim, batchSize))
    # Start first element 
    inputTensor, targetTensor = \
        StepTargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints)
    # Continue:
    
    for ii in range(1, batchSize):
        curInputTensor, curTargetTensor = \
            StepTargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)
    return  inputTensor, targetTensor

def StepTargetSingleSequence(useValVec, delayToInput, inputOnLength, timePoints):
    numDim = len(useValVec)
    inputTensorList = []
    targetTensorList = []

    for ii in range(numDim):
        inputSigCurrent, targetSigCurrent, inputTensorCurrent, targetTensorCurrent = \
            GenerateOneDimensionalStepTarget(useValVec[ii],delayToInput, inputOnLength, timePoints)
        inputTensorList.append(inputTensorCurrent)
        targetTensorList.append(targetTensorCurrent)


    #Create multidimensional input, hidden, and target Tensors using cat
    inputTensor = inputTensorList[0]
    targetTensor = targetTensorList[0]

    for currentDim in range(1, numDim):
        inputTensor = torch.cat((inputTensor, inputTensorList[currentDim]), 2)
        targetTensor = torch.cat((targetTensor, targetTensorList[currentDim]), 2)
    inputTensor = torch.transpose(inputTensor, 0, 1)
    targetTensor = torch.transpose(targetTensor,0, 1)

    #print('Input tensor size:') #' %d' % (inputTensor.size()))
    #print(inputTensor.size())
    return  inputTensor, targetTensor

def DefineInputSignals(inputVal, delayToInput, inputOnLength, timePoints):
    inputSig = np.zeros(timePoints)
    inputSig[delayToInput:(delayToInput+inputOnLength)] = inputVal
    inputTensor = torch.zeros(timePoints, 1, 1)
    inputTensor[delayToInput:(delayToInput+inputOnLength),0,0] = inputVal
    return inputSig, inputTensor

def ComputeDecay(targetVal,dt,delayToInput):
    delayTensor=np.zeros((delayToInput+1,1))
    delayTensor[0]=targetVal

    for i in range(delayToInput):
        delayTensor[i+1]=delayTensor[i]*dt


    return delayTensor


# Creates a step of targetVal height after delay
def DefineOutputTarget(targetVal, delayToInput, timePoints,dt):
    targetSig = np.zeros((timePoints,1))
    targetSig[:delayToInput+1]=ComputeDecay(targetVal,dt,delayToInput)
    targetSig[(delayToInput+1):timePoints] = targetVal
    targetTensor = torch.zeros(timePoints, 1, 1)
    # targetTensor[(delayToInput):,0,0] = targetVal
    targetTensor[:,:,0] = torch.from_numpy(targetSig)
    return targetSig, targetTensor

def GenerateOneDimensionalStepTarget(useVal, delayToInput, inputOnLength, timePoints,dt):
    inputSig, inputTensor = DefineInputSignals(useVal, delayToInput, inputOnLength, timePoints)
    targetSig, targetTensor = DefineOutputTarget(useVal, delayToInput, timePoints)
    return inputSig, targetSig, inputTensor, targetTensor
