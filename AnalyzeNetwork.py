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

from RateUnitNetwork import RateUnitNetwork
import GenerateDiffDynamics
# getStepBatch = GenerateDiffDynamics.TargetBatch
# getStepTestSet = GenerateDiffDynamics.TargetOneDimTestSet
getMultiDimTestSet = GenerateDiffDynamics.TargetMultiDimTestSet

class AnalyzeNetwork():
    def __init__(self,args):
        self.args=args
        return
    
    def LoadModel(self, saved_model_path):
        if saved_model_path:
            if os.path.isfile(saved_model_path):
                self.args = argparse.Namespace()
                print("=> loading checkpoint '{}'".format(saved_model_path))
                checkpoint = torch.load(saved_model_path, map_location=lambda storage, loc: storage)
                
                self.args.hiddenUnitNum = checkpoint['hiddenUnitNum']
                self.args.inputSize = checkpoint['inputSize']
                self.args.outputSize = checkpoint['outputSize']
                self.args.dt = checkpoint['dt']
                self.args.numVariables = checkpoint['numVariables']
                self.args.noise_std = checkpoint['noise_std']
                self.args.dropUnit = None
                #self.args.dropUnit = checkpoint['dropUnit']

                network = RateUnitNetwork(
                    self.args.inputSize, self.args.hiddenUnitNum, self.args.outputSize, self.args.dt, self.args.noise_std)
                network.load_state_dict(checkpoint['state_dict'])
                print("=> loaded model '{}'".format(saved_model_path))
                
        self.network = network
        print(network)
        return

    def RunMultiDimTestSet(self, perDimTestSetSize, dimNum,outputType):
        # optimizer = optim.Adam(self.network.parameters(), 0.01) # HACK Should be replaced by no gradient block
        self.network.eval() #add this line to make sure the network is in "evaluation" mode
        delayToInput = 20
        inputOnLength = 50
        timePoints = 400
        device = 'cpu'

        inputTensor, targetTensor = getMultiDimTestSet(perDimTestSetSize, dimNum, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None)

        pdb.set_trace()
        inputTensor = Variable(inputTensor).to(device)
        print(targetTensor.shape)
        #targetTensor = torch.transpose(targetTensor[:,:,0], 1, 0).to(device)
        optimizer.zero_grad()
        
        testSetSize = perDimTestSetSize ** dimNum
        hiddenState = Variable(torch.randn(testSetSize, self.args.hiddenUnitNum)*1e-2).to(device)
        output = []
        for i in range(0,timePoints):
            outputState, hiddenState = self.network(inputTensor[:,i,:], hiddenState)
            output.append(outputState)
        o = torch.cat(output,1)
        outputTensor = o.reshape(testSetSize,timePoints,self.args.outputSize)

        return outputTensor, inputTensor, targetTensor

if __name__=='__main__':
    instance=AnalyzeNetwork()
    instance.LoadModel('testing/ramp/dim/hidden200_corr')
    instance.RunMultiDimTestSet(10,3,'step')

