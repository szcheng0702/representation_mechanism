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
import matplotlib.pyplot as plt
import RateUnitNetwork as RUN
import csv
import GenerateTargetDynamics
getStepBatch = GenerateTargetDynamics.StepTargetBatch

parser = argparse.ArgumentParser(description='Train future predictor')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochNum', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--baseSaveFileName', type=str, default='tempTrainingResults', metavar='N',
                    help='where to save result file')
parser.add_argument('--baseDirectory', type=str, default='.', metavar='N',
                    help='where to save result file')
parser.add_argument('--learningRate', type=float, default=0.00005, metavar='N',
                    help='optimizer learning rate')
parser.add_argument('--inputSize', type=int, default=1, metavar='N',
                    help='size of input dimension')
parser.add_argument('--outputSize', type=int, default=1, metavar='N',
                    help='size of output dimension')
parser.add_argument('--hiddenUnitNum', type=int, default=100, metavar='N',
                    help='number of hidden units')
parser.add_argument('--dt', type=float, default=0.1, metavar='N',
                    help='time constant')
parser.add_argument('--noise', type=float, default=None, metavar='N',
                    help='noise within network')
parser.add_argument('--numVariables', type=int, default=1, metavar='N',
                    help='number of variables encoded')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training')
parser.add_argument('--dropUnit', type=int, default=None, metavar='N',
                    help='set selected hidden unit to zero')
parser.add_argument('--gpu_idx', type=int, default=0, metavar='N',
                    help='set GPU index')
parser.add_argument('--save_every', type=int, default=1000, metavar='N',
                    help='set how often to save')
parser.add_argument('--noise_std', type=float, default=0, metavar='N',
                    help='set noise value')


def save_checkpoint(state, currentIter):
        file_name = args.baseDirectory+args.baseSaveFileName+str(currentIter)
        torch.save(state, file_name)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Code block:
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.set_device(args.gpu_idx)
    print('Running on gpu idx: ')
    print(args.gpu_idx)

device = torch.device('cuda' if args.cuda else 'cpu')
print('Running on: ')
print(device)

lastSavedIter = 0

if args.resume:

    # Load network
    savedNetworkName = args.baseSaveFileName
    baseSplit = args.baseSaveFileName.split('_')
    defaultBaseName = baseSplit[0]+"_"+baseSplit[1]+"_"+baseSplit[2]+"_e"
    lastSavedIter = int(baseSplit[3][1:])
    args.baseSaveFileName = defaultBaseName

    checkpoint = torch.load(savedNetworkName, map_location=lambda storage, loc: storage)
    args.hiddenUnitNum = checkpoint['hiddenUnitNum']
    args.inputSize = checkpoint['inputSize']
    args.outputSize = checkpoint['outputSize']
    args.dt = checkpoint['dt']
    args.numVariables = checkpoint['numVariables']
    args.noise = checkpoint['noise']
    #args.dropUnit = checkpoint['dropUnit']

    network = RUN.RateUnitNetwork(
        args.inputSize, args.hiddenUnitNum, args.outputSize, args.dt, args.numVariables, args.noise)
#        args.inputSize, args.hiddenUnitNum, args.outputSize, args.dt, args.numVariables, args.noise, args.dropUnit)
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model '{}'".format(args.baseSaveFileName))

else:
    print("Don't resume.")

    network = RUN.RateUnitNetwork(args.inputSize, args.hiddenUnitNum, \
        args.outputSize, args.dt, args.noise_std).to(device)

optimizer = optim.Adam(network.parameters(), args.learningRate)
criterion = nn.MSELoss()

if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])

delayToInput = 20
inputOnLength = 50
timePoints = 400


save_loss_every = 5
plot_every = math.floor(0.1*args.epochNum)

# Keep track of losses for plotting
current_loss = 0
test_loss = 0
all_losses = []
all_lossesX = []
test_losses = []
test_lossesX = []

start = time.time()
filename = args.baseSaveFileName
numDim = args.inputSize
lowestLoss = 1e6
current_avg_loss = lowestLoss

print('Training network')
for iter in range(lastSavedIter+1, args.epochNum + 1):
    inputTensor, targetTensor = getStepBatch(args.batch_size, numDim, delayToInput, inputOnLength, timePoints)

    inputTensor = Variable(inputTensor).to(device)
    #targetTensor = torch.transpose(targetTensor[:,:,0], 1, 0).to(device)
    optimizer.zero_grad()
    
    hiddenState = Variable(torch.randn(args.batch_size, args.hiddenUnitNum)*1e-2).to(device)
    output = []
    for i in range(0,timePoints):
        outputState, hiddenState = network(inputTensor[:,i,:], hiddenState)
        output.append(outputState)
    o = torch.cat(output,1)
    oo = o.reshape(args.batch_size,timePoints,args.outputSize)

    loss = criterion(oo, targetTensor)
    loss.backward()
    optimizer.step()
    #print(z.data.numpy())
    #print(loss.data.numpy())
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % save_loss_every == 0:
        current_avg_loss = current_loss.data.numpy()/save_loss_every
        pLoss = loss.cpu()
        all_losses.append(pLoss.data.numpy())
        all_lossesX.append(iter)
        current_loss = 0
        print('Iteration %d Time (%s) Average loss: %.4f ' % (iter, timeSince(start), current_avg_loss))

        with open(args.baseDirectory+args.baseSaveFileName+str(args.epochNum)+'_losses.csv', 'w') as lossFile:
            wr = csv.writer(lossFile, delimiter='\t')
            wr.writerows(zip(all_lossesX, all_losses))

    # Edit this to match
    if iter % args.save_every == 0:
        print('===> Storing results at epoch: %d Latest average loss: %.4f' % (iter, current_avg_loss))
        state = {
            'inputSize': args.inputSize,
            'outputSize': args.outputSize,
            'hiddenUnitNum': args.hiddenUnitNum,
            'dt': args.dt,
            'state_dict': network.state_dict(),
            'current_avg_loss': current_avg_loss,
            'optimizer': optimizer.state_dict(),
            'numVariables': args.numVariables,
            'noise_std': args.noise_std,
            'dropUnit': args.dropUnit
            }

        save_checkpoint(state, iter)

print('Done training network')

#print('Average training loss =', np.mean(all_losses))
#plotScatter(all_lossesX, all_losses)


    # Creates a square wave with an initial delay, signal length, and specified height
def DefinedTestExample(inputVal, delayToInput, inputOnLength, timePoints):    
    inputSig, inputTensor = DefineInputSignals(inputVal,delayToInput,inputOnLength,timePoints)
    targetSig, targetTensor = DefineOutputTarget(inputVal,delayToInput,timePoints)
    return inputSig, targetSig, inputTensor, targetTensor

def plotScatter(dataX, dataY):
    plt.figure()
    plt.scatter(dataX, dataY)
    
    plt.draw()
    plt.show(0.001)

# plotOutput takes tensors as an input and plots target and output
def plotOutput(targetTensors, outputTensors):

    numDims = len(targetTensors[0])

    colorList = ['b', 'g', 'r', 'k', 'c', 'm']
    plt.figure()

    for i in range(numDims):
        currentColor = colorList[i%len(colorList)]
        plt.plot(torch.cat(targetTensors, dim=1)[i].numpy(), currentColor+':')
        plt.plot(torch.cat(outputTensors, dim=1).data.numpy()[i], currentColor+'-')
    
    plt.draw()
    plt.pause(0.001)


def generateTestSet(targetError):
    plt.figure()
    plt.plot(lossesToPlot)
    
    plt.draw()
    plt.pause(0.001)

def LoadSavedNetwork():
    return
