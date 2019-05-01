import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import configargparse
import sys
import os
import time
import math
import matplotlib.pyplot as plt
import RateUnitNetwork as RUN
import csv
import pandas as pd
import GenerateDiffDynamics
from config import Config#for input
import random
getBatch = GenerateDiffDynamics.TargetBatch

import pdb

parser = configargparse.ArgumentParser(default_config_files=["config.ini"])
parser.add('-c', '--config_file', required=False, is_config_file=True, help='config file path')
parser.add('--time', type=int, default=0, metavar='N',
                    help='i-th time to call using same training settings (inputs arguments may vary)')
parser.add('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add('--epochNum', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add('--baseSaveFileName', type=str, default='tempTrainingResults', metavar='N',
                    help='where to save result file')
parser.add('--baseDirectory', type=str, default='.', metavar='N',
                    help='where to save result file')
parser.add('--learningRate', type=float, default=0.00005, metavar='N',
                    help='optimizer learning rate')
parser.add('--inputSize', type=int, default=1, metavar='N',
                    help='size of input dimension')
parser.add('--outputSize', type=int, default=1, metavar='N',
                    help='size of output dimension')
parser.add('--hiddenUnitNum', type=int, default=100, metavar='N',
                    help='number of hidden units')
parser.add('--dt', type=float, default=0.1, metavar='N',
                    help='time constant')
parser.add('--noise', type=float, default=0, metavar='N',
                    help='noise within network')
parser.add('--numVariables', type=int, default=1, metavar='N',
                    help='number of variables encoded')
parser.add('--resume', action='store_true', default=False,
                    help='resume training')
parser.add('--dropUnit', type=int, default=None, metavar='N',
                    help='set selected hidden unit to zero')
parser.add('--gpu_idx', type=int, default=0, metavar='N',
                    help='set GPU index')
parser.add('--save_every', type=int, default=1000, metavar='N',
                    help='set how often to save')
parser.add('--noise_std', type=float, default=0, metavar='N',
                    help='set noise value')
parser.add('--dynamics',type=str,default='step',metavar='D',
                    help='output dynamics type, choose from [step,ramp,sinusoidal]')


def save_checkpoint(args,state, currentIter):
    file_name = args.baseDirectory+args.baseSaveFileName+'_'+str(currentIter)+'_'+str(args.time)
    torch.save(state, file_name)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

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


# def generateTestSet(targetError):
#     plt.figure()
#     plt.plot(lossesToPlot)
    
#     plt.draw()
#     plt.pause(0.001)

# def LoadSavedNetwork(args):
#    # Load network


#     return

def signal_arguments(config):
    delayToInput = random.choice(config.delayToInput)
    inputOnLength = random.choice(config.inputOnLength)
    timePoints = random.choice(config.timePoints)
    targetSlope=random.choice(config.targetSlope)

    return delayToInput,inputOnLength,timePoints,targetSlope

def main():
    # Code block:
    config=Config()
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(args.gpu_idx)
        print('Running on gpu idx: ')
        print(args.gpu_idx)

    device = torch.device('cuda' if args.cuda else 'cpu')
    print('Running on: ')
    print(device)
    print(args.dynamics)

    lastSavedIter = 0

    if args.resume:
        savedNetworkName = args.baseSaveFileName
        baseSplit = args.baseSaveFileName.split('_')
        # defaultBaseName = baseSplit[0]+"_"+baseSplit[1]+"_"+baseSplit[2]+"_e"
        defaultBaseName=baseSplit[0]
        # # lastSavedIter = int(baseSplit[3][1:])
        args.baseSaveFileName = defaultBaseName

        checkpoint = torch.load(savedNetworkName, map_location=lambda storage, loc: storage)
        args.hiddenUnitNum = checkpoint['hiddenUnitNum']
        args.inputSize = checkpoint['inputSize']
        args.outputSize = checkpoint['outputSize']
        args.dt = checkpoint['dt']
        args.numVariables = checkpoint['numVariables']
        args.noise = checkpoint['noise']
        args.dropUnit = checkpoint['dropUnit']
        lastSavedIter=checkpoint['iter']
        delayToInput=checkpoint['delayToInput']
        inputOnLength=checkpoint['timePoints']
        timePoints=checkpoint['timePoints']
        targetSlope=checkpoint['targetSlope']

        network = RUN.RateUnitNetwork(
            args.inputSize, args.hiddenUnitNum, args.outputSize, args.dt, args.noise)
        network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model '{}'".format(args.baseSaveFileName))
        
    else:
        print("Don't resume.")

        network = RUN.RateUnitNetwork(args.inputSize, args.hiddenUnitNum, \
            args.outputSize, args.dt, args.noise_std).to(device)

        delayToInput,inputOnLength,timePoints,targetSlope=signal_arguments(config)

    optimizer = optim.Adam(network.parameters(), args.learningRate)
    criterion = nn.MSELoss()

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])


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

    args.save_every=min(args.save_every,args.epochNum)

    print('Training network')


    #write input information
    inputarg_name=['delayToInput','inputOnLength','timePoints','targetSlope']
    inputarg_value=[delayToInput,inputOnLength,timePoints,targetSlope]

    with open(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_'+str(args.time)+'_inputarg.csv', 'w') as lossFile:
        wr = csv.writer(lossFile, delimiter='\t')
        wr.writerows(zip(inputarg_name, inputarg_value))


    for iter in range(lastSavedIter+1, args.epochNum + 1):
        inputTensor, targetTensor = getBatch(args.batch_size, numDim, delayToInput, inputOnLength, timePoints,args.dynamics,targetSlope)
        inputTensor = Variable(inputTensor).to(device)
        # targetTensor = torch.transpose(targetTensor[:,:,0], 1, 0).to(device)
        
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

            # with open(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_'+str(args.time)+'_losses.csv', 'w') as lossFile:
            #     wr = csv.writer(lossFile, delimiter='\t')
            #     wr.writerows(zip(all_lossesX, all_losses))

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
                'noise':args.noise,
                'noise_std': args.noise_std,
                'dropUnit': args.dropUnit,
                'iter':iter,
                'delayToInput':delayToInput,
                'inputOnLength':inputOnLength,
                'timePoints':timePoints,
                'targetSlope':targetSlope
                }
            save_checkpoint(args,state, iter)

    print('Done training network')



    list_of_tuples=list(zip(all_lossesX, all_losses))

    df = pd.DataFrame(list_of_tuples, columns = ['Iteration', 'MSELoss']) 
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_'+str(args.time)+'_losses.csv')

    print('Average training loss =', np.mean(all_losses))
    # plotScatter(all_lossesX, all_losses)



if __name__=='__main__':
    main()

