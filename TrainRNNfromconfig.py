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
import pickle
from mpl_toolkits import mplot3d
from AnalyzeNetwork import *
from scipy import stats
getBatch = GenerateDiffDynamics.TargetBatch
getCorrelatedBatch=GenerateDiffDynamics.TargetCorrelatedBatch

import pdb

parser = configargparse.ArgumentParser(default_config_files=["config.ini"])
parser.add('-c', '--config_file', required=False, is_config_file=True, help='config file path')
parser.add('--time', type=int, default=1, metavar='N',
                    help='total number of time to call using same training settings')
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
parser.add('--randomDim', type=int, default=None, metavar='N',
                    help='number of dimensions which has no correalation. It is always smaller than args.inputSize')
parser.add('--corrMultiplier', type=float, default=None, metavar='N',
                    help='correalation multiplier. Used in correlation case only')
parser.add('--biasedCorrMultiplier', type=float, default=None, metavar='N',
                    help='Biased correalation multiplier. Current only valid for dimNum=4. Used in correlation case only')
parser.add('--corrNoise', type=float, default=0, metavar='N',
                    help='noise multiplier added when generating correlated dimension, so that the correlation dimension becauses\
                    corrMultiplier*prevDim+corrNoise*U(-1,1). Default:0.1')
parser.add('--testSetSize', type=int, default=100, metavar='N',
                    help='test set size')
parser.add('--hiddenUnitNum', type=int, default=100, metavar='N',
                    help='number of hidden units')
parser.add('--perDimTestSetSize', type=int, default=10, metavar='N',
                    help='perDimTestSetSize')
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
parser.add('--scheduler', action='store_true', default=False,
                    help='Use scheduler in training or not')
parser.add('--plotAllTrial', action='store_true', default=False,
                    help='Plot the diagram for all trials in the same hyperparameter settings. If false, plot the last trial only.')
parser.add('--gpu_idx', type=int, default=0, metavar='N',
                    help='set GPU index')
parser.add('--save_every', type=int, default=1000, metavar='N',
                    help='set how often to save')
parser.add('--noise_std', type=float, default=0, metavar='N',
                    help='set noise value')
parser.add('--dynamics',type=str,default='step',metavar='D',
                    help='output dynamics type, choose from [step,ramp,newramp,ramp_PRRandom,ramp_2Dinput,sine]')
parser.add('--mode',type=str,default='train',metavar='D',
                    help='task mode, choose from [train,test]')

def save_checkpoint(args,state, currentIter,ithrun):
    file_name = args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(currentIter)+'_'+str(ithrun)
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
def plotOutput(targetTensors, outputTensors,figfilename):
    if targetTensors[0].is_cuda:
        targetTensors=[targetTensor.cpu() for targetTensor in targetTensors]

    if outputTensors[0].is_cuda:
        outputTensors=[outputTensor.cpu() for outputTensor in outputTensors]

    batch_size = len(targetTensors[0])
    numDims=targetTensors[0].size(2)

    colorList = ['b', 'g', 'r', 'k', 'c', 'm']

    for j in range(numDims):
        target_currDim=[targetTensor[:,:,j] for targetTensor in targetTensors]
        output_currDim=[outputTensor[:,:,j] for outputTensor in outputTensors]

        plt.figure()
        for i in range(batch_size):
            currentColor = colorList[i%len(colorList)]
            plt.plot(torch.cat(target_currDim, dim=1)[i].numpy(), currentColor+':')
            plt.plot(torch.cat(output_currDim, dim=1).data.numpy()[i], currentColor+'-')
    
        plt.savefig(figfilename.replace('.png','dim'+str(j+1)+'.png'))

    # if numDims==2:
    #     target_numDim1=[targetTensor[:,:,0] for targetTensor in targetTensors]
    #     target_numDim2=[targetTensor[:,:,1] for targetTensor in targetTensors]
    #     timePoints=targetTensors[0].size(1)
    #     X=np.tile(np.arange(timePoints),(batch_size,1))
    #     pdb.set_trace()
    #     plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.plot(np.arange(timePoints),targetTensors[0][0,:,0].numpy(),targetTensors[0][0,:,1].numpy())

    #     ax.set_title('surface')
    #     plt.show()

    # plt.show()
    # plt.pause(3)
    # plt.close()


    # plt.savefig(figfilename)
    
    # plt.draw()
    # plt.pause(0.001)

def plot2ndVs1st3rd(targetTensors, outputTensors,figfilename):
    if targetTensors[0].is_cuda:
        targetTensors=[targetTensor.cpu() for targetTensor in targetTensors]

    if outputTensors[0].is_cuda:
        outputTensors=[outputTensor.cpu() for outputTensor in outputTensors]

    batch_size = len(targetTensors[0])
    numDims=targetTensors[0].size(2)

    colorList = ['b', 'g', 'r', 'k', 'c', 'm']

    target_3rdDim=[targetTensor[:,:,2] for targetTensor in targetTensors]
    target_1stDim=[targetTensor[:,:,0] for targetTensor in targetTensors]
    output_2ndDim=[outputTensor[:,:,1] for outputTensor in outputTensors]

    plt.figure()
    for i in range(batch_size):
        currentColor = colorList[i%len(colorList)]
        plt.plot(torch.cat(target_1stDim, dim=1)[i].numpy(), currentColor+':')
        plt.plot(torch.cat(output_2ndDim, dim=1).data.numpy()[i], currentColor+'-')

    plt.savefig(figfilename.replace('.png','2ndvs1stdim.png'))

    plt.figure()
    for i in range(batch_size):
        currentColor = colorList[i%len(colorList)]
        plt.plot(torch.cat(target_3rdDim, dim=1)[i].numpy(), currentColor+':')
        plt.plot(torch.cat(output_2ndDim, dim=1).data.numpy()[i], currentColor+'-')

    plt.savefig(figfilename.replace('.png','2ndvs3rddim.png'))

def plot3dCorr(inputTensor, targetTensor, outputTensor,timePoint2show,randomDim,figfilename):
    if targetTensor.is_cuda:
        targetTensor=targetTensor.cpu()

    if outputTensor.is_cuda:
        outputTensor=outputTensor.cpu()
        inputTensor=inputTensor.cpu()

    batch_size = targetTensor.size(0)
    numDims=targetTensor.size(2)


    target_randomDims=targetTensor[:,timePoint2show,numDims-randomDim:].numpy()
    target_corrDims=targetTensor[:,timePoint2show,:numDims-randomDim-1].numpy()
    input_randomDims=inputTensor[:,100,numDims-randomDim:].numpy()
    input_corrDims=inputTensor[:,100,:numDims-randomDim-1].numpy()
    output_queryDim=outputTensor[:,timePoint2show,numDims-randomDim-1].numpy()


    for i in range(target_corrDims.shape[1]):
        plt.figure()
        slope, intercept, r_value, p_value, std_err = stats.linregress(output_queryDim,input_corrDims[:,i])        
        line = slope*output_queryDim+intercept
        plt.plot(output_queryDim,input_corrDims[:,i],'o', output_queryDim, line)
        plt.scatter(output_queryDim,input_corrDims[:,i])
        plt.text(0.1, 0.1, 'R-squared = %0.2f' % r_value**2)
        plt.savefig(figfilename.replace('.png',str(numDims-randomDim)+'dimvs'+str(i+1)+'diminput.png'))


    for j in range(target_randomDims.shape[1]):
        plt.figure()
        slope, intercept, r_value, p_value, std_err = stats.linregress(output_queryDim,input_randomDims[:,j])
        line = slope*output_queryDim+intercept
        plt.plot(output_queryDim,input_randomDims[:,j],'o', output_queryDim, line)
        plt.scatter(output_queryDim,input_randomDims[:,j])
        plt.text(0.1, 0.1 , 'R-squared = %0.2f' % r_value**2)
        plt.savefig(figfilename.replace('.png',str(numDims-randomDim)+'dimvs'+str(numDims-randomDim+j+1)+'diminput.png'))



    # fig=plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(-1,1)
    # ax.set_zlim3d(-1,1)
    # ax.plot_trisurf(output_2ndDim,target_1stDim,target_3rdDim)
    # plt.savefig(figfilename.replace('.png','3D2ndvsOutputs.png'))

    # fig=plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(-1,1)
    # ax.set_zlim3d(-1,1)
    # ax.plot_trisurf(output_2ndDim,input_1stDim,input_3rdDim)
    # plt.savefig(figfilename.replace('.png','3D2ndvsInputs.png'))




def signal_arguments(config):
    delayToInput = random.choice(config.delayToInput)
    inputOnLength = random.choice(config.inputOnLength)
    timePoints = random.choice(config.timePoints)
    rampPeak=random.choice(config.rampPeak)

    return delayToInput,inputOnLength,timePoints,rampPeak

def run_singletrial(config,args,ithrun):

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
        defaultBaseName=baseSplit[0]
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
        inputOnLength=checkpoint['inputOnLength']
        timePoints=checkpoint['timePoints']
        rampPeak=checkpoint['rampPeak']

        network = RUN.RateUnitNetwork(
            args.inputSize, args.hiddenUnitNum, args.outputSize, args.dt, args.noise)
        network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model '{}'".format(args.baseSaveFileName))
        
    else:
        print("Don't resume.")

        network = RUN.RateUnitNetwork(args.inputSize, args.hiddenUnitNum, \
            args.outputSize, args.dt, args.noise_std).to(device)

    delayToInput,inputOnLength,timePoints,rampPeak=signal_arguments(config)

    optimizer = optim.Adam(network.parameters(), args.learningRate)
    criterion = nn.MSELoss()

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.scheduler:
        if args.resume:
            scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=100, verbose=True, threshold=5e-6, min_lr=1e-7, eps=1e-8)


    save_loss_every = 10
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
    numDim = args.outputSize
    lowestLoss = 1e6
    current_avg_loss = lowestLoss

    args.save_every=min(args.save_every,args.epochNum)

    print('Training network')
    network.train() #add this line to make sure the network is in "training" mode

    # if args.randomDim: 
    #     inputTensor, targetTensor =getCorrelatedBatch(args.batch_size, numDim, args.randomDim,delayToInput, inputOnLength, timePoints,config.dt,args.dynamics,args.corrMultiplier,args.corrNoise,rampPeak,args.biasedCorrMultiplier)
    # else:
    #     inputTensor, targetTensor = (args.batch_size, numDim, delayToInput, inputOnLength, timePoints,config.dt,args.dynamics,rampPeak)
    # inputTensor = Variable(inputTensor).to(device)
    # targetTensor=targetTensor.to(device)

    for iter in range(lastSavedIter+1, args.epochNum + 1):

        if args.randomDim: 
            inputTensor, targetTensor =getCorrelatedBatch(args.batch_size, numDim, args.randomDim,delayToInput, inputOnLength, timePoints,config.dt,args.dynamics,args.corrMultiplier,args.corrNoise,rampPeak,args.biasedCorrMultiplier)
        else:
            inputTensor, targetTensor = getBatch(args.batch_size, numDim, delayToInput, inputOnLength, timePoints,config.dt,args.dynamics,rampPeak)
        inputTensor = Variable(inputTensor).to(device)
        targetTensor=targetTensor.to(device)

        
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
        current_loss += loss

        if args.scheduler:
            scheduler.step(loss)

        # Print iter number, loss, name and guess
        if iter % save_loss_every == 0:
            current_avg_loss = current_loss.cpu().data.numpy()/save_loss_every
            pLoss = loss.cpu()
            all_losses.append(pLoss.data.numpy())
            all_lossesX.append(iter)
            current_loss = 0

            if iter % (5 * save_loss_every)==0:
                print('Iteration %d Time (%s) Average loss: %.6f ' % (iter, timeSince(start), current_avg_loss))

            # with open(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_'+str(ithrun)+'_losses.csv', 'w') as lossFile:
            #     wr = csv.writer(lossFile, delimiter='\t')
            #     wr.writerows(zip(all_lossesX, all_losses))

        # Edit this to match
        if iter % args.save_every == 0:
            print('===> Storing results at epoch: %d Latest average loss: %.6f' % (iter, current_avg_loss))
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
                'rampPeak':rampPeak
                }
            if args.scheduler:
                state['scheduler']=scheduler.state_dict()

            save_checkpoint(args,state, iter,ithrun)


    print('Done training network')
    # plotOutput([targetTensor],[oo])


    list_of_tuples=list(zip(all_lossesX, all_losses))

    df = pd.DataFrame(list_of_tuples, columns = ['Iteration', 'MSELoss']) 
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(ithrun)+'_losses.csv')

    print('Average training loss =', np.mean(all_losses))

    #write input information
    inputarg_name=['delayToInput','inputOnLength','timePoints','rampPeak']
    inputarg_value=[delayToInput,inputOnLength,timePoints,rampPeak]

    with open(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(ithrun)+'_inputarg.csv', 'w') as lossFile:
        wr = csv.writer(lossFile, delimiter='\t')
        wr.writerows(zip(inputarg_name, inputarg_value))

    if args.cuda:
        targetTensor=targetTensor.cpu()
        oo=oo.cpu()
        inputTensor=inputTensor.cpu()
        
    np.savez(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(ithrun)+'_arrays.npz',input=inputTensor.numpy(),target=targetTensor.numpy(),out=oo.detach().numpy())

    return targetTensor,oo,all_losses[-1]

def run_multipletrials_samesetting(config,args,option):
    targetTensors=[]
    outputTensors=[]
    lastLosses=[]

    if option=='hidden':
        opt_str='_hidden'+str(args.hiddenUnitNum)
    elif option=='dim':
        opt_str='_numdim'+str(args.outputSize)
    else:
        opt_str='_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)
 

    for i in range(args.time):
        print('Trial {}\n'.format(i+1))
        targetTensor,outputTensor,lastloss=run_singletrial(config,args,i)

        targetTensors.append(targetTensor)
        outputTensors.append(outputTensor)
        lastLosses.append(lastloss)

        # plot the last trial only
        if not args.plotAllTrial:
            if i==args.time-1:
                plotOutput([targetTensor],[outputTensor],args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(args.time)+'trials.png')

    df=pd.DataFrame({opt_str.lstrip('_'):lastLosses})


    return df

def run_multiple_diffhiddenUnit(hiddenUnitLst,config,args):
    df_lst=[]
    for i in range(len(hiddenUnitLst)):
        args.hiddenUnitNum=hiddenUnitLst[i]
        print('Hidden Units: {}'.format(args.hiddenUnitNum))
        df_current=run_multipletrials_samesetting(config,args,'hidden')
        df_lst.append(df_current)

    df=pd.concat(df_lst,axis=1)
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_LastLossDiffHidden.csv')


def run_multiple_diffdim(dimLst,config,args):
    df_lst=[]
    for i in range(len(dimLst)):
        args.outputSize=dimLst[i]
        args.outputSize=dimLst[i]
        print('num of Dim: {}'.format(args.outputSize))
        df_current=run_multipletrials_samesetting(config,args,'dim')
        df_lst.append(df_current)

    df=pd.concat(df_lst,axis=1)
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_LastLossDiffDim.csv',index=False)




if __name__=='__main__':
    config=Config()
    args = parser.parse_args()
    # hiddenUnitLst=range(100,450,50)
    # dimLst=range(1,6)
    # run_multiple_diffdim(dimLst,config,args)
    if args.mode=='train':
        run_multipletrials_samesetting(config,args,'')
    if args.mode=='test':
        print(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(args.time-1))
        network=LoadModel(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(args.time-1))
        # network=LoadModel(networkname)
        out,inputTensor,target=RunMultiDimTestSet(network,config,args)
        plotOutput([target],[out],args.baseDirectory+'TEST_'+args.baseSaveFileName+'_'+str(args.time)+'.png')
        plot3dCorr(inputTensor, target, out,110,args.randomDim,args.baseDirectory+'TEST_'+args.baseSaveFileName+'_'+str(args.time)+'.png')

