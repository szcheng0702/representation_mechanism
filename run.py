from TrainRNNfromconfig import *
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import csv
from config import Config#for input
from AnalyzeNetwork import *
from scipy import stats

def run_multipletrials_samesetting(config,args,opt_str):
    targetTensors=[]
    outputTensors=[]
    lastLosses=[]

    for i in range(args.time):
        print('Trial {}\n'.format(i+1))
        targetTensor,outputTensor,lastloss=run_singletrial(config,args,i,opt_str)

        targetTensors.append(targetTensor)
        outputTensors.append(outputTensor)
        lastLosses.append(lastloss)

        # plot the last trial only
        if not args.plotAllTrial:
            if i==args.time-1:
                plotOutput([targetTensor],[outputTensor],args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(args.time)+'trials.png')

    df=pd.DataFrame({opt_str.lstrip('_'):lastLosses})

    return df

def run_multiple_samesetting_andtest(config,args,opt_str):
    targetTensors=[]
    outputTensors=[]
    lastLosses=[]

    dimslst=[]
    r_squareslst=[]
    corrlst=[]

    if 'corrNoise' in opt_str:
        var=args.corrNoise
    elif 'biasedCorrMultiplier' in opt_str:
        var=args.biasedCorrMultiplier

    for i in range(args.time):
        print('Trial {}\n'.format(i+1))
        targetTensor,outputTensor,lastloss=run_singletrial(config,args,i,opt_str)

        targetTensors.append(targetTensor)
        outputTensors.append(outputTensor)
        lastLosses.append(lastloss)

        # plot the last trial only
        if not args.plotAllTrial:
            if i==args.time-1:
                plotOutput([targetTensor],[outputTensor],args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(args.time)+'trials.png')
        else:
            plotOutput([targetTensor],[outputTensor],args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(i)+'trials.png')

        #test
        device = torch.device('cuda' if args.cuda else 'cpu')
        print(args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(args.epochNum)+'_'+str(i))
        network=LoadModel(args.baseDirectory+args.baseSaveFileName+opt_str+'_'+str(args.epochNum)+'_'+str(i)).to(device)
        out,inputTensor,target=RunMultiDimTestSet(network,config,args)
        plotOutput([target],[out],args.baseDirectory+'TEST_'+args.baseSaveFileName+'_'+str(i)+'.png')
        dims,r_square=test_calculate_rsquare(inputTensor, target, out,110,args.randomDim)


        dimslst+=dims
        r_squareslst+=r_square
        corrlst+=[var]*len(r_square)


    return dimslst,r_squareslst,corrlst



def run_multiple_diffcorrNoise(corrNoiseLst,config,args):
    dims_LST=[]
    r_squaresLST=[]
    corrNoiseLST=[]

    for i in range(len(corrNoiseLst)):
        args.corrNoise=corrNoiseLst[i]
        print('corrNoise: {}'.format(args.corrNoise))
        dims,r_squares,corrNoises=run_multiple_samesetting_andtest(config,args,'_corrNoise'+str(args.corrNoise))


        dims_LST+=dims
        r_squaresLST+=r_squares
        corrNoiseLST+=corrNoises


    df=pd.DataFrame({'dim':dims_LST,'r^2':r_squaresLST,'corrNoise':corrNoiseLST})
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_DiffCorrNoise.csv')



def run_multiple_diffbiasedCorrMult(biasedCorrMultLst,config,args):
    dims_LST=[]
    r_squaresLST=[]
    biasedCorrMultLST=[]

    for i in range(len(biasedCorrMultLst)):
        args.biasedCorrMultiplier=biasedCorrMultLst[i]
        print('biasedCorrMultiplier: {}'.format(args.biasedCorrMultiplier))
        dims,r_squares,biasedCorrMults=run_multiple_samesetting_andtest(config,args,'_biasedCorrMultiplier'+str(args.biasedCorrMultiplier))

        dims_LST+=dims
        r_squaresLST+=r_squares
        biasedCorrMultLST+=biasedCorrMults


    df=pd.DataFrame({'dim':dims_LST,'r^2':r_squaresLST,'biasedCorrMultiplier':biasedCorrMultLST})
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_DiffbiasedCorrMult.csv')


def plot_difftrials(csvfilename,dimlst,exp_type):
    df=pd.read_csv(csvfilename)
    for d in dimlst:
        plt.figure()
        df_curr=df.loc[df['dim'] == d].groupby(exp_type,as_index=False).mean()
        df_curr_err=df.loc[df['dim'] == d].groupby(exp_type,as_index=False).std()
        # df_curr.plot(x=exp_type,y='r^2',yerr=df_curr_err)
        plt.errorbar(x=df_curr[exp_type],y=df_curr['r^2'],yerr=df_curr_err['r^2'])
        plt.savefig(csvfilename.replace('.csv','dim'+str(d)+'.png'))
        df_curr.to_csv(csvfilename.replace('.csv','dim'+str(d)+'_mean.csv'))
        df_curr_err.to_csv(csvfilename.replace('.csv','dim'+str(d)+'_std.csv'))


def run_multiple_diffhiddenUnit(hiddenUnitLst,config,args):
    df_lst=[]
    for i in range(len(hiddenUnitLst)):
        args.hiddenUnitNum=hiddenUnitLst[i]
        print('Hidden Units: {}'.format(args.hiddenUnitNum))
        df_current=run_multipletrials_samesetting(config,args,'_hidden'+str(args.hiddenUnitNum))
        df_lst.append(df_current)

    df=pd.concat(df_lst,axis=1)
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_LastLossDiffHidden.csv')


def run_multiple_diffdim(inputdimLst,outputdimLst,config,args):
    df_lst=[]
    for i in range(len(dimLst)):
        args.inputSize=dinputimLst[i]
        args.outputSize=outputdimLst[i]
        print('num of Dim: {}'.format(args.outputSize))
        df_current=run_multipletrials_samesetting(config,args,'_numdim'+str(args.outputSize))
        df_lst.append(df_current)

    df=pd.concat(df_lst,axis=1)
    df.to_csv(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_LastLossDiffDim.csv',index=False)




if __name__=='__main__':
    config=Config()
    args = parser.parse_args()
    # biasedCorrMultLst=[0.9,0.8,0.4,0.2,0.1,0.05,0.02]
    corrNoiseLst=[0.2,0.1,0.08,0.07,0.06,0.05,0.02,0.01,0.008,0.005,0.002]
    dimlst=[1,3]
    # hiddenUnitLst=range(100,450,50)
    # dimLst=range(1,6)
    # run_multiple_diffdim(dimLst,config,args)
    if args.mode=='train':
        # run_multipletrials_samesetting(config,args,'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize))
        run_multiple_diffcorrNoise(corrNoiseLst,config,args)
        plot_difftrials(args.baseDirectory+args.baseSaveFileName+'_'+str(args.epochNum)+'_DiffCorrNoise.csv',dimlst,'corrNoise')
        # run_multiple_diffbiasedCorrMult(biasedCorrMultLst,config,args)
    # if args.mode=='test':
        # print(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(args.time-1))
        # network=LoadModel(args.baseDirectory+args.baseSaveFileName+'_hidden'+str(args.hiddenUnitNum)+'_numdim'+str(args.outputSize)+'_'+str(args.epochNum)+'_'+str(args.time-1))
        # out,inputTensor,target=RunMultiDimTestSet(network,config,args)
        # plotOutput([target],[out],args.baseDirectory+'TEST_'+args.baseSaveFileName+'_'+str(args.time)+'.png')
        # plot3dCorr(inputTensor, target, out,110,args.randomDim,args.baseDirectory+'TEST_'+args.baseSaveFileName+'_'+str(args.time)+'.png')