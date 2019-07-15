from GenerateTargetDynamics import *
import matplotlib.pyplot as plt
import pdb

def DefineOutputRampTarget(targetSlope,targetPeak,delayToInput,timePoints,dt):
    targetSig = np.zeros((timePoints,1))
    targetTensor = torch.zeros(timePoints, 1, 1)
    targetSig[:delayToInput+1]=ComputeDecay(targetPeak,dt,delayToInput)

    rampsign=int(targetSlope>0)*2-1
    rampend=rampsign*targetPeak


    PeakReachTime=int(rampend/targetSlope)+delayToInput
    # if PeakReachTime<timePoints: We have make sure before call this function that this is satisfied
    targetSig[(delayToInput+1):PeakReachTime+1]=np.expand_dims(np.linspace(targetSlope+targetSig[delayToInput][0],rampend,num=int(rampend/targetSlope)),axis=-1)
    targetSig[(PeakReachTime+1):timePoints] = rampend

  
    targetTensor[:,:,0] = torch.from_numpy(targetSig)
    return targetSig, targetTensor


def DefineOutputRampTarget_PeakRandom(PeakReachTime,targetPeak,delayToInput,timePoints,dt):
    targetSig = np.zeros((timePoints,1))
    targetTensor = torch.zeros(timePoints, 1, 1)
    targetSig[:delayToInput+1]=ComputeDecay(targetPeak,dt,delayToInput)

    targetSlope=targetPeak/(PeakReachTime-delayToInput)


    targetSig[(delayToInput+1):PeakReachTime+1]=np.expand_dims(np.linspace(targetSlope+targetSig[delayToInput][0],targetPeak,num=PeakReachTime-delayToInput),axis=-1)
    targetSig[(PeakReachTime+1):timePoints] = targetPeak

  
    targetTensor[:,:,0] = torch.from_numpy(targetSig)
    return targetSig, targetTensor


def GenerateOneDimensionalTarget(useVal, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None,PeakReachTime=None):
    inputSig, inputTensor = DefineInputSignals(useVal, delayToInput, inputOnLength,timePoints)

    if outputType=='step':
        targetSig, targetTensor = DefineOutputTarget(useVal, delayToInput,timePoints,dt)
    elif outputType=='ramp':
        targetSig, targetTensor = DefineOutputRampTarget(useVal/10, rampPeak, delayToInput, timePoints,dt)
    elif outputType=='newramp' or outputType=='ramp_PRRandom':
        targetSig,targetTensor = DefineOutputRampTarget_PeakRandom(PeakReachTime,useVal,delayToInput,timePoints,dt)

    return inputSig, targetSig, inputTensor, targetTensor

def TargetSingleSequence(useValVec, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None,PeakReachTime=None):
    numDim = len(useValVec)
    inputTensorList = []
    targetTensorList = []

    for ii in range(numDim):
        inputSigCurrent, targetSigCurrent, inputTensorCurrent, targetTensorCurrent = \
            GenerateOneDimensionalTarget(useValVec[ii],delayToInput, inputOnLength,timePoints,dt,outputType,rampPeak,PeakReachTime[ii])
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

def GenerateFactorCorrelated(numDim,randomDim,batchSize,correlation_multiplier,add_noise=0):
    firstdim=np.random.uniform(-1,1,(1, batchSize))

    final_array_lsts=[firstdim]
    for dimNum in range(1,numDim-randomDim):
        nextdim=firstdim*(correlation_multiplier**dimNum)
        if add_noise:
            nextdim+=add_noise*np.random.uniform(-1,1,(1, batchSize))
        final_array_lsts.append(nextdim)


    return np.concatenate(final_array_lsts,axis=0)

def Dim4BiasedFactorCorrelated(batchSize,biased_correlation_multiplier,add_noise=0):
    firstdim=np.random.uniform(-1,1,(1, batchSize))
    seconddim=np.random.uniform(-1,1,(1, batchSize))

    final_array_lsts=[firstdim,seconddim]
    nextdim=firstdim*(biased_correlation_multiplier)+seconddim*(1-biased_correlation_multiplier)
    if add_noise:
        nextdim+=add_noise*np.random.uniform(-1,1,(1, batchSize))
    final_array_lsts.append(nextdim)


    return np.concatenate(final_array_lsts,axis=0)


def get_validSlopeArray(targetSlopeArray,rampPeak,delayToInput,timePoints):

    PeakReachTime=abs((rampPeak/targetSlopeArray*10).astype(int))+delayToInput #+10: loose the bound

    need2change=PeakReachTime[(PeakReachTime+10)>timePoints]
    indx=(PeakReachTime+10>timePoints).nonzero()


    for i in range(need2change.shape[0]):
        reachtime=need2change[i]
        while (reachtime+10)>timePoints: #+10: loose the bound

            targetSlope=np.random.uniform(-1,1)
            reachtime=abs(int(rampPeak/targetSlope*10))+delayToInput

        targetSlopeArray[indx[0][i]][indx[1][i]]=targetSlope


    return targetSlopeArray

#optimize later
def get_validCorrelatedSlopeArray(targetSlopeArray,rampPeak,delayToInput,timePoints,randomDim,correlation_multiplier,add_noise_mult,biased=False):
    corrDim=targetSlopeArray.shape[0]

    PeakReachTime=abs((rampPeak/targetSlopeArray*10).astype(int))+delayToInput #+10: loose the bound


    need2change=PeakReachTime[(PeakReachTime+10)>timePoints]
    indx=(PeakReachTime+10>timePoints).nonzero()

    for i in range(need2change.shape[0]):
        reachtime=PeakReachTime[:,indx[1][i]]
        while np.count_nonzero(reachtime+10>timePoints)!=0:
            if biased:
                currArray=Dim4BiasedFactorCorrelated(1,correlation_multiplier,add_noise_mult)
            else:
                currArray=GenerateFactorCorrelated(corrDim+randomDim,randomDim,1,correlation_multiplier,add_noise_mult)
            reachtime=np.absolute((rampPeak/currArray*10).astype(int))+delayToInput

        targetSlopeArray[:,indx[1][i]]=np.reshape(currArray,-1)

    return targetSlopeArray



def TargetBatch(batchSize, numDim, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    PeakReachTime=np.full((numDim,batchSize),delayToInput+100)

    if outputType=='ramp_PRRandom':
        PeakReachTime=np.random.randint(delayToInput+10,timePoints-10,size=(numDim,batchSize))
    useValArray = np.random.uniform(-1,1,(numDim, batchSize))


    if outputType=='ramp':
        useValArray=get_validSlopeArray(useValArray,rampPeak,delayToInput,timePoints)

    # Start first element 
    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTime[:,0])
    # Continue:
    for ii in range(1, batchSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTime[:,ii])
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)



    return  inputTensor, targetTensor



def TargetCorrelatedBatch(batchSize, numDim, randomDim,delayToInput, inputOnLength, timePoints,dt,outputType,correlation_multiplier,add_noise_mult,rampPeak=None,biasedCorrMult=None):
    
    PeakReachTime=np.full((randomDim,batchSize),delayToInput+100)

    if outputType=='ramp_PRRandom':
        PeakReachTime=np.random.randint(delayToInput+10,timePoints-10,size=(randomDim,batchSize))

    random_useValArray = np.random.uniform(-1,1,(randomDim, batchSize))
    #correlated_ValArray
    if numDim==4 and biasedCorrMult:
        correlated_ValArray=Dim4BiasedFactorCorrelated(batchSize,biasedCorrMult,add_noise_mult)
    else:
        correlated_ValArray=GenerateFactorCorrelated(numDim,randomDim,batchSize,correlation_multiplier,add_noise_mult)


    if outputType=='ramp':
        random_useValArray=get_validSlopeArray(random_useValArray,rampPeak,delayToInput,timePoints)
        if numDim==4 and biasedCorrMult:
            correlated_ValArray=get_validCorrelatedSlopeArray(correlated_ValArray,rampPeak,delayToInput,timePoints,randomDim,biasedCorrMult,add_noise_mult,biased=True)
        else:
            correlated_ValArray=get_validCorrelatedSlopeArray(correlated_ValArray,rampPeak,delayToInput,timePoints,randomDim,correlation_multiplier,add_noise_mult)

    useValArray=np.concatenate((correlated_ValArray,random_useValArray),axis=0)
    PeakReachTimeArray=np.tile(PeakReachTime,(numDim,1))


    # Start first element 
    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTimeArray[:,0])
    # Continue:
    for ii in range(1, batchSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTimeArray[:,ii])
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)

    return  inputTensor, targetTensor



def TargetMultiDimTestSet(testSetSize, dimNum, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    useValVec = np.linspace(-1,1, testSetSize)
    multiValArray = np.tile(useValVec,(dimNum,1))
    useValArray = np.asarray(np.meshgrid(*multiValArray))
    testSetSize = useValVec.shape[0] ** dimNum
    useValArray = np.reshape(useValArray,(dimNum, testSetSize))

    PeakReachTimeArray=np.full((dimNum,testSetSize),delayToInput+100)

    if outputType=='ramp_PRRandom':
        PeakReachTime=np.linspace(delayToInput+10,timePoints-10,testSetSize).astype(int)
        multiValReachArray = np.tile(PeakReachTime,(dimNum,1))
        PeakReachTimeArray = np.asarray(np.meshgrid(*multiValReachArray))
        PeakReachTimeArray = np.reshape(PeakReachTimeArray,(dimNum, testSetSize))


    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTimeArray[:,0])

    for ii in range(1, testSetSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak,PeakReachTimeArray[:,ii])
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)


    return  inputTensor, targetTensor



# delayToInput = 20
# inputOnLength = 50
# timePoints = 200 #100
# dt=0.1

# numDim=3
# batchSize=10
# useValArray = np.random.uniform(-1,1,(numDim, batchSize))
# TargetCorrelatedBatch(batchSize, numDim, 1,delayToInput, inputOnLength, timePoints,dt,'ramp',0.2,rampPeak=1)
