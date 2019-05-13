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
    # else:
    #     pdb.set_trace()
    #     targetSig[(delayToInput+1):]=np.expand_dims(np.linspace(targetSlope+targetSig[delayToInput][0],rampend,num=int(rampend/targetSlope)),axis=-1)[:timePoints-delayToInput]
  
    targetTensor[:,:,0] = torch.from_numpy(targetSig)
    return targetSig, targetTensor


def GenerateOneDimensionalTarget(useVal, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    inputSig, inputTensor = DefineInputSignals(useVal, delayToInput, inputOnLength, timePoints)

    if outputType=='step':
        targetSig, targetTensor = DefineOutputTarget(useVal, delayToInput,timePoints,dt)
    elif outputType=='ramp':
        targetSig, targetTensor = DefineOutputRampTarget(useVal/10, rampPeak, delayToInput, timePoints,dt)


    return inputSig, targetSig, inputTensor, targetTensor

def TargetSingleSequence(useValVec, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    numDim = len(useValVec)
    inputTensorList = []
    targetTensorList = []

    for ii in range(numDim):
        inputSigCurrent, targetSigCurrent, inputTensorCurrent, targetTensorCurrent = \
            GenerateOneDimensionalTarget(useValVec[ii],delayToInput, inputOnLength,timePoints,dt,outputType,rampPeak)
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


def TargetBatch(batchSize, numDim, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    useValArray = np.random.uniform(-1,1,(numDim, batchSize))

    if outputType=='ramp':
        useValArray=get_validSlopeArray(useValArray,rampPeak,delayToInput,timePoints)

    # Start first element 
    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak)
    # Continue:
    
    for ii in range(1, batchSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)
    return  inputTensor, targetTensor

def TargetMultiDimTestSet(testSetSize, dimNum, delayToInput, inputOnLength, timePoints,dt,outputType,rampPeak=None):
    useValVec = np.linspace(-1,1, testSetSize)
    multiValArray = np.tile(useValVec,(dimNum,1))
    useValArray = np.asarray(np.meshgrid(*multiValArray))
    testSetSize = useValVec.shape[0] ** dimNum
    useValArray = np.reshape(useValArray,(dimNum, testSetSize))


    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,outputType,rampPeak)

    for ii in range(1, testSetSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,outputType,rampPeak)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)

    return  inputTensor, targetTensor

# delayToInput = 20
# inputOnLength = 50
# timePoints = 100
# # dt=0.1

# numDim=2
# batchSize=10
# useValArray = np.random.uniform(-1,1,(numDim, batchSize))
# get_validSlopeArray(useValArray,1,delayToInput,timePoints)