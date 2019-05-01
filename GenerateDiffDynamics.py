from GenerateTargetDynamics import *
import matplotlib.pyplot as plt
import pdb

def DefineOutputRampTarget(targetSlope,targetPeak,delayToInput,timePoints):
    targetSig = np.zeros((timePoints,1))
    targetTensor = torch.zeros(timePoints, 1, 1)

    rampsign=int(targetPeak>0)*2-1
    rampstart=rampsign*targetSlope


    PeakReachTime=int(targetPeak/rampstart)+delayToInput
    if PeakReachTime<timePoints:

        targetSig[(delayToInput+1):PeakReachTime+1]=np.expand_dims(np.linspace(rampstart,targetPeak,num=rampsign*int(targetPeak/targetSlope)),axis=-1)
        targetSig[(PeakReachTime+1):timePoints] = targetPeak
    else:
        targetSig[(delayToInput+1):]=np.expand_dims(np.linspace(rampstart,targetPeak,num=rampsign*int(targetPeak/targetSlope)),axis=-1)[:timePoints-delayToInput]
  
    targetTensor[:,:,0] = torch.from_numpy(targetSig)
    return targetSig, targetTensor


def GenerateOneDimensionalTarget(useVal, delayToInput, inputOnLength, timePoints,outputType,targetSlope=None):
    inputSig, inputTensor = DefineInputSignals(useVal, delayToInput, inputOnLength, timePoints)

    if outputType=='step':
        targetSig, targetTensor = DefineOutputTarget(useVal, delayToInput, timePoints)
    elif outputType=='ramp':
        targetSig, targetTensor = DefineOutputRampTarget(targetSlope, useVal, delayToInput, timePoints)

    return inputSig, targetSig, inputTensor, targetTensor

def TargetSingleSequence(useValVec, delayToInput, inputOnLength, timePoints,outputType,targetSlope=None):
    numDim = len(useValVec)
    inputTensorList = []
    targetTensorList = []

    for ii in range(numDim):
        inputSigCurrent, targetSigCurrent, inputTensorCurrent, targetTensorCurrent = \
            GenerateOneDimensionalTarget(useValVec[ii],delayToInput, inputOnLength,timePoints,outputType,targetSlope)
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

def TargetBatch(batchSize, numDim, delayToInput, inputOnLength, timePoints,outputType,targetSlope=None):
    useValArray = np.random.uniform(-1,1,(numDim, batchSize))
    # Start first element 
    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,outputType,targetSlope)
    # Continue:
    
    for ii in range(1, batchSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(useValArray[:,ii], delayToInput, inputOnLength, timePoints,outputType,targetSlope)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)
    return  inputTensor, targetTensor


def TargetMultiDimTestSet(testSetSize, dimNum, delayToInput, inputOnLength, timePoints,outputType,targetSlope=None):
    useValVec = np.linspace(-1,1, testSetSize)
    multiValArray = np.tile(useValVec,(dimNum,1))
    useValArray = np.asarray(np.meshgrid(*multiValArray))
    testSetSize = useValVec.shape[0] ** dimNum
    useValArray = np.reshape(useValArray,(dimNum, testSetSize))


    inputTensor, targetTensor = \
        TargetSingleSequence(useValArray[:,0], delayToInput, inputOnLength, timePoints,outputType,targetSlope)

    for ii in range(1, testSetSize):
        curInputTensor, curTargetTensor = \
            TargetSingleSequence(np.array(useValArray[:,ii]), delayToInput, inputOnLength, timePoints,outputType,targetSlope)
        inputTensor = torch.cat((inputTensor, curInputTensor), 0)
        targetTensor = torch.cat((targetTensor, curTargetTensor), 0)

    return  inputTensor, targetTensor

# delayToInput = 20
# inputOnLength = 50
# timePoints = 400

# TargetMultiDimTestSet(3,2,delayToInput,inputOnLength,timePoints,'ramp',0.01)

#targetSig,targetTensor=DefineOutputRampTarget(0.01,2,delayToInput,timePoints)
# TargetSingleSequence([2,1],delayToInput,inputOnLength,timePoints,'step',0.01)
# plt.plot(targetSig)
# plt.show()
