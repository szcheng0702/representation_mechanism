import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RateUnitNetwork(nn.Module):
    def __init__(self, inputSize, hiddenUnitNum, outputSize, dt, noise=0):
        super(RateUnitNetwork, self).__init__()

        self.hiddenUnitNum = hiddenUnitNum

        self.i2h = nn.Linear(inputSize, hiddenUnitNum)
        self.h2h = nn.Linear(hiddenUnitNum, hiddenUnitNum)
        self.h2o = nn.Linear(hiddenUnitNum, outputSize)
        self.tanh = nn.Tanh()
        self.dt = dt
        self.noise = noise

    def forward(self, input, hidden):
        recurrentInput = self.h2h(self.tanh(hidden))

        if self.noise==0:
            hidden = ((1-self.dt)*hidden + self.dt*(self.i2h(input)+recurrentInput))
        else:
            randomVal = torch.Tensor(hidden.size()).normal_(0,self.noise)
            hidden = ((1-self.dt)*hidden + self.dt*(self.i2h(input)+recurrentInput+randomVal))

        output = self.h2o(hidden)
        return output, hidden
