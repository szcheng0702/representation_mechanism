from argparse import Namespace
import torch
import pickle


class Config(Namespace):
	def __init__(self):
		self.dynamics='step'
		
		self.delayToInput = [20,30,40]
		self.inputOnLength = [50,60,70,80,90,100]
		self.timePoints = [400,500,600]

		self.targetSlope=[0.01,0.05,0.1]

