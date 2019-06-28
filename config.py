from argparse import Namespace
import torch
import pickle


class Config(Namespace):
	def __init__(self):
		self.dynamics='step'
		
		#alpha=0.1
		self.dt=0.1
		# self.delayToInput = [20,30,40]
		# self.inputOnLength = [50,60,70,80,90,100]
		# # self.timePoints = [400,500,600]
		# self.timePoints = [100,150,200]
		self.delayToInput = [30]
		self.inputOnLength = [80]
		self.timePoints = [400]
		self.rampPeak=[1]



		#30-50 time constants
		# 2 time constants the input
		# 3-5 for the delay

