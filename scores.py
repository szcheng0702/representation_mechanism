import csv
import numpy as np
import pdb
import pandas as pd

def mse_arrays(predictions,targets):
	if predictions.shape[0]==0:
		score=0
	else:
		score=np.mean((predictions-targets)**2)

	return score

def determine_stable_slope(nparray,threshold):
	prev_diff=nparray[1,:]-nparray[0,:]
	stable_time=0

	for i in range(1,nparray.shape[0]-1):	
		curr_diff=nparray[i+1,:]-nparray[i,:]

		if abs(np.mean(curr_diff-prev_diff))<threshold:
			stable_time+=1
		else:
			stable_time=0
	
		if stable_time==15:
			start_time=i-15
			break

		prev_diff=curr_diff	

	# end_time=stable_time+start_time

	return start_time

def determine_peakReach(nparray,threshold): #2d array
	stable_time=0

	for i in range(1,nparray.shape[0]):	

		if abs(np.mean(nparray[i,:]-nparray[i-1,:]))<threshold:
			stable_time+=1
		else:
			stable_time=0
	
		if stable_time==10:
			start_time=i-10
			break

	return start_time

def our_metrics(predictions,targets):
	scores=0
	for i in range(predictions.shape[0]): #loop through batch
		prediction=predictions[i,:,:]
		target=targets[i,:,:]
		stable_slope_starttime=determine_stable_slope(prediction,1e-3)
		peakReachTime=determine_peakReach(prediction,1e-2)

		scores+=mse_arrays(prediction[stable_slope_starttime:peakReachTime,:],target[stable_slope_starttime:peakReachTime,:])*0.8\
						+mse_arrays(prediction[:stable_slope_starttime,:],target[:stable_slope_starttime,:])*0.1\
						+mse_arrays(prediction[peakReachTime:,:],target[peakReachTime:,:])*0.1


	avg_scores=scores/predictions.shape[0]

	return avg_scores

def calculate_scores(npzfilename):
	#load
	data=np.load(npzfilename)
	out=data['out']
	target=data['target']

	inputargfile=npzfilename.replace('_arrays.npz','_inputarg.csv')
	with open(inputargfile,'r') as f:
		reader=csv.reader(f,delimiter="\t")
		for i ,line in enumerate(reader):
			if i==0:
				delayToInput=int(line[1])
				break
	timePoints=target.shape[1]

	mse=our_metrics(out,target)

	return mse

def combine_scores_oneexperiment(csvfile_precendent,numTrials,dimLst,epochNum):
	thedict={}
	for dim in dimLst:
		scores_currdim=[]
		for i in range(numTrials):
			npzfilename=csvfile_precendent+'_numdim'+str(dim)+'_'+str(epochNum)+'_'+str(i)+'_arrays.npz'
			scores_currdim.append(calculate_scores(npzfilename))
		
		thedict[dim]=scores_currdim

	df=pd.DataFrame(thedict)
	pdb.set_trace()



dimLst=range(1,6)
combine_scores_oneexperiment('./results/ramp/dim/hidden250/fixedHiddenTrainingResults_hidden250',5,dimLst,500)





