import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import pickle

# def loadall(filename):
#     with open(filename, "rb") as f:
#         while True:
#             try:
#                 yield pickle.load(f)
#             except EOFError:
#                 break

def combine_results(csvfile_precendent,num):
	df_list=[]
	for i in range(num):
		csvfilename=csvfile_precendent+'_'+str(i)+'_losses.csv'
		df_list.append(pd.read_csv(csvfilename))
	
	df_concat=pd.concat(df_list)


	by_row_index = df_concat.groupby(df_concat.index)
	df_means = by_row_index.mean()
	df_errors=by_row_index.std()

	outputname=csvfile_precendent+'_combined.csv'
	errorsfilename=csvfile_precendent+'_stderror.csv'

	df_means.to_csv(outputname)
	df_errors.to_csv(errorsfilename)


def run(times_to_call,epochs,mode,precedent='tempTrainingResults',directory='results/'):
	config_file=mode+'config.ini'

	# for i in range(times_to_call):

	stdoutput=subprocess.call(['python36', 'TrainRNNfromconfig.py',
		                         '--config_file',config_file,
		                         '--dynamics', mode,
		                         '--baseDirectory',directory+mode+'/',
		                         '--baseSaveFileName',precedent,
		                         '--time',str(times_to_call),
		                         '--epochNum',str(epochs)])

	# combine_results(directory+mode+'/'+precedent+'_'+str(epochs),10)
	# plotOutput(targetTensors,outputTensors)
	# df=pd.DataFrame(lastLosses, columns = ['loss by the end of EPOCH'+str(epochs)]) 
	# df_means=df.mean()
	# df_errors=df.std()
	# print(df_means,df_errors)
	# read_plot(precedent+'_combined.csv',precedent+'_stderror.csv')

if __name__=='__main__':
	run(30,100,'ramp',directory='results/')