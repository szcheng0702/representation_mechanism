import subprocess
import pandas as pd

def combine_results(csvfile_precendent,num):
	df_list=[]
	for i in range(num):
		csvfilename=csvfile_precendent+'_'+str(i)+'_losses.csv'
		df_list.append(pd.read_csv(csvfilename))
	
	df_concat=pd.concat(df_list)

	print(df_concat)

	by_row_index = df_concat.groupby(df_concat.index)
	df_means = by_row_index.mean()
	df_errors=by_row_index.std()

	outputname=csvfile_precendent+'_combined.csv'
	errorsfilename=csvfile_precendent+'_stderror.csv'

	df_means.to_csv(outputname)
	df_errors.to_csv(errorsfilename)

def run(times_to_call,epochs,mode,precedent='tempTrainingResults',directory='results/'):
	config_file=mode+'config.ini'
	for i in range(times_to_call):
		subprocess.call(['python3', 'TrainRNNfromconfig.py',
			                         '--config_file',config_file,
			                         '--baseDirectory',directory,
			                         '--baseSaveFileName',precedent,
			                         '--time',str(i),
			                         '--epochNum',str(epochs)])

	combine_results(directory+precedent+'_'+str(epochs),10)
	# read_plot(precedent+'_combined.csv',precedent+'_stderror.csv')

if __name__=='__main__':
	run(2,100,'step')