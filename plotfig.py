import matplotlib.pyplot as plt
import pandas as pd
import pdb
import numpy as np
import csv

def combine_results_plot(csvfile_precendent,option):
	# modes=['step','ramp']
	modes=['step']

	df_meanlst=[]
	df_stdlst=[]

	if option=='hidden':
		strip_str='hiddenUnit:'
		xtitle='Number of hidden units'
	elif option=='dim':
		strip_str='num of dim:'
		xtitle='Number of dimension'

	for mode in modes:
		# csvfilename='results/'+mode+'/'+csvfile_precendent+'_LastLossDiffHidden.csv'
		csvfilename='testing/'+mode+'/'+csvfile_precendent+'_LastLossDiff'+option+'.csv'
		df_curr=pd.read_csv(csvfilename)
		df_meanlst.append(df_curr.mean(axis=0))
		df_stdlst.append(df_curr.std(axis=0))


	df_mean=pd.concat(df_meanlst,axis=1)
	df_mean.columns=modes
	df_mean=df_mean.iloc[1:]
	df_mean.index = df_mean.index.map(lambda x: x.lstrip(strip_str))

	
	df_stdev=pd.concat(df_stdlst,axis=1)
	df_stdev.columns=modes
	df_stdev=df_stdev.iloc[1:]
	df_stdev.index = df_stdev.index.map(lambda x: x.lstrip(strip_str))


	# figname='results/'+csvfile_precendent+'LastLoss.png'
	figname='testing/'+csvfile_precendent+'LastLoss.png'

	plt.figure()
	# plt.rcParams.update({'font.size': 14.5})
	df_mean.plot(yerr=df_stdev,capsize=2)
	# plt.xlabel('Number of hidden units')
	# plt.ylabel('MSE Loss at the end of Iteration 100')
	plt.xlabel(xtitle)
	plt.ylabel('MSE Loss at the end of Iteration 100')
	plt.savefig(figname)

def combine_plot_crosstrials(csvfile_precendent,numTrials,dimLst,epochNum):
	for dim in dimLst:
		df_list=[]
		for i in range(numTrials):
			csvfilename=csvfile_precendent+'_numdim'+str(dim)+'_'+str(epochNum)+'_'+str(i)+'_losses.csv'
			df_list.append(pd.read_csv(csvfilename))
		
		df_concat=pd.concat(df_list)

		by_row_index = df_concat.groupby(df_concat.index)

		if dim==1:
			df_means=pd.DataFrame({'Iteration':by_row_index.mean()['Iteration']})
			df_errors=pd.DataFrame({'Iteration':by_row_index.mean()['Iteration']})
		df_means[dim]=by_row_index.mean()['MSELoss']
		df_errors[dim]=by_row_index.std()['MSELoss']

	df_means=df_means.set_index('Iteration')
	df_errors=df_errors.set_index('Iteration')
	outputname=csvfile_precendent+'_combined.csv'
	errorsfilename=csvfile_precendent+'_stderror.csv'

	df_means.to_csv(outputname,index=False)
	df_errors.to_csv(errorsfilename,index=False)


	plt.figure()
	df_means.plot(yerr=df_errors,capsize=2,errorevery=2)
	plt.savefig(csvfile_precendent+'dim.png')







dimLst=range(1,6)
combine_plot_crosstrials('./results/step/dim/hidden200_epo4000/fixedHiddenTrainingResults_hidden200',5,dimLst,4000)
# combine_results_plot('dim/tempTrainingResults_100','dim')
# /Users/sizhucheng/Desktop/representation_mechanism/results/ramp/tempTrainingResults_100_LastLossDiffHidden.csv