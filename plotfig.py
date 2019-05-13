import matplotlib.pyplot as plt
import pandas as pd
import pdb

def combine_results_plot(csvfile_precendent):
	modes=['step','ramp']

	df_meanlst=[]
	df_stdlst=[]

	for mode in modes:
		csvfilename='results/'+mode+'/'+csvfile_precendent+'_LastLossDiffHidden.csv'
		df_curr=pd.read_csv(csvfilename)
		df_meanlst.append(df_curr.mean(axis=0))
		df_stdlst.append(df_curr.std(axis=0))


	df_mean=pd.concat(df_meanlst,axis=1)
	df_mean.columns=modes
	df_mean=df_mean.iloc[1:]
	df_mean.index = df_mean.index.map(lambda x: x.lstrip('hiddenUnit:'))

	
	df_stdev=pd.concat(df_stdlst,axis=1)
	df_stdev.columns=modes
	df_stdev=df_stdev.iloc[1:]
	df_stdev.index = df_stdev.index.map(lambda x: x.lstrip('hiddenUnit:'))


	figname='results/'+csvfile_precendent+'LastLoss.png'

	plt.figure()
	# plt.rcParams.update({'font.size': 14.5})
	df_mean.plot(yerr=df_stdev,capsize=2)
	plt.xlabel('Number of hidden units')
	plt.ylabel('MSE Loss at the end of Iteration 100')
	# plt.annotate('non-symbolic comparison training start', xy=(loss_compx,loss_compy), xytext=(loss_compx+100,loss_compy),
			# arrowprops=dict(facecolor='black', shrink=0.05),)
	# plt.annotate('non-symbolic addition training started', xy=(loss_addx,loss_addy), xytext=(loss_addx+50,loss_addy),
	# 		arrowprops=dict(facecolor='black', shrink=0.05),)
	plt.savefig(figname)


combine_results_plot('tempTrainingResults_100')
# /Users/sizhucheng/Desktop/representation_mechanism/results/ramp/tempTrainingResults_100_LastLossDiffHidden.csv