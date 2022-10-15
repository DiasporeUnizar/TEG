"""
@Author: Simona Bernardi
@Date: updated 15/10/2022

Datasets source: https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm
Selected: Spain - Madrid - Pollutant PM2.5
- Air pollution (in microg/m3), every hour, registered by a station located in Madrid
durig the 2020 (366 days)
- The training set is over 42 weeks and the two testing sets are over 10 weeks
- Anomalous testing set is synthetically generated from the original testing set.

The script:
- Plots the original time-series
- Generate a training set and two testing sets. 
The second testing set is generated synthetically from the first one (original time-series)
by multiplying the original value with a random value from a uniform distribution.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DS_PATH = "/dataset/pollution/"

DS = ["ES_6001_11850_2020_timeseries.csv", 
		#"ES_6001_11963_2020_timeseries.csv",
		#"ES_6001_12315_2020_timeseries.csv",
		#"ES_6001_12435_2020_timeseries.csv",
		#"ES_6001_15139_2020_timeseries.csv", 
		#"ES_6001_61369_2020_timeseries.csv",
		]
PERC = 80

def preprocess(cwd,file):
	ds_path = cwd + DS_PATH + file
	ds = pd.read_csv(ds_path)
	ds = ds[['DatetimeBegin','Concentration']]
	ds_padded = ds.fillna(method="pad")
	return ds_padded

def synthetize_noise(ds,a,b):
	original_obs = ds.to_numpy()
	noise =  np.random.uniform(a,b,original_obs.shape[0]) 
	original_obs[:,1] = np.multiply(original_obs[:,1],noise)
	synthetized = pd.DataFrame(original_obs, columns = ['DatetimeBegin','Concentration'])
	return synthetized


def generate_sets(ds,perc,fich,cwd):
	total_obs = len(ds.index)
	train_obs = int(total_obs * perc / 100)
	prefix = cwd + DS_PATH  + fich[:fich.find("_timeseries.csv")]
	train = ds.iloc[:train_obs,:]
	test = ds.iloc[train_obs+1:,:]
	test_synthetic = synthetize_noise(test,0.1,2.0)
	train.to_csv(prefix+ "_training.csv",index=False)
	test.to_csv(prefix+"_test_normal.csv",index=False)
	test_synthetic.to_csv(prefix+"_test_anomalous.csv",index=False)


if __name__ == '__main__':


	cwd = os.getcwd()
	for fich in DS:

		ds_padded = preprocess(cwd,fich)
		print("File:", fich)
		print(ds_padded.describe())
		x = range(ds_padded['Concentration'].size) 
		plt.plot(x,ds_padded['Concentration'],label=fich)		

		generate_sets(ds_padded,PERC,fich,cwd)

	#Plot time-series
	plt.xlabel('Time (every hour)')
	plt.ylabel('AirPollutant PM2.5 (microg/m3)')
	plt.legend()
	plt.show()



