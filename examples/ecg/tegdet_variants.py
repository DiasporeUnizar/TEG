"""
@Author: Simona Bernardi
@Date: updated 12/10/2022

Input dataset:
- ecg0606.csv: index column and colums headings are added from  "ecg0606_1.csv" 
--> Source: "ecg0606_1.csv" (https://github.com/GrammarViz2/grammarviz2_src)
--> Original source: ECG recordings "sele0606" (https://physionet.org/content/qtdb/1.0.0/)

The script:
- Builds a prediction model on the training set for the set of dissimilarity metric (TEG-detectors variants)
- Makes predictions on the testing set 
- Compute the confusion matrix and performance metrics 

Output:
- Stores them in RESULTS_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.teg import TEGDetector

#Input dataset/output results paths
TS_PATH = "/dataset/ecg/ecg0606.csv"
RESULTS_PATH = "/script_results/ecg/tegdet_variants_results.csv"


#List of metrics (detector variants)
list_of_metrics = ("Hamming", "Cosine", "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                "Clark", "Additivesymmetric" )

def build_and_predict(metric,n_bins,n_obs_per_period,alpha):
       
    #Create a new tegdet
    tegd = TEGDetector(metric,n_bins,n_obs_per_period,alpha)
    
    #Load time series
    cwd = os.getcwd() 
    ts_path = cwd + TS_PATH
    ts = tegd.get_dataset(ts_path)

    #Partition the time series in training and testing sets
    test = ts[:600]
    train = ts[600:]
    
    #Build model
    model, time2build = tegd.build_model(train)

    #Make prediction
    outliers, n_periods, time2predict = tegd.predict(test, model)

    #Set ground true values 
    ground_true = np.zeros(n_periods)
    ground_true[n_periods-1] = 1

    #Compute confusion matrix
    cm = tegd.compute_confusion_matrix(ground_true, outliers)

    #Collect detector configuration
    detector = {'metric': metric, 'n_bins': n_bins, 'n_obs_per_period':n_obs_per_period, 'alpha': alpha}

    #Collect performance metrics in a dictionary
    perf = {'tmc': time2build, 'tmp': time2predict}

    #Store basic metrics
    results_path = cwd + RESULTS_PATH
    tegd.metrics_to_csv(detector, "testing", perf, cm, results_path)
        

if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric,30,170,5)
