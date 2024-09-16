"""
@Author: Simona Bernardi
@Date: updated 12/10/2022

Input dataset:
- ecg0606.csv 
--> index column and colums headings added from  "ecg0606_1.csv" 
(https://github.com/GrammarViz2/grammarviz2_src)
--> Original source: https://physionet.org/content/qtdb/1.0.0/ (ECG recordings: sele0606)
The script:
- Performs sensitivity analysis on the TEG-detector parameters:
-- n_bins: number of discretization levels 
-- alpha: significance level 100-alpha 
- Builds the metric specific TEG-detector and makes predictions with the testing set
- Computes the confusion matrix and performance metrics 
Output:
- Stores the confusion matrix and metrics in RESULTS_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.teg import TEGDetector


#Input datasets/output results paths
TS_PATH = "/dataset/ecg/ecg0606.csv"
RESULTS_PATH = "/script_results/ecg/tegdet_params_sensitivity_results.csv"


#List of metrics (detector variants)
list_of_metrics =   ["Hamming", "Clark" , "Canberra", "Lorentzian", "Kulczynski", "Divergence", 
                    "Cosine","Jaccard", "Dice", "KL", "Jeffreys", "JS", "Euclidean", "Cityblock", 
                    "Chebyshev",  "Minkowski", "Braycurtis", "Gower", "Soergel", "Bhattacharyya", "Hellinger", 
                    "Matusita", "Squaredchord", "Pearson", "Neyman", "Squared", "Probsymmetric", 
                    "Additivesymmetric" 
                    ]

#Parameters
list_of_n_obs_per_period = [170]
list_of_n_bins = [10, 20, 30, 40, 50]
list_of_alpha = [1, 2.5, 5, 7.5, 10]


def build_and_predict(metric, n_bins, n_obs_per_period, alpha):
    
    #Create a new tegdet
    teg = TEGDetector(metric, n_bins, n_obs_per_period, alpha)

    #Load time series
    cwd = os.getcwd() 
    ts_path = cwd + TS_PATH
    ts = teg.get_dataset(ts_path)

    #Partition the time series in training and testing sets
    test_ds = ts[:600]
    train_ds = ts[600:]

    #Build model
    model, time2build, time2graphs, time2global, time2metrics, mem2graphs = teg.build_model(train_ds)
    
    #Make prediction
    outliers, obs, time2predict = teg.predict(test_ds, model)
        
    #Set ground true values
    groundtrue = np.zeros(obs)
    groundtrue[obs-1] = 1

            
    #Compute confusion matrix
    cm = teg.compute_confusion_matrix(groundtrue, outliers)
    #Collect detector configuration
    detector = {'metric': metric, 'n_bins': n_bins,'n_obs_per_period':n_obs_per_period,'alpha': alpha}
    #Collect performance metrics in a dictionary
    perf = {'tmc': time2build, 'tmg': time2graphs, 'tmgl': time2global, 'tmm': time2metrics, 'tmp': time2predict, 'm2g': mem2graphs}
    #Print and store basic metrics
    teg.print_metrics(detector, "testing", perf, cm)
    results_path = cwd + RESULTS_PATH
    teg.metrics_to_csv(detector, "testing", perf, cm, results_path)
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        for n_bins in list_of_n_bins:

            for alpha in list_of_alpha:

                build_and_predict(metric, n_bins, 170, alpha)
