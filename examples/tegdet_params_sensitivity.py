"""
@Author: Simona Bernardi
@Date: updated 02/05/2022

Input dataset:
- Energy consumption (in kWh), every half-an-hour, registered by a smart meter.
- The training set is over 60 weeks and the two testing sets are over 15 weeks
The script:
- Performs sensitivity analysis on the TEG-detector parameters:
-- n_obs_per_period: number of observation per period  
-- n_bins: number of discretization levels 
-- alpha: significance level 100-alpha 
- Builds the metric specific TEG-detector and makes predictions with the two testing sets
- Computes the confusion matrix and performance metrics 
Output:
- Stores the confusion matrix and metrics in RESULTS_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.teg import TEGDetector


#Input datasets/output results paths
TRAINING_DS_PATH = "/dataset/training.csv"
TEST_DS_PATH = "/dataset/test_"
RESULTS_PATH = "/script_results/tegdet_params_sensitivity_results.csv"

#List of testing
list_of_testing = ("normal", "anomalous")

#List of metrics (detector variants)
#Uncomment these lines to carry out an exhaustive analysis over all the available dissimilarity metrics
list_of_metrics =   ["Hamming", "Clark" #, "Canberra", "Lorentzian", "Kulczynski", "Divergence", "Cosine"
                     #"Jaccard", "Dice", "KL", "Jeffreys", "JS", "Euclidean", "Cityblock", 
                     #"Chebyshev", "Minkowski", "Braycurtis", "Gower", "Soergel", "Bhattacharyya", "Hellinger", 
                     #"Matusita", "Squaredchord", "Pearson", "Neyman", "Squared", "Probsymmetric", "Additivesymmetric" 
                    ]

#Parameters
list_of_n_obs_per_period = [24, 48, 96, 168, 192, 336, 480, 672, 816, 1008]
list_of_n_bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
list_of_alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def build_and_predict(metric, n_bins, n_obs_per_period, alpha):
    cwd = os.getcwd() 
    train_ds_path = cwd + TRAINING_DS_PATH

    teg = TEGDetector(metric, n_bins, n_obs_per_period, alpha)
    #Load training dataset
    train_ds = teg.get_dataset(train_ds_path)
    #Build model
    model, time2build = teg.build_model(train_ds)
    for testing in list_of_testing:
        #Path of the testing
        test_ds_path = cwd + TEST_DS_PATH + testing + ".csv"               
        #Load testing dataset
        test_ds = teg.get_dataset(test_ds_path)
        #Make prediction
        outliers, obs, time2predict = teg.predict(test_ds, model)
        #Set ground true values
        if testing == "anomalous":
            groundtrue = np.ones(obs)        
        else:
            groundtrue = np.zeros(obs)
        #Compute confusion matrix
        cm = teg.compute_confusion_matrix(groundtrue, outliers)
        #Collect detector configuration
        detector = {'metric': metric, 'n_bins': n_bins,'n_obs_per_period':n_obs_per_period,'alpha': alpha}
        #Collect performance metrics in a dictionary
        perf = {'tmc': time2build, 'tmp': time2predict}
        #Print and store basic metrics
        teg.print_metrics(detector, testing, perf, cm)
        results_path = cwd + RESULTS_PATH
        teg.metrics_to_csv(detector, testing, perf, cm, results_path)
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        for n_obs_per_period in list_of_n_obs_per_period:

            for n_bins in list_of_n_bins:

                for alpha in list_of_alpha:

                    build_and_predict(metric, n_bins, n_obs_per_period, alpha)
