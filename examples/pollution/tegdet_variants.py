"""
@Author: Simona Bernardi
@Date: updated 15/10/2022

Input dataset:
- Air pollution (in microg/m3), every hour, registered by a station located in Madrid
durig the 2020 (366 days)
- The training set is over 42 weeks and the two testing sets are over 10 weeks
- Anomalous testing set is synthetically generated from the original testing set.
The script:
- Builds a prediction model for a subset of dissimilarity metric (TEG-detectors variants)
- Makes predictions with the two testing sets and
- Compute the confusion matrix and performance metrics 
Output:
- Prints the confusion matrix and metrics on stdout and stores them in RESULTS_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.teg import TEGDetector

#Input datasets/output results paths
TRAINING_DS_PATH = "/dataset/pollution/ES_6001_11850_2020_training.csv"
TEST_DS_PATH = "/dataset/pollution/ES_6001_11850_2020_test_"
RESULTS_PATH = "/script_results/pollution/tegdet_variants_results.csv"

#List of testing
list_of_testing = ["normal","anomalous"]

#List of metrics (detector variants)
list_of_metrics = ("Hamming", "Cosine", "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                    "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                    "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                    "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                    "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                    "Clark", "Additivesymmetric" )

#Parameters: default values
n_bins = 40
n_obs_per_period = 168
alpha = 5

def build_and_predict(metric,n_bins,n_obs_per_period,alpha):
    cwd = os.getcwd() 
    train_ds_path = cwd  + TRAINING_DS_PATH

    teg = TEGDetector(metric,n_bins,n_obs_per_period,alpha)
    #Load training dataset
    train_ds = teg.get_dataset(train_ds_path)
    #Build model
    model, time2build, time2graphs, time2global, time2metrics = teg.build_model(train_ds)

    for testing in list_of_testing:

        #Path of the testing
        test_ds_path = cwd  + TEST_DS_PATH + testing + ".csv"              
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
        detector = {'metric': metric, 'n_bins': n_bins, 'n_obs_per_period':n_obs_per_period, 'alpha': alpha}
        #Collect performance metrics in a dictionary
        perf = {'tmc': time2build, 'tmg': time2graphs, 'tmgl': time2global, 'tmm': time2metrics, 'tmp': time2predict}

        #Print and store basic metrics
        teg.print_metrics(detector, testing, perf, cm)
        results_path = cwd  + RESULTS_PATH
        teg.metrics_to_csv(detector, testing, perf, cm, results_path)
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric,n_bins,n_obs_per_period,alpha)
