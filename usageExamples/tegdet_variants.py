"""
@Author: Simona Bernardi
@Date: updated 30/03/2022

Input dataset:
- Energy consumption (in KhW), every half-an-hour, registered by a smart meter.
- The training set is over 60 weeks and the two testing sets are over 15 weeks
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
from tegdet.TEG import TEG

#Input datasets/output results paths
TRAINING_DS_PATH = "/dataset/training.csv"
TEST_DS_PATH = "/dataset/test_"
RESULTS_PATH = "/script_results/userScript_results.csv"

#List of scenarios
list_of_scenarios = ("normal", "anomalous")
#List of metrics (detector variants)
list_of_metrics = ["Hamming", "Cosine", "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                    "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                    "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                    "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                    "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                    "Clark", "Additivesymmetric" ]

def build_and_predict():
    cwd = os.getcwd() 
    train_ds_path = cwd + TRAINING_DS_PATH
    for metric in list_of_metrics:

        teg = TEG(metric)
        #Load training dataset
        train_ds = teg.get_dataset(train_ds_path)
        #Build model
        model, time2build = teg.build_model(train_ds)

        for scenario in list_of_scenarios:

            #Path of the scenario
            test_ds_path = cwd + TEST_DS_PATH + scenario + ".csv"               
            #Load testing dataset
            test_ds = teg.get_dataset(test_ds_path)
            #Make prediction
            outliers, obs, time2predict = teg.predict(test_ds, model)
            #Set ground true values
            if scenario == "anomalous":
                groundtrue = np.ones(obs)        
            else:
                groundtrue = np.zeros(obs)
            #Compute confusion matrix
            cm = teg.compute_confusion_matrix(groundtrue, outliers)
            #Collect performance metrics in a dictionary
            perf = {'tmc': time2build, 'tmp': time2predict}
            #Print and store basic metrics
            teg.print_metrics(metric, scenario, perf, cm)
            results_path = cwd + RESULTS_PATH
            teg.metrics_to_csv(metric, scenario, perf, cm, results_path)
        
if __name__ == '__main__':

    build_and_predict()
