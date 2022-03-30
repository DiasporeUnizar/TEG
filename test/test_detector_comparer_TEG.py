"""
@Author: Simona Bernardi
@Date: updated 30/03/2022

Input dataset:
- energy consumption (in KhW), every half-an-hour, registered by a smartmeter.
- the training set is over 60 weeks
- the testing sets are over 15 weeks: a testing set are either normal or anomalous scenarios
- The training set and testing set are stored in separate files in the "/dataset/" folder.

The test program:
- builds a prediction model based on the training set for each dissimilarity metric (TEG-detectors variants)
- makes predictions in testing sets and
- compute the confusion matrix and performance metrics  

Output:
- Confusion matrices and performance metrics are stored in RESULTS_PATH
- Validation: results are compared with the ones obtained from the TEG version of the Diaspore repository
(assumed correct), which are stored in REFERENCE_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.TEG import TEG

#Input datasets/output results Paths
TRAINING_DS_PATH = "/dataset/training.csv"
TEST_DS_PATH = "/dataset/test_"
RESULTS_PATH = "/test/script_results/detector_comparer_TEG_results.csv"
REFERENCE_PATH = "/test/script_results/reference_results.csv"

#List of scenarios
list_of_scenarios = ("normal", "anomalous")

#List of metrics (detector variants)
list_of_metrics = [ "Cosine", "Jaccard", "Hamming", "KL", "Jeffreys", "JS", "Euclidean", 
                    "Cityblock", "Chebyshev", "Minkowski", "Braycurtis", "Kulczynski", 
                    "Canberra", "Bhattacharyya", "Squared", "Divergence", "Additivesymmetric"]

#Parameters of TEG detectors: default values

def test_generate_results():

    cwd = os.getcwd() 

    train_ds_path = cwd + TRAINING_DS_PATH

    for metric in list_of_metrics:

        teg = TEG(metric)

        #Load training dataset
        train_ds = teg.get_dataset(train_ds_path)

        assert not train_ds.empty, "The training dataset is empty."

        #Build model
        model, time2build = teg.build_model(train_ds)

        for scenario in list_of_scenarios:

            #Path of the scenario
            test_ds_path = cwd + TEST_DS_PATH + scenario + ".csv"
                
            #Load testing dataset
            test_ds = teg.get_dataset(test_ds_path)

            assert not test_ds.empty, "The testing dataset is empty."

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

        assert os.path.exists(results_path), "Results file has not been created."
        assert os.path.getsize(results_path) > 0, "The result file is empty"

def test_results():

    #Get current working directory
    cwd = os.getcwd() 

    #Load results and reference results
    results_path = cwd + RESULTS_PATH
    reference_path = cwd + REFERENCE_PATH
    results = pd.read_csv(results_path)
    reference = pd.read_csv(reference_path)
    
    #Select confusion matrix entries
    cm = ['n_tp','n_tn','n_fp','n_fn']
    results = results[cm]
    reference = reference[cm]

    #Compare results with  the reference results
    assert results.equals(reference), "Results are not equal to the reference results."

if __name__ == '__main__':

    test_generate_results()

    test_results()


