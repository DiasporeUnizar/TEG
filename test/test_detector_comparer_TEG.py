"""
@Author: Simona Bernardi
@Date: updated 13/09/2024

Input dataset:
- energy consumption (in KhW), every half-an-hour, registered by a smartmeter.
- the training set is over 60 weeks
- the testing sets are over 15 weeks: a testing set is either normal or anomalous testings
- The training set and testing sets are stored in separate files in the "/dataset/" folder.

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

from tegdet.teg import TEGDetector


#Input datasets/output results Paths
TRAINING_DS_PATH = "/dataset/energy/training.csv"
TEST_DS_PATH = "/dataset/energy/test_"
RESULTS_PATH = "/script_results/energy/test_detector_comparer_TEG_results.csv"
REFERENCE_PATH = "/script_results/energy/reference_results.csv"

#List of testing
list_of_testing = ("normal", "anomalous")

#List of metrics (detector variants)
list_of_metrics = [ "Cosine",  "Jaccard", "Hamming", "KL", "Jeffreys", "JS", "Euclidean", 
                    "Cityblock", "Chebyshev", "Minkowski", "Braycurtis", "Kulczynski", 
                    "Canberra", "Bhattacharyya", "Squared", "Divergence", "Additivesymmetric"
                    ]

#Parameters of TEG detectors: default values
n_bins = 30
n_obs_per_period=336
alpha=5


def test_generate_results():

    cwd = os.getcwd() 

    train_ds_path = cwd + TRAINING_DS_PATH

    for metric in list_of_metrics:

        teg = TEGDetector(metric)

        #Load training dataset
        train_ds = teg.get_dataset(train_ds_path)

        assert not train_ds.empty, "The training dataset is empty."

        
        #Build model
        model, time2build, time2graphs, time2global, time2metrics = teg.build_model(train_ds)

        for testing in list_of_testing:

            #Path of the testing
            test_ds_path = cwd + TEST_DS_PATH + testing + ".csv"
                
            #Load testing dataset
            test_ds = teg.get_dataset(test_ds_path)

            assert not test_ds.empty, "The testing dataset is empty."

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
            results_path = cwd + RESULTS_PATH
            teg.metrics_to_csv(detector, testing, perf, cm, results_path)

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
    cm = ['tp','tn','fp','fn']
    results = results[cm]
    reference = reference[cm]

    # Ensure both DataFrames have the same index and columns
    results = results.reset_index(drop=True).sort_index(axis=1)
    reference = reference.reset_index(drop=True).sort_index(axis=1)

    # Find differences
    try:
        diff = results.compare(reference)
        if not diff.empty:
            print("Differences found between results and reference:")
            print(diff)
        else:
            print("No differences found. Results match the reference.")
    except ValueError as e:
        print(f"Error comparing results: {e}")

    #Compare results with  the reference results
    assert results.equals(reference), "Results are not equal to the reference results."

if __name__ == '__main__':

    test_generate_results()

    test_results()


