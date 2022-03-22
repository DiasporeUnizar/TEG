"""
@Author: Simona Bernardi
@Date: updated 22/03/2022

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
- Confusion matrices and performance metrics are stored in "/script_results/detector_comparer_results.csv"
- Validation: results are compared with the ones obtained from the TEG version of the Diaspore repository
(assumed correct), which are stored in "/script_results/reference_results.csv"
"""

import os
import pandas as pd
import numpy as np
from tegdet.TEG import TEG

#List of scenarios
list_of_scenarios = ("Normal", "Anomalous")

#List of metrics (detector variants)
list_of_metrics = [ "Cosine", "Jaccard", "Hamming", "KL", "Jeffreys", "JS", "Euclidean", 
                    "Cityblock", "Chebyshev", "Minkowski", "Braycurtis", "Kulczynski", 
                    "Canberra", "Bhattacharyya", "Squared", "Divergence", "Additivesymmetric"]

#Parameters of TEG detectors
N_BINS = 30 
RESULTS_PATH = "/test/script_results/detector_comparer_TEG_results.csv"

def test_generate_results():

    cwd = os.getcwd() 

    training_ds = cwd + "/dataset/training_0_60.csv"

    for metric in list_of_metrics:

        detector = TEG(metric)

        #Load training dataset
        training_dataset = detector.get_dataset(training_ds)

        assert not training_dataset.empty, "The training dataset is empty."

        #Build model
        model, time_model_creation = detector.build_model(training_dataset)

        for scenario in list_of_scenarios:

            #Path of the scenario
            testing_ds = cwd + "/dataset/test_" + scenario + "_61_75.csv"
                
            #Load testing dataset
            testing_dataset = detector.get_dataset(testing_ds)

            assert not testing_dataset.empty, "The testing dataset is empty."

            #Make prediction
            predictions, obs, time_model_prediction = detector.predict(testing_dataset, model)

            #Set ground true vector
            if scenario == "Anomalous":
                groundtrue = np.ones(obs)        
            else:
                groundtrue = np.zeros(obs)

            #Compute confusion matrix
            cm = detector.compute_confusion_matrix(groundtrue,predictions)

            #Performance metrics
            perf = {'tmc': time_model_creation, 'tmp': time_model_prediction}

            #Print and store basic metrics
            detector.print_metrics(metric, scenario, perf, cm)
            results_path = cwd + RESULTS_PATH
            detector.metrics_to_csv(metric, scenario, perf, cm, results_path)

        assert os.path.exists(results_path), "Results file has not been created."
        assert os.path.getsize(results_path) > 0, "The result file is empty"

def test_results():

    #Get current working directory
    cwd = os.getcwd() 

    #Load results and reference results
    results_path = cwd + RESULTS_PATH
    reference_path = cwd + "/test/script_results/reference_results.csv"
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


