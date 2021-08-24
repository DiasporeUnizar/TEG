"""
@Author: Simona Bernardi
@Date: updated 24/08/2021

Simplified version of the detector_comparer in Diaspore repository to test the TEG library
The main program:
- builds a model based on the training set related to a given meterID
- makes prediction based on a set of *scenarios* and
- generates *basic metrics* to facilitate the comparison of the *TEG detectors* 
The training set and testing set (*scenario*) are stored in separate files 
in the "/dataset/" folder.
The results with all the basic metrics are generated in "/script_results/detector_comparer_results.csv"
- compare the results with the ones obtained from the TEG version of the Diaspore repository
(assumed correct).
"""

import os
import pandas as pd
from tegdet.TEG import TEG

# The meterIDs used are specific for each dataset
# You can customize the attacks and the detectors you want to use here
tuple_of_attacks = (False, "RSA_0.5_1.5", "RSA_0.25_1.1", "RSA_0.5_3", "Avg", "Min-Avg", "Swap", 
                    "FDI0", "FDI5", "FDI10", "FDI20", "FDI30")
list_of_metrics = [ "Cosine", "Jaccard", "Hamming", "KL", "Jeffreys", "JS", "Euclidean", 
                    "Cityblock", "Chebyshev", "Minkowski", "Braycurtis", "Kulczynski", 
                    "Canberra", "Bhattacharyya", "Squared", "Divergence", "Additivesymmetric"]

#Parameters of TEG detectors
N_BINS = 30 
RESULTS_PATH = "/test/script_results/detector_comparer_TEG_results.csv"

def test_generate_results():

    cwd = os.getcwd() 

    training_ds = cwd + "/test/dataset/1540_0_60.csv"

    for metric in list_of_metrics:

        detector = TEG(metric,N_BINS)

        #Load training dataset
        training_dataset = detector.get_training_dataset(training_ds)

        assert not training_dataset.empty, "The training dataset is empty."

        #Build model
        model, time_model_creation = detector.build_model(training_dataset)

        for attack in tuple_of_attacks:
            
            if attack:
                testing_ds = cwd + "/test/dataset/1540_" + attack + "_61_75.csv"
                
            else:
                testing_ds = cwd + "/test/dataset/1540_61_75.csv"

            #Load testing dataset
            testing_dataset = detector.get_testing_dataset(testing_ds)

            assert not testing_dataset.empty, "The testing dataset is empty."

            #Make prediction
            predictions, obs, time_model_prediction = detector.predict(testing_dataset, model)

            #Compute quality  metrics
            cm = detector.compute_confusion_matrix(obs, predictions, attack)

            #Performance metrics
            perf = {'tmc': time_model_creation, 'tmp': time_model_prediction}

            #Print and store basic metrics
            detector.print_metrics(metric, attack, perf, cm)
            results_path = cwd + RESULTS_PATH
            detector.metrics_to_csv(metric, attack, perf, cm, results_path)

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


