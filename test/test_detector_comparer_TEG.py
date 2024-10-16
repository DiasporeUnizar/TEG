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

        # Slide window method after the first build
        slide_window_scheme = teg.get_sw()

        for testing in list_of_testing:

            # We preserve the schema from the model built, for anomalous and normal datasets
            OriginalModel = model
            teg.update_mb(OriginalModel)

            # cm to store all cms generated during sliding window
            cm_accumulative = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

            #Path of the testing
            test_ds_path = cwd + TEST_DS_PATH + testing + ".csv"
                
            #Load testing dataset
            test_ds = teg.get_dataset(test_ds_path)

            assert not test_ds.empty, "The testing dataset is empty."

            # Full dataset with test and train
            full_ds = pd.concat([train_ds, test_ds], ignore_index=True)

            # Initialize the window using the Initial sheme given a test dataset (normal or anomalous)
            slide_window = slide_window_scheme
            slide_window.initialize_window(full_ds)

            #Make prediction only on the first week
            outliers, obs, time2predict = teg.predict(test_ds.head(n_obs_per_period), OriginalModel)

            #Set ground true values
            if testing == "anomalous":
                groundtrue = np.ones(obs)        
            else:
                groundtrue = np.zeros(obs)

            #Compute confusion matrix
            cm = teg.compute_confusion_matrix(groundtrue, outliers)

            # Accumulate the cms
            cm_accumulative['tp'] += cm['tp']
            cm_accumulative['tn'] += cm['tn']
            cm_accumulative['fp'] += cm['fp']
            cm_accumulative['fn'] += cm['fn']

            # Metrics to store the time during window processing
            time2window = 0

            # Compute the rest of the weeks on the testing data
            while True:

                window = slide_window.slide_window(full_ds) # Moves the window immediately after the original build process and the latest one

                if window is None:
                    #print(f"No more data available for sliding window in dataset {testing}.")
                    break

                time2window += teg.process_window(train_ds, n_bins + 2)

                # Make prediction on latest week
                model_w = teg.get_mb()
                outliers, obs, time2predict = teg.predict(window.iloc[-n_obs_per_period:], model_w)

                #Set ground true values
                if testing == "anomalous":
                    groundtrue = np.ones(obs)        
                else:
                    groundtrue = np.zeros(obs)

                #Compute confusion matrix
                cm = teg.compute_confusion_matrix(groundtrue, outliers)

                # Accumulate the cms
                cm_accumulative['tp'] += cm['tp']
                cm_accumulative['tn'] += cm['tn']
                cm_accumulative['fp'] += cm['fp']
                cm_accumulative['fn'] += cm['fn']

            #Collect detector configuration
            detector = {'metric': metric, 'n_bins': n_bins, 'n_obs_per_period':n_obs_per_period, 'alpha': alpha}

            #Collect performance metrics in a dictionary
            perf = {'tmc': time2build, 'tmg': time2graphs, 'tmgl': time2global, 'tmm': time2metrics, 'tmp': time2predict, 'tmw': time2window}

            #Print and store basic metrics
            teg.print_metrics(detector, testing, perf, cm_accumulative)
            results_path = cwd + RESULTS_PATH
            teg.metrics_to_csv(detector, testing, perf, cm_accumulative, results_path)

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


