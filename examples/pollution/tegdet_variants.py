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

    # Slide window method after the first build
    slide_window_scheme = teg.get_sw()

    for testing in list_of_testing:

        # We preserve the schema from the model built, for anomalous and normal datasets
        OriginalModel = model
        teg.update_mb(OriginalModel)

        # cm to store all cms generated during sliding window
        cm_accumulative = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        #Path of the testing
        test_ds_path = cwd  + TEST_DS_PATH + testing + ".csv"              
        #Load testing dataset
        test_ds = teg.get_dataset(test_ds_path)

        # Full dataset with test and train
        full_ds = pd.concat([train_ds, test_ds], ignore_index=True)

        # Initialize the window using the Initial sheme given a test dataset (normal or anomalous)
        slide_window = slide_window_scheme
        slide_window.initialize_window(full_ds)

        #Make prediction
        outliers, obs, time2predict = teg.predict(test_ds.head(n_obs_per_period), model)
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
        results_path = cwd  + RESULTS_PATH
        teg.metrics_to_csv(detector, testing, perf, cm_accumulative, results_path)
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric,n_bins,n_obs_per_period,alpha)
