"""
@Author: Simona Bernardi, Ángel Villanueva
@Date: updated 07/10/2024
Input dataset:
- Energy consumption (in kWh), every half-an-hour, registered by a smart meter.
- The training set is over 60 weeks and the two testing sets are over 15 weeks
The script:
- Builds a prediction model for the set of dissimilarity metric (TEG-detectors variants)
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
TRAINING_DS_PATH = "/dataset/energy/training.csv"
TEST_DS_PATH = "/dataset/energy/test_"
RESULTS_PATH = "/script_results/energy/tegdet_variants_results.csv"

#List of testing
list_of_testing = ("normal", "anomalous")

#List of metrics (detector variants)
list_of_metrics = ("Hamming", "Cosine", "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                    "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                    "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                    "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                    "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                    "Clark", "Additivesymmetric" )

#Parameters: default values
n_bins = 30
n_obs_per_period = 336
alpha = 5

def build_and_predict(metric):
    cwd = os.getcwd() 
    train_path = cwd + TRAINING_DS_PATH

    tegd = TEGDetector(metric)
    #Load training dataset
    train = tegd.get_dataset(train_path)
    #Build model
    model, time2build, time2graphs, time2global, time2metrics = tegd.build_model(train)

    for testing in list_of_testing:

        # cm to store all cms generated during sliding window
        cm_accumulative = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        #Path of the testing
        test_path = cwd + TEST_DS_PATH + testing + ".csv"               
        #Load testing dataset
        test = tegd.get_dataset(test_path)

        # Full dataset with test and train
        full_ds = pd.concat([train, test], ignore_index=True)

        # Initialize the window using the Initial sheme given a test dataset (normal or anomalous)
        tegd.initialize_window(full_ds)

        #Make prediction
        outliers, n_periods, time2predict = tegd.predict(test.head(n_obs_per_period), model)
        #Set ground true values
        if testing == "anomalous":
            ground_true = np.ones(n_periods)        
        else:
            ground_true = np.zeros(n_periods)

        #Compute confusion matrix
        cm = tegd.compute_confusion_matrix(ground_true, outliers)

        # Accumulate the cms
        cm_accumulative['tp'] += cm['tp']
        cm_accumulative['tn'] += cm['tn']
        cm_accumulative['fp'] += cm['fp']
        cm_accumulative['fn'] += cm['fn']

        # Metrics to store the time during window processing
        time2window = 0

        # Compute the rest of the weeks on the testing data
        while True:

            window = tegd.slide_window(full_ds) # Moves the window immediately after the original build process and the latest one

            if window is None:
                #print(f"No more data available for sliding window in dataset {testing}.")
                break

            time2window += tegd.process_window(train, n_bins + 2)

            # Make prediction on latest week
            outliers, obs, time2predict = tegd.predict(window.iloc[-n_obs_per_period:], model)

            #Set ground true values
            if testing == "anomalous":
                groundtrue = np.ones(obs)        
            else:
                groundtrue = np.zeros(obs)

            #Compute confusion matrix
            cm = tegd.compute_confusion_matrix(groundtrue, outliers)

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
        tegd.print_metrics(detector, testing, perf, cm_accumulative)
        results_path = cwd + RESULTS_PATH
        tegd.metrics_to_csv(detector, testing, perf, cm_accumulative, results_path)
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric)
