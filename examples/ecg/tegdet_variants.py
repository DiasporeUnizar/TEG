"""
@Author: Simona Bernardi
@Date: updated 12/10/2022

Input dataset:
- ecg0606.csv: index column and colums headings are added from  "ecg0606_1.csv" 
--> Source: "ecg0606_1.csv" (https://github.com/GrammarViz2/grammarviz2_src)
--> Original source: ECG recordings "sele0606" (https://physionet.org/content/qtdb/1.0.0/)

The script:
- Builds a prediction model on the training set for the set of dissimilarity metric (TEG-detectors variants)
- Makes predictions on the testing set 
- Compute the confusion matrix and performance metrics 

Output:
- Stores them in RESULTS_PATH
"""

import os
import pandas as pd
import numpy as np
from tegdet.teg import TEGDetector

#Input dataset/output results paths
TS_PATH = "/dataset/ecg/ecg0606.csv"
RESULTS_PATH = "/script_results/ecg/tegdet_variants_results.csv"


#List of metrics (detector variants)
list_of_metrics = ("Hamming", "Cosine", "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                "Clark", "Additivesymmetric" )

def build_and_predict(metric,n_bins,n_obs_per_period,alpha):
       
    #Create a new tegdet
    tegd = TEGDetector(metric,n_bins,n_obs_per_period,alpha)
    
    #Load time series
    cwd = os.getcwd() 
    ts_path = cwd + TS_PATH
    ts = tegd.get_dataset(ts_path)

    #Partition the time series in training and testing sets
    test = ts[:600] 
    train = ts[600:]
    
    #Build model
    model, time2build, time2graphs, time2global, time2metrics = tegd.build_model(train)

    # Slide window method after the first build
    slide_window_scheme = tegd.get_sw()

    # cm to store all cms generated during sliding window
    cm_accumulative = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    # Full dataset with test and train
    full_ds = pd.concat([train, test], ignore_index=True)

    # Initialize the window using the Initial sheme given a test dataset (normal or anomalous)
    slide_window = slide_window_scheme
    slide_window.initialize_window(full_ds)

    #Make prediction
    outliers, n_periods, time2predict = tegd.predict(test.head(n_obs_per_period), model)

    #Set ground true values 
    ground_true = np.zeros(n_periods)
    ground_true[n_periods-1] = 1

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

        window = slide_window.slide_window(full_ds) # Moves the window immediately after the original build process and the latest one

        if window is None:
            #print(f"No more data available for sliding window in dataset {testing}.")
            break

        time2window += tegd.process_window(train, n_bins + 2)

        # Make prediction on latest week
        model_w = tegd.get_mb()
        outliers, obs, time2predict = tegd.predict(window.iloc[-n_obs_per_period:], model_w)

        #Set ground true values
        ground_true = np.zeros(n_periods)
        ground_true[n_periods-1] = 1

        #Compute confusion matrix
        cm = tegd.compute_confusion_matrix(ground_true, outliers)

        # Accumulate the cms
        cm_accumulative['tp'] += cm['tp']
        cm_accumulative['tn'] += cm['tn']
        cm_accumulative['fp'] += cm['fp']
        cm_accumulative['fn'] += cm['fn']

    #Collect detector configuration
    detector = {'metric': metric, 'n_bins': n_bins, 'n_obs_per_period':n_obs_per_period, 'alpha': alpha}
    #Collect performance metrics in a dictionary
    perf = {'tmc': time2build, 'tmg': time2graphs, 'tmgl': time2global, 'tmm': time2metrics, 'tmp': time2predict, 'tmw': time2window}

    #Store basic metrics
    results_path = cwd + RESULTS_PATH
    tegd.metrics_to_csv(detector, "testing", perf, cm_accumulative, results_path)
        

if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric,30,170,5)
