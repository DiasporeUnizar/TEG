"""
@Author: Simona Bernardi
@Date: updated 14/2/2023
Input dataset:
- Energy consumption (in kWh), every half-an-hour, registered by a solar smart meter.
- The training set is over 50 weeks and the two testing sets are over 50 weeks
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
TRAINING_DS_PATH = "/dataset/energy/100_0_50.csv"
TEST_DS_PATH = [ "/dataset/energy/100_51_101.csv", 
                 "/dataset/energy/100_Rating_51_101.csv", 
                 "/dataset/energy/100_Percentile_51_101.csv" ]

RESULTS_PATH = "/script_results/energy/tegdet_variants_results_solar.csv"

#List of testing
list_of_testing = [ #"normal", 
                    #"rating attack", 
                    "percentile attack"
                    ]

#List of metrics (detector variants)
list_of_metrics = [# "Hamming", 
                    "Cosine" #, "Jaccard", "Dice", "KL", "Jeffreys", "JS", 
                   # "Euclidean", "Cityblock", "Chebyshev", "Minkowski", "Braycurtis",
                   # "Gower", "Soergel", "Kulczynski", "Canberra", "Lorentzian",
                   # "Bhattacharyya", "Hellinger", "Matusita", "Squaredchord",
                   # "Pearson", "Neyman", "Squared", "Probsymmetric", "Divergence",
                   # "Clark", "Additivesymmetric" 
                 ]

#Parameters: default values
n_bins = 6
n_obs_per_period = 336
alpha = 5

def build_and_predict(metric):
    cwd = os.getcwd() 
    train_path = cwd + TRAINING_DS_PATH

    tegd = TEGDetector(metric, n_bins, n_obs_per_period, alpha)
    #Load training dataset
    train = tegd.get_dataset(train_path)
    #Build model
    model, time2build = tegd.build_model(train)

    ############### This part should be implemented as a new API to export the teg
    tegg_path = cwd + "/script_results/energy/tegg_training.txt"
    f = open(tegg_path,'a')

    #### Print teg (graphs already resized according to the global graph)
    matrix_global = model.get_global_graph().get_matrix()
    teg = model.get_tegg().get_teg()
    n = len(teg)

    #Number of graphs (including global grah) and matrix sizes
    f.write("{} {}\n".format(n+1, matrix_global.size))
 
    # Global graph
    print("Global graph:", matrix_global)
    np.savetxt(f,matrix_global, fmt='%.1f', delimiter=" ", newline= " ")
    
    # Graphs of the training period
    for g in range(n):
        matrix = teg[g].get_matrix()
        print(matrix)
        np.savetxt(f,matrix, fmt='%.1f', delimiter=" ", newline= " ")
        f.write("\n")

    f.close()

    # Print baseline
    baseline_path = cwd + "/script_results/energy/baseline.txt"
    f = open(baseline_path,'a')
    baseline = model.get_baseline();
    np.savetxt(f,baseline, fmt='%.8f', delimiter=" ", newline= " ")
    print("Baseline:", baseline)
    f.close()
 
    #######################################


    for (test_set, test_type) in zip(TEST_DS_PATH, list_of_testing):

        #Path of the testing
        test_path = cwd + test_set             
        #Load testing dataset
        test = tegd.get_dataset(test_path)
        #Make prediction
        outliers, n_periods, time2predict = tegd.predict(test, model)
        #Set ground true values
        if test_type == "normal":
            ground_true = np.zeros(n_periods)        
        else:
            ground_true = np.ones(n_periods)

        #Compute confusion matrix
        cm = tegd.compute_confusion_matrix(ground_true, outliers)

        #Collect detector configuration
        detector = {'metric': metric, 'n_bins': n_bins, 'n_obs_per_period':n_obs_per_period, 'alpha': alpha}
        #Collect performance metrics in a dictionary
        perf = {'tmc': time2build, 'tmp': time2predict}

        #Print and store basic metrics
        tegd.print_metrics(detector, test_type, perf, cm)
        results_path = cwd + RESULTS_PATH
        tegd.metrics_to_csv(detector, test_type, perf, cm, results_path)

        ############### This part should be implemented as a new API to export the teg
        tegg_path = cwd + "/script_results/energy/tegg_testing.txt"
        f = open(tegg_path,'a')

        teg = tegd.get_ad().get_tegg().get_teg()
        n = len(teg)

        #Number of graphs (including global grah) and matrix sizes
        f.write("{} {}\n".format(n+1, matrix_global.size))

        # Global graph
        print("Testing - Global graph:", matrix_global)
        np.savetxt(f,matrix_global, fmt='%.1f', delimiter=" ", newline= " ")

        # Graphs of the testing period
        for g in range(n):
            matrix = teg[g].get_matrix()
            print(matrix)
            np.savetxt(f,matrix, fmt='%.1f', delimiter=" ", newline= " ")
            f.write("\n")

        f.close()

        ###############
        
if __name__ == '__main__':

    for metric in list_of_metrics:

        build_and_predict(metric)
