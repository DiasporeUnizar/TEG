"""
@Author: Simona Bernardi
@Date: updated 01/04/2022

- Produces the plot of the two testing sets during the first week
- Produces the bar-plot of showing the accuracy of the TEG-detector variants from RESULTS_PATH file
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Dataset/results paths
TEST_NORMAL_DS_PATH = "/dataset/test_normal.csv"
TEST_ANOMALOUS_DS_PATH = "/dataset/test_anomalous.csv"
RESULTS_PATH = "/script_results/userScript_results.csv"

def compare_scenarios(cwd):
    
    n_datapoints  = range(336) #two days
    test_normal_ds_path = cwd+TEST_NORMAL_DS_PATH
    test_anomalous_ds_path = cwd+TEST_ANOMALOUS_DS_PATH
    scenarioN = pd.read_csv(test_normal_ds_path)['Usage'].head(len(n_datapoints))
    plt.plot(n_datapoints, scenarioN, label="normal")
    scenarioA = pd.read_csv(test_anomalous_ds_path)['Usage'].head(len(n_datapoints))
    plt.plot(n_datapoints, scenarioA, label="anomalous", linestyle=':') #dotted lines

    plt.legend()
    plt.xlabel('Time (every half an hour)')
    plt.ylabel('Usage (kWh)')

    plt.show()

def generate_quality_report(cwd):
    results_path = cwd + RESULTS_PATH
    df = pd.read_csv(results_path)
    #Removing testing_set column
    df = df[['detector','time_model_creation','time_model_prediction','n_tp','n_tn','n_fp','n_fn']]
    #Grouping on detector by sum
    df_grouped = df.groupby('detector').sum()
    #Getting the detectors list
    detectors = df_grouped.index.tolist()
    #Computing mean values of times
    time2build = df_grouped['time_model_creation']/2
    time2predict = df_grouped['time_model_prediction']/2
    print(time2build.describe())
    print(time2predict.describe())

    #Computing the accuracy
    num = df_grouped['n_tp']+df_grouped['n_tn']
    den = num + df_grouped['n_fp']+ df_grouped['n_fn']
    accuracy = num / den

    fig, ax = plt.subplots()
    y_pos = np.arange(len(detectors))
    ax.barh(y_pos, accuracy, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos,fontsize=10)
    ax.set_yticklabels(detectors,fontsize=10)
    ax.invert_yaxis()  
    ax.set_xlabel('Accuracy')
    plt.show()


if __name__ == '__main__':

    cwd = os.getcwd() 
    compare_scenarios(cwd)
    generate_quality_report(cwd)
    
    