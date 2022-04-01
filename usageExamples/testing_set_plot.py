"""
@Author: Simona Bernardi
@Date: updated 30/03/2022

- Produces the plot of the two training sets during the first week
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TEST_NORMAL_DS_PATH = "/dataset/test_normal.csv"
TEST_ANOMALOUS_DS_PATH = "/dataset/test_anomalous.csv"

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


if __name__ == '__main__':

    cwd = os.getcwd() 
    compare_scenarios(cwd)