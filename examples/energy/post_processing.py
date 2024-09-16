"""
@Author: Simona Bernardi
@Date: updated 15/10/2022

The script:
- Produces the plot of the two testing sets during the first week
- From the file TEGDET_VARIANTS_RESULTS_PATH
-- Prints on stdout the statistics of the times to build and to make predictions 
-- Shows the accuracy of the TEG-detectors variants as a barplot
- From the file TEGDET_PARAMS_SENSITIVITY_RESULTS_PATH
For a given TEG-detector variant:
--- Shows the mean times to build and to execute vs/ n_bins and n_obs_per_period
--- Shows the accuracy vs/ alpha and n_obs_per_period (n_bins fixed to default value)
--- Shows the accuracy vs/ n_bins and n_obs_per_period (alpha fixed to default value)
--- Shows the accuracy vs/ alpha and n_bins (n_obs_per_period fixed to default value)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Datasets path
TEST_NORMAL_DS_PATH = "/dataset/energy/test_normal.csv"
TEST_ANOMALOUS_DS_PATH = "/dataset/energy/test_anomalous.csv"

#Results path
#Result files generated by the scripts "tegdet_variants" and "tegdet_params_sensitivity.py"
#TEGDET_VARIANTS_RESULTS_PATH = "/script_results/energy/tegdet_variants_results_ref.csv"
#TEGDET_PARAMS_SENSITIVITY_RESULTS_PATH = "/script_results/energy/tegdet_params_sensitivity_results_ref.csv"
#Uncomment these lines to analyse the results once the scripts have been run
TEGDET_VARIANTS_RESULTS_PATH = "/script_results/energy/tegdet_variants_results.csv"
TEGDET_PARAMS_SENSITIVITY_RESULTS_PATH = "/script_results/energy/tegdet_params_sensitivity_results.csv"

#Uncomment these lines to post-process the results related to all the available dissimilarity metrics
list_of_metrics = ["Hamming", #"Clark", "Canberra", "Lorentzian", "Kulczynski", "Divergence", "Cosine",
                  #  "Jaccard", "Dice", "KL", "Jeffreys", "JS", "Euclidean", "Cityblock", 
                  #   "Chebyshev", "Minkowski", "Braycurtis", "Gower", "Soergel", "Bhattacharyya", "Hellinger", 
                  #   "Matusita", "Squaredchord", "Pearson", "Neyman", "Squared", "Probsymmetric", "Additivesymmetric" 
                    ]

def compare_testing_sets(cwd, n_obs):
    
    dp_range  = range(n_obs) #one week
    test_normal_ds_path = cwd+TEST_NORMAL_DS_PATH
    test_anomalous_ds_path = cwd+TEST_ANOMALOUS_DS_PATH
    testing_normal = pd.read_csv(test_normal_ds_path)['Usage'].head(len(dp_range))
    testing_anomalous = pd.read_csv(test_anomalous_ds_path)['Usage'].head(len(dp_range))

    #Generate plot
    plt.plot(dp_range, testing_normal, label="normal")
    plt.plot(dp_range, testing_anomalous, label="anomalous", linestyle=':') #dotted lines
    plt.legend(fontsize="12")
    plt.xlabel('Time (every half an hour)',fontsize="12")
    plt.ylabel('Usage (kWh)',fontsize="12")
    plt.show()

def generate_report_teg_variants(cwd):

    print("-------- TEG variants analysis report ----------------")
    #Load the results
    results_path = cwd + TEGDET_VARIANTS_RESULTS_PATH
    df = pd.read_csv(results_path)   
    #Remove parameters and testing_set columns
    df = df[['detector','time2build', 'time2graphs', 'time2global', 'time2metrics','time2predict', 'mem2graphs','tp','tn','fp','fn']]
    #Group by detector and takes the sum (of the two testing sets results)
    df_grouped = df.groupby('detector').sum()
    

    #Extract execution times (in ms.: sum_of_the_times / 2 * 1000)
    time2build = df_grouped['time2build'] * 500
    time2graphs = df_grouped['time2graphs'] * 500
    time2global = df_grouped['time2global'] * 500
    time2metrics = df_grouped['time2metrics'] * 500
    time2predict = df_grouped['time2predict'] * 500
    mem2graphs = df_grouped['mem2graphs']
    #Timing statistics on stdout
    print("Time to build the model (ms):", time2build.describe())
    print("Time to generate TEGs (ms):", time2graphs.describe())
    print("Time to compute global graph (ms):", time2global.describe())
    print("Time to compute metrics (ms):", time2metrics.describe())
    print("Time to make predictions: (ms)", time2predict.describe())
    print("Memory used to build all graphs: (mb)", mem2graphs.describe())
    print("------------------------------------------------------")

    #Get the detectors list
    detectors = df_grouped.index.tolist()
    #Compute the accuracy from the confusion matrix
    num = df_grouped['tp']+df_grouped['tn']
    den = num + df_grouped['fp']+ df_grouped['fn']
    accuracy = num / den   
    print("Accuracy:")
    print(accuracy)

    #Generate barplot
    fig, ax = plt.subplots()
    y_pos = np.arange(len(detectors))
    ax.barh(y_pos, accuracy, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(detectors,fontsize=10)
    ax.invert_yaxis()  
    ax.set_xlabel('Accuracy')
    plt.show()
    

def plot_3D(xlabel,ylabel,zlabel,x,y,z):
    #Set figure 
    ax = plt.axes(projection='3d')
    #Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_zlabel(zlabel)

    #Set coordinates
    X, Y = np.meshgrid(x,y)

    #Generate surface plot for tmc
    Z = np.resize(z, (len(X),len(Y)))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #Plot
    plt.show()

def plot_3D_accuracy(df_view, fixed_param, ref_value, y_view, x_view):
    df_view = df_view[df_view[fixed_param] == ref_value]
    #Group by parameter configuration and take the sum
    df_grouped = df_view.groupby([y_view,fixed_param, x_view]).sum() 
    
    #Compute the accuracy from the confusion matrix
    num = df_grouped['tp']+df_grouped['tn']
    den = num + df_grouped['fp']+ df_grouped['fn']
    accuracy = num / den
    
    print("-------- Parameters sensitivity analysis report ----------------")
    print("Accuracy statistics", accuracy.describe())
    print("----------------------------------------------------------------")
      
    #Plot accuracy figures
    x = np.unique(df_view[[x_view]].to_numpy())
    y = np.unique(df_view[[y_view]].to_numpy())
    accuracy = accuracy.to_numpy()
    plot_3D(x_view,y_view,"accuracy",x,y, accuracy)


def generate_report_params_sensitivity(cwd,detector):
    
    print("Metric: ", detector)
    #Load the results
    results_path = cwd + TEGDET_PARAMS_SENSITIVITY_RESULTS_PATH
    df = pd.read_csv(results_path)   
    
    #Select the detector
    df = df[df["detector"] == detector]

    #Remove detector and testing_set columns
    df = df[['n_bins','n_obs_per_period','alpha','time2build', 'time2graphs', 'time2global',
            'time2metrics','time2predict', 'mem2graphs','tp','tn','fp','fn']]
 
    #Performance sensitivity analysis
    #Get parameters ranges
    n_bins = np.unique(df[['n_bins']].to_numpy())
    n_obs = np.unique(df[['n_obs_per_period']].to_numpy())
    alpha = np.unique(df[['alpha']].to_numpy())

    #Remove alpha and confusion matrix columns
    df_view = df[['n_bins','n_obs_per_period','time2build','time2graphs', 'time2global',
            'time2metrics','time2predict', 'mem2graphs']]

    #Group by parameter configuration and take mean times (converted in ms: * 1000)
    df_grouped = df_view.groupby(['n_obs_per_period','n_bins']).mean() * 1000 

    print("Execution times:", df_grouped.describe())

    tmc = df_grouped['time2build'].to_numpy()
    tmg = df_grouped['time2graphs'].to_numpy()
    tmgl = df_grouped['time2global'].to_numpy()
    tmm = df_grouped['time2metrics'].to_numpy()
    tmp = df_grouped['time2predict'].to_numpy()
    df_grouped = df_view.groupby(['n_obs_per_period','n_bins']).mean()
    m2g = df_grouped['mem2graphs'].to_numpy()

    #Plot performance figures    
    plot_3D("n_bins","n_obs_per_period","time2build (ms)",n_bins,n_obs,tmc)
    plot_3D("n_bins","n_obs_per_period","time2graphs (ms)",n_bins,n_obs,tmg)
    plot_3D("n_bins","n_obs_per_period","time2global (ms)",n_bins,n_obs,tmgl)
    plot_3D("n_bins","n_obs_per_period","time2metrics (ms)",n_bins,n_obs,tmm)
    plot_3D("n_bins","n_obs_per_period","time2predict (ms)",n_bins,n_obs,tmp)
    plot_3D("n_bins","n_obs_per_period","mem2graphs (mb)",n_bins,n_obs,m2g)


    #Accuracy sensitivity analysis
 
    #Remove time columns
    df_view = df[['n_bins','n_obs_per_period','alpha','tp','tn','fp','fn']]   

    #Fix n_bins = 30 (reference configuration)
    plot_3D_accuracy(df_view, "n_bins", 30, "n_obs_per_period", "alpha")

    #Fix alpha = 5 (reference configuration)
    plot_3D_accuracy(df_view, "alpha", 5, "n_obs_per_period", "n_bins")

    #Fix n_obs_per_period = 336 (reference configuration)
    plot_3D_accuracy(df_view, "n_obs_per_period", 336, "n_bins", "alpha")
    

if __name__ == '__main__':

    #Get the current directory
    cwd = os.getcwd() 

    #Compare testing sets (first week of observations)
    compare_testing_sets(cwd, 336)

    generate_report_teg_variants(cwd)

    for metric in list_of_metrics:
        generate_report_params_sensitivity(cwd, metric)
    
    