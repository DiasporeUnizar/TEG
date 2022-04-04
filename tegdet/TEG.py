"""
@Author: Simona Bernardi, Raúl Javierre
@Date: 04/04/2022

Time-Evolving-Graph detector Version 1.0
This modules includes the following classes:

- TEG
- LevelExtractor
- TEGdetector

that implements the detectors based on TEG.

"""

from time import time
import numpy as np
import pandas as pd
import os
import sys      
from tegdet.graph_discovery import GraphGenerator, Graph
from tegdet.graph_comparison import *

class TEG():
    #Default values
    _N_BINS = 30
    _N_OBS_PER_PERIOD = 336
    _ALPHA = 5

    def __init__(self, metric, n_bins=_N_BINS, n_obs_per_period=_N_OBS_PER_PERIOD, alpha=_ALPHA):
        self.metric = metric
        self.n_bins = n_bins
        self.n_obs_per_period= n_obs_per_period
        self.alpha = alpha
        self._baseline = None
        self._global_graph= None

    def get_dataset(self,ds_path):
        """
        Loads the dataset from "ds_path" (csv file), renames the two columns as TS (timestamp) and 
        data points (DP), respectively, and return the pandas dataframe "dt"
        """
        df = pd.read_csv(ds_path)
        df.columns = ['TS','DP']
        return df

    def build_model(self, training_dataset):
        """
        Builds the prediction model based on the "training_dataset" and returns it together with
        the time to build the model
        """
        t0 = time()
        obs = training_dataset['DP']
        tegd = TEGdetector(obs, self.n_bins)
        self._baseline, self._global_graph = tegd.buildModel(self.metric, obs, int(len(training_dataset.index) / self.n_obs_per_period))

        return tegd, time() - t0

    def predict(self, testing_dataset, model):
        """
        Makes predictions on the "testing_dataset" using the model. It returns three values:
        number of outliers, total number of observations, and time to make predictions
        """
        t0 = time()
        dataPoints = testing_dataset['DP']
        test = model.makePrediction(self._baseline, self._global_graph, self.metric, dataPoints, int(len(testing_dataset.index) / self.n_obs_per_period))
        outliers = model.computeOutliers(self._baseline, test, 100 - self.alpha)

        return outliers, int(len(testing_dataset.index) / self.n_obs_per_period), time() - t0
        
    def compute_confusion_matrix(self, groundTrue, predictions):
        """
        Pre: "groundTrue" is a vector with the true values, "predictions" is a vector 
        with predicted values (0,1)
        Post: Computes the confusion matrix. It returns the confusion matrix as dictionary type.
        """
        cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        cm['tp'] = ((groundTrue == 1) & (predictions == 1)).sum()
        cm['tn'] = ((groundTrue == 0) & (predictions == 0)).sum()
        cm['fp'] = ((groundTrue == 0) & (predictions == 1)).sum()
        cm['fn'] = ((groundTrue == 1) & (predictions == 0)).sum()

        return cm

    def print_metrics(self, detector, testing_set, perf, cm):
        """
        Prints on the stdout: the "detector" configuration, the "testing set", the performance metrics "perf" 
        (time to build the model and to make predictions) and the confusion matrix "cm" 
        """
        print("Detector:\t\t\t", detector['metric'])
        print("N_bins:\t\t\t\t", detector['n_bins'])
        print("N_obs_per_period:\t\t", detector['n_obs_per_period'])
        print("Alpha:\t\t\t\t", detector['alpha'])
        print("Testing set:\t\t\t", testing_set)
        print("Time to build the model:\t", perf['tmc'], "seconds")
        print("Time to make prediction:\t", perf['tmp'], "seconds")
        print("Confusion matrix:\t\n\n", cm)

    def metrics_to_csv(self, detector, testing_set, perf, cm,results_csv_path):
        """
        Saves in the csv file "result_csv_path", the "detector" configuration, the  "testing set",
        the performance metrics "perf" (time to build the model and to make predictions)
        and the confusion matrix "cm" 
        """
        df = pd.DataFrame({
                           'detector': detector['metric'],
                           'n_bins': detector['n_bins'],
                           'n_obs_per_period': detector['n_obs_per_period'],
                           'alpha': detector['alpha'],
                           'testing_set': testing_set,
                           'time2build': perf['tmc'],
                           'time2predict': perf['tmp'],
                           'tp': cm['tp'],
                           'tn': cm['tn'],
                           'fp': cm['fp'],
                           'fn': cm['fn']},
                           index=[0])

        df.to_csv(results_csv_path, mode='a', header=not os.path.exists(results_csv_path), index=False)


class LevelExtractor:
    """Extractor of levels """

    def __init__(self, minValue, step, n_bins):
        """
        Creates levels [0,1,..,n_bins+2], the last two positions for the possible 
        outliers of the testing dataset (minimum than the min_train_value and maximum 
        of the max_train_value)
        """
        self.level = np.arange(n_bins+2)
        self.step = step
        self.minValue = minValue

    def getLevel(self, observations):
        """
        Discretization of  "observations" according to the "self.level"
        "observations" is a np.array (of floats)
        """
        nObs = len(observations)  # number of observations
        level = -1 * np.ones(nObs)  # array initialized with -1
        level = level.astype(int)  # level is a np.array of int

        # Case: "observations" (testing set) is lower than the min_train_value
        #       level position the last
        level = np.where((observations < self.minValue), self.level[-1], level)
        
        n_bins= len(self.level)-2
        i = 0  # while iterator
        while i < n_bins:
            lowerB = self.minValue + i * self.step
            upperB = self.minValue + (i + 1) * self.step
            level = np.where((lowerB <= observations) & (observations < upperB),
                             self.level[i], level)
            i += 1

        # Case: "observations" (testing set) is greater than the max_train_value 
        # level position the penultimate      
        level = np.where(upperB <= observations, self.level[-2], level)

        return level


class TEGdetector:
    """Builds the model and makes predictions """

    def __init__(self, observations, n_bins):
        """
        Constructor that initializes the TEGdetector based on the training dataset "observations" and "n_bins"
        Creates a new level extractor.
        """
        m = observations.min()  # min of the TRAINING dataset
        M = observations.max()  # max of the TRAINING dataset
        step = (M - m) / n_bins  # usage increment step

        self.le = LevelExtractor(m, step, n_bins)

    
    def sumGraph(self, gr1, gr2):
        """
        Pre: Graph "gr1" nodes set includes graph "gr2" nodes set
        Adds to graph "gr1" the graph "gr2"
        """
        for i in range(gr2.nodes.size):
            row = gr2.nodes[i]
            gr1.nodesFreq[row] += gr2.nodesFreq[i]
            for j in range(gr2.nodes.size):
                col = gr2.nodes[j]
                gr1.matrix[row][col] += gr2.matrix[i][j]

    def getGlobalGraph(self, graphs):
        """
        Creates and returns a global graph as the sum of a list of "graphs".
        """
        global_graph = Graph()
        n_bins = len(self.le.level)
        global_graph.nodes = np.arange(n_bins, dtype=int)
        global_graph.nodesFreq = np.zeros((n_bins), dtype=int)
        global_graph.matrix = np.zeros((n_bins, n_bins), dtype=int)
        for gr in graphs:
            self.sumGraph(global_graph, gr.graph)

        return global_graph

    def generateTEG(self, observationsClassified, n_periods):
        """
        Creates and returns the time evolving graphs series from the discretized observations
        "observationClassified" and the number of periods "n_periods"
        """
        nObs = int(len(observationsClassified) / n_periods)  # number of observations per period
        graphs = []
        for period in range(n_periods):
            gr = GraphGenerator()
            obsClass = observationsClassified[period * nObs:(period + 1) * nObs]
            # Transforms to a dataframe (needed to generate the graph)
            el = pd.DataFrame({'Period': period * np.ones(nObs), 'DP': obsClass})
            gr.generateGraph(el)
            graphs.append(gr)

        return graphs

    def computeGraphDist(self, gr1, gr2, metric):
        """
        Computes and returns the distance between graphs "gr1" and "gr2" using the dissimilarity "metric"
        """
        grcomp = GraphComparator(gr1, gr2)
        # Graph resizing
        grcomp.resizeGraphs()

        # Computes the difference based on the metric
        className = "Graph" + metric + "Dissimilarity"
        grcomp.__class__ = getattr(sys.modules[__name__], className) #Convert a string to a class object
        metricValue = grcomp.compareGraphs()

        return metricValue

    def buildModel(self, metric, observations, n_periods):
        """
        Builds the distribution of the dissimilarities based on "metric", "observations", and
        "n_periods". It returns the ditribution of the dissimilarities and the global graph.
        """

        # Gets observation levels (discretization)
        obsDiscretized = self.le.getLevel(observations)

        # Generates the time-evolving graphs
        graphs = self.generateTEG(obsDiscretized, n_periods)

        # Gets the graph of the training period
        global_graph = self.getGlobalGraph(graphs)

        # Computes the distance between each graph and the global graph
        graph_dist = np.empty(n_periods)
        for period in range(n_periods):
            graph_dist[period] = self.computeGraphDist(graphs[period].graph, global_graph, metric)

        return graph_dist, global_graph

    def makePrediction(self, baseline, global_graph, metric, observations, n_periods):
        """
        Makes and returns the predictions of the "observations" based on the "baseline" distribution 
        of the dissimilarities, "global_graph", the dissimilarity "metric" and "n_periods"
        """

        # Gets consumption levels (consumption discretization)
        obsDiscretized = self.le.getLevel(observations)

        # Generates the time-evolving graphs
        graphs = self.generateTEG(obsDiscretized, n_periods)

        # Computes the distance between each graph and the global graph
        graph_dist = np.empty(n_periods)
        for period in range(n_periods):
            graph_dist[period] = self.computeGraphDist(graphs[period].graph, global_graph, metric)

        return graph_dist

    def computeOutliers(self, baseline, prediction, sigLevel):
        """
        Computes the outliers based on the "baseline" distribution of the dissimilarities,
        the "prediction" and the significance level "sigLevel"
        """
        perc = np.percentile(baseline, sigLevel)

        # Sets a counter vector to zero
        n_out = np.zeros(prediction.size)
        # Dissimilarity tests
        n_out = np.where(prediction > perc, n_out + 1, n_out)

        return n_out
