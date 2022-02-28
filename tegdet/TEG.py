"""
@Author: Simona Bernardi, Raúl Javierre
@Date: 28/02/2022

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
import sys      # needed to convert a string to a class object
from tegdet.graph_discovery import GraphGenerator, Graph
from tegdet.graph_comparison import GraphComparator, GraphHammingDissimilarity, \
    GraphCosineDissimilarity, GraphJaccardDissimilarity, GraphDiceDissimilarity, \
    GraphKLDissimilarity, GraphJeffreysDissimilarity, GraphJSDissimilarity, \
    GraphEuclideanDissimilarity, GraphCityblockDissimilarity, GraphChebyshevDissimilarity, \
    GraphMinkowskiDissimilarity, \
    GraphBraycurtisDissimilarity, GraphGowerDissimilarity, GraphSoergelDissimilarity, \
    GraphKulczynskiDissimilarity, GraphCanberraDissimilarity, \
    GraphLorentzianDissimilarity, \
    GraphBhattacharyyaDissimilarity, GraphHellingerDissimilarity, GraphMatusitaDissimilarity, \
    GraphSquaredchordDissimilarity, \
    GraphPearsonDissimilarity, GraphNeymanDissimilarity, GraphSquaredDissimilarity, \
    GraphProbsymmetricDissimilarity, \
    GraphDivergenceDissimilarity, GraphClarkDissimilarity, GraphAdditivesymmetricDissimilarity


class TEG():
    #Default values
    _N_OBS_PER_PERIOD = 336
    _N_BINS = 30
    _ALPHA = 5

    def __init__(self, metric, n_bins=_N_BINS, alpha=_ALPHA, n_obs_per_period=_N_OBS_PER_PERIOD):
        self.metric = metric
        self.n_bins = n_bins
        self.alpha = alpha
        self.n_obs_per_period= n_obs_per_period
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
        dataPoints = training_dataset['DP']
        teg = TEGdetector(dataPoints, self.n_bins)
        self._baseline, self._global_graph = teg.buildModel(self.metric, dataPoints, int(len(training_dataset.index) / self.n_obs_per_period))

        return teg, time() - t0

    def predict(self, testing_dataset, model):
        """
        Makes predictions on the "testing_dataset" using the model. It returns three values:
        number of outliers, total number of observations, and time to make predictions
        """
        t0 = time()
        dataPoints = testing_dataset['DP']
        test = model.makePrediction(self._baseline, self._global_graph, self.metric, dataPoints, int(len(testing_dataset.index) / self.n_obs_per_period))
        n_outliers = model.computeOutliers(self._baseline, test, 100 - self.alpha)

        return n_outliers, int(len(testing_dataset.index) / self.n_obs_per_period), time() - t0

    def compute_confusion_matrix(self, n_observations, n_outliers, is_anomalous):
        """
        Pre: "n_observations" and "n_outliers" are computed from a testing dataset that can be 
        either a normal scenario (without anomalies) or an anomalous scenario (all the observations are anomalous)
        Post: Computes the confusion matrix based on the total number of observations "n_observations",
        "n_outliers", and the type of scenario "is_anomalous" (boolean).
        It returns the confusion matrix as dictionary type.
        """
        cm = {'n_tp': 0, 'n_tn': 0, 'n_fp': 0, 'n_fn': 0}
        if is_anomalous:  # if anomaly is detected, it is a true positive
            cm['n_tp'] = int(n_outliers)
            cm['n_fn'] = int(n_observations - n_outliers)
        else:  # if anomay is detected, it is a false positive
            cm['n_fp'] = int(n_outliers)
            cm['n_tn'] = int(n_observations - n_outliers)

        return cm

    def print_metrics(self, detector, scenario, perf, cm):
        """
        Prints the performance metrics "perf" (time to build the model and to make predictions)
        and the confusion matrix "cm" on the stdout for a given "detector" and "scenario"
        """
        print("Detector:\t\t\t", detector)
        print("Scenario:\t\t\t\t", scenario)
        print("Exec. time of model creation:\t", perf['tmc'], "seconds")
        print("Exec. time of model prediction:\t", perf['tmp'], "seconds")
        print("Confusion matrix:\t\n\n", cm)

    def metrics_to_csv(self, detector, scenario, perf, cm,results_csv_path):
        """
        Saves the performance metrics "perf" (time to build the model and to make predictions)
        and the confusion matrix "cm" on the csv file "result_csv_path" for a given "detector" and "scenario"
        """
        df = pd.DataFrame({
                           'detector': detector,
                           'scenario': scenario,
                           'time_model_creation': perf['tmc'],
                           'time_model_prediction': perf['tmp'],
                           'n_tp': cm['n_tp'],
                           'n_tn': cm['n_tn'],
                           'n_fp': cm['n_fp'],
                           'n_fn': cm['n_fn']},
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
        # Graph normalization
        grcomp.normalizeGraphs()

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
        observationsClassified = self.le.getLevel(observations)

        # Generates the time-evolving graphs
        graphs = self.generateTEG(observationsClassified, n_periods)

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
        observationsClassified = self.le.getLevel(observations)

        # Generates the time-evolving graphs
        graphs = self.generateTEG(observationsClassified, n_periods)

        # Computes the distance between each graph and the global graph
        graph_dist = np.empty(n_periods)
        for period in range(n_periods):
            graph_dist[period] = self.computeGraphDist(graphs[period].graph, global_graph, metric)

        return graph_dist

    def computeOutliers(self, baseline, prediction, sigLevel):
        """
        Computes the number of outliers based on the "baseline" distribution of the dissimilarities,
        the "prediction" and the significance level "sigLevel"
        """
        perc = np.percentile(baseline, sigLevel)

        # Sets a counter vector to zero
        n_out = np.zeros(prediction.size)
        # Dissimilarity tests
        n_out = np.where(prediction > perc, n_out + 1, n_out)

        return np.sum(n_out)
