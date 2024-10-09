"""
@Authors: Simona Bernardi, Raúl Javierre, Ángel Villanueva
@Date: 13/09/2024

teg module 
This modules includes the following classes:

- TEGDetector (API class)
- LevelExtractor
- TEGGenerator
- GraphDistanceCollector
- ModelBuilder
- AnomalyDetector

that implements the detectors based on Time Evolving Graph (TEG) and graph dissimilarity distribution.

v1.0.1:
---> added private attribute tegg to ModelBuilder and AnomalyDetector
---> added private attributes mb and ad to TEGDetector
v1.1.0:
---> added `n_bins` parameter to the `__init__` method of the `TEGGenerator` class to enable the creation of graphs with the same size as the global graph.
---> updated `sum_graphs` function in the `ModelBuilder` class to adapt it for sparse matrix operations.
---> added `n_bins` parameter to the `__init__` method of the `AnomalyDetector` class to generate graphs with the correct dimensions.
---> changed the `build_model` function in the `ModelBuilder` class to measure different times taking part in the construction of a model
---> changed the `print_metrics` and `metrics_to_csv` functions in the `TEGDetector` class to print a more exhaustive report
v1.1.1:
---> added the new class `Sliding Window` to perform the new training and test process
---> updated `TEGDetector`, adding the `update_mb` method to update model data and `ModelBuilder` class adding the `update_data` to update the data of the model
"""

from time import time
import numpy as np
import pandas as pd
import os
import sys   

from tegdet.graph_comparison import *


class TEGDetector():
    """ 
    API 
    """
    #Default values
    __N_BINS = 30
    __N_OBS_PER_PERIOD = 336
    __ALPHA = 5

    def __init__(self, metric, n_bins=__N_BINS, n_obs_per_period=__N_OBS_PER_PERIOD, alpha=__ALPHA):
        self.__metric = metric
        self.__n_bins = n_bins
        self.__n_obs_per_period= n_obs_per_period
        self.__alpha = alpha
        self.__mb = None
        self.__ad = None
        self.__sw = None

    def get_mb(self):
        return self.__mb

    def update_mb(self, model):
        self.__mb = model

    def get_ad(self):
        return self.__ad
    
    def get_sw(self):
        return self.__sw
    
    def get_dataset(self,ds_path):
        """
        Load the dataset from "ds_path" (csv file), rename the two columns as TS (timestamp) and 
        data points (DP), respectively, and return the pandas dataframe "dt"
        """
        df = pd.read_csv(ds_path)
        df.columns = ['TS','DP']
        return df
    
    def initialize_window(self, dataset):
        self.__sw.initialize_window(dataset)

    def slide_window(self, dataset):
        self.__sw.slide_window(dataset)
    
    def build_model(self, training_dataset):
        """
        Build the prediction model based on the "training_dataset" and return it together with
        the time to build the model
        """

        t0 = time()
        obs = training_dataset['DP']
        self.__mb = ModelBuilder(obs, self.__n_bins)
        time2graphs, time2global, time2metrics = self.__mb.build_model(self.__metric, int(len(training_dataset.index) / self.__n_obs_per_period))

        self.__sw = SlidingWindow(len(training_dataset) + self.__n_obs_per_period, self.__n_obs_per_period, self.__mb)

        return self.__mb, time() - t0, time2graphs, time2global, time2metrics
    
    def process_window(self, training_data, n_bins):
        # Time to process the current window
        t0 = time()
        self.__sw.process_window(self.__metric, int(len(training_data.index) / self.__n_obs_per_period), n_bins)
        return time() - t0

    def predict(self, testing_dataset, model):
        """
        Make predictions on the "testing_dataset" using the model. It returns three values:
        number of outliers, total number of observations, and time to make predictions
        """
        t0 = time()
        data_points = testing_dataset['DP']
        self.__ad = AnomalyDetector(model, np.arange(self.__n_bins+2))
        test = self.__ad.make_prediction(self.__metric, data_points, int(len(testing_dataset.index) / self.__n_obs_per_period))
        outliers = self.__ad.compute_outliers(test, 100 - self.__alpha)

        return outliers, int(len(testing_dataset.index) / self.__n_obs_per_period), time() - t0
        
    def compute_confusion_matrix(self, ground_true, predictions):
        """
        Pre: "ground_true" is a vector with the true values, "predictions" is a vector 
        with predicted values (0,1)
        Post: Computes the confusion matrix. It returns the confusion matrix as dictionary type.
        """
        cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        cm['tp'] = ((ground_true == 1) & (predictions == 1)).sum()
        cm['tn'] = ((ground_true == 0) & (predictions == 0)).sum()
        cm['fp'] = ((ground_true == 0) & (predictions == 1)).sum()
        cm['fn'] = ((ground_true == 1) & (predictions == 0)).sum()

        return cm

    def print_metrics(self, detector, testing_set, perf, cm):
        """
        Print on the stdout: the "detector" configuration, the "testing set", the performance metrics "perf" 
        (time to build the model and to make predictions) and the confusion matrix "cm" 
        """
        print("Detector:\t\t\t", detector['metric'])
        print("N_bins:\t\t\t\t", detector['n_bins'])
        print("N_obs_per_period:\t\t", detector['n_obs_per_period'])
        print("Alpha:\t\t\t\t", detector['alpha'])
        print("Testing set:\t\t\t", testing_set)
        print("Time to build the model:\t", perf['tmc'], "seconds")
        print("Time to generate all TEG:\t", perf['tmg'], "seconds")
        print("Time to compute global graph:\t", perf['tmgl'], "seconds")
        print("Time to compute all metrics:\t", perf['tmm'], "seconds")
        print("Time to make prediction:\t", perf['tmp'], "seconds")
        print("Time to process all windows:\t", perf['tmw'], "seconds")
        print("Confusion matrix:\t\n\n", cm)

    def metrics_to_csv(self, detector, testing_set, perf, cm,results_csv_path):
        """
        Save in the csv file "result_csv_path", the "detector" configuration, the  "testing set",
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
                           'time2graphs': perf['tmg'],
                           'time2global': perf['tmgl'],
                           'time2metrics': perf['tmm'],
                           'time2predict': perf['tmp'],
                           'time2window': perf['tmw'],
                           'tp': cm['tp'],
                           'tn': cm['tn'],
                           'fp': cm['fp'],
                           'fn': cm['fn']},
                           index=[0])

        df.to_csv(results_csv_path, mode='a', header=not os.path.exists(results_csv_path), index=False)


class LevelExtractor:
    """ 
    Extractor of levels, univariate time-series discretizer 
    """

    def __init__(self, min_value, step, n_bins):
        """
        Create levels [0,1,..,n_bins+2], the last two positions for the possible 
        outliers of the testing dataset (minimum than the min_train_value and maximum 
        of the max_train_value)
        """
        self.__levels = np.arange(n_bins+2)
        self.__step = step
        self.__min_value = min_value

    def get_levels(self):
        return self.__levels

    def discretize(self, observations):
        """
        Discretization of  "observations" according to the "self.levels"
        "observations" is a np.array (of floats)
        """
        n_obs = len(observations)  # number of observations
        discretized_obs = -1 * np.ones(n_obs)  # array initialized with -1
        discretized_obs = discretized_obs.astype(int)  # np.array of int

        # Case: "observations" (testing set) is lower than the min_train_value
        #       level position the last
        discretized_obs = np.where((observations < self.__min_value), self.__levels[-1], discretized_obs)
        
        n_bins= len(self.__levels)-2
        i = 0  # while iterator
        while i < n_bins:
            lower = self.__min_value + i * self.__step
            upper = self.__min_value + (i + 1) * self.__step
            discretized_obs = np.where((lower <= observations) & (observations < upper),
                             self.__levels[i], discretized_obs)
            i += 1

        # Case: "observations" (testing set) is greater than the max_train_value 
        # level position the penultimate      
        discretized_obs = np.where(upper <= observations, self.__levels[-2], discretized_obs)

        return discretized_obs


class TEGGenerator:
    """
    Time Evolving Graph generator
    """

    def __init__(self, observations_discretized, n_periods, n_bins):
        """
        Generates and set the time evolving graphs series from the discretized observations
        "observation_discretized" and the number of periods "n_periods"
        """
        n_obs = int(len(observations_discretized) / n_periods)  # number of observations per period
        self.__teg = []
        for period in range(n_periods):
            obs_discr_period = observations_discretized[period * n_obs:(period + 1) * n_obs]
            # Transforms to a dataframe (needed to generate the graph)
            df = pd.DataFrame({'Period': period * np.ones(n_obs), 'DP': obs_discr_period})
            gr = Graph(np.arange(n_bins, dtype=int), np.zeros((n_bins), dtype=int), np.zeros((n_bins, n_bins), dtype=int))     
            gr.generate_graph(df)
            self.__teg.append(gr)

    def get_teg(self):

        return self.__teg


class GraphDistanceCollector:
    """
    Collector of distances between graphs in teg and the global graph
    """

    def __init__(self, n_periods):

        self.__distance = np.empty(n_periods)
 
    def compute_graphs_dist(self, teg, global_graph, metric):
        """
        Compute and return the distances between  each graph in "teg" and the "global_graph"
        using the dissimilarity "metric"
        """
        gc_name = "Graph" + metric + "Dissimilarity"

        for period in range(self.__distance.size):
            #Create an instance of the metric specific graph comparator
            #by converting a string to a class object
            gc = getattr(sys.modules[__name__], gc_name)(teg[period], global_graph)

            #Compute the difference based on the metric
            self.__distance[period] = gc.compare_graphs()
        
        return self.__distance

class ModelBuilder:
    """
    Prediction model builder based on teg and baseline graph dissimilarity distribution
    """

    def __init__(self, observations, n_bins):
        """
        Constructor that initializes attributes based on the training dataset "observations" and "n_bins"
        Creation of a new level extractor.
        """
        m = observations.min()  # min of the TRAINING dataset
        M = observations.max()  # max of the TRAINING dataset
        step = (M - m) / n_bins  # usage increment step
        self.__obs = observations
        self.__le = LevelExtractor(m, step, n_bins)
        self.__tegg = None
        self.__baseline = None
        self.__global_graph= None

    def get_level_extractor(self):
        return self.__le

    def get_baseline(self):
        return self.__baseline

    def get_global_graph(self):
        return self.__global_graph

    def get_tegg(self):
        return self.__tegg

    def __sum_graphs(self, gr1, gr2):
        """
        Pre: Graph "gr1" nodes set includes graph "gr2" nodes set
        Post: Added the graph "gr2" to graph "gr1" by summing their adjacency matrices directly.
        """
        # Sum the frequency of the nodes
        nodes = gr2.get_nodes()
        nodes_freq = gr2.get_nodes_freq()

        for i, node in enumerate(nodes):
            gr1.update_node_freq(node, nodes_freq[i])

        # Get the matrix in CSR format
        matrix1 = gr1.get_matrix()
        matrix2 = gr2.get_matrix()

        # Add the matrix to the global graph
        gr1.update_matrix(matrix1 + matrix2)

    def __compute_global_graph(self, graphs):
        """
        Create and return a global graph as the sum of a list of "graphs".
        """
        n_bins = len(self.__le.get_levels())  
        self.__global_graph = Graph(np.arange(n_bins, dtype=int), np.zeros((n_bins), dtype=int), np.zeros((n_bins, n_bins), dtype=int))      
        
        for gr in graphs:
            self.__sum_graphs(self.__global_graph, gr)


    def build_model(self, metric, n_periods):
        """
        Build the distribution of the dissimilarities based on "metric", "observations", and
        "n_periods". It returns the ditribution of the dissimilarities and the global graph.
        """

        # Get observation levels (discretization)
        obs_discretized = self.__le.discretize(self.__obs)

        # Measure time when generating TEGs
        time2graphs = time()

        # Get the time-evolving graphs
        n_bins = len(self.__le.get_levels())
        self.__tegg = TEGGenerator(obs_discretized, n_periods, n_bins)
        graphs = self.__tegg.get_teg()
        time2graphs = time() - time2graphs

        # Measure time when generating the global graph
        time2global = time()

        # Get the global graph of the training period
        self.__compute_global_graph(graphs)
        time2global = time() - time2global

        # Measure time when computing dissimilarity metrics
        time2metrics = time()

        # Get the graph dissimilarity baseline distribution
        gdc = GraphDistanceCollector(n_periods)
        self.__baseline = gdc.compute_graphs_dist(graphs, self.__global_graph, metric)
        time2metrics = time() - time2metrics

        return time2graphs, time2global, time2metrics
    
    def update_data(self, global_graph, tegs, baseline):
        self.__global_graph = global_graph
        self.__tegg = tegs
        self.__baseline = baseline


class SlidingWindow:
    """
    Sliding Window model based on a incremental change of the data inside
    """

    def __init__(self, window_size, step_size, model:ModelBuilder):
        """
        Constructor that initializes attributes based on the dataset used.
        """
        self.__window_size = window_size # Window size (Number of observations)
        self.__step_size = step_size # Number of observations the window scrolls through
        self.__current_window = None
        self.__current_position = 0
        self.__global_graph = model.get_global_graph()
        self.__graphs = model.get_tegg().get_teg()
        self.__mb = model

    def initialize_window(self, dataset):
        """
        Initializes the window given a dataset
        """
        self.__current_window = dataset.iloc[:self.__window_size] # Initial dataset + window size
        self.__current_position = self.__window_size

    def slide_window(self, dataset):
        """
        Slides the window to a next observation given a dataset
        """
        if self.__current_position + self.__step_size <= len(dataset):
            self.__current_window = dataset.iloc[self.__current_position:self.__current_position + self.__window_size]
            self.__current_position += self.__step_size
        else:
            # if there isn't enough data
            self.__current_window = None
        return self.__current_window
    
    def __update_data(self, old, new, global_graph):
        """
        Updates the data of the model given the old and new graphs
        """
        # Sum the frequency of the new nodes
        nodes_new = new.get_nodes()
        nodes_freq_new = new.get_nodes_freq()

        for i, node in enumerate(nodes_new):
            global_graph.update_node_freq(node, nodes_freq_new[i])

        # Substract the frecuency of the old nodes
        nodes_old = old.get_nodes()
        nodes_freq_old = old.get_nodes_freq()

        for i, node in enumerate(nodes_old):
            global_graph.update_node_freq(node, - nodes_freq_old[i])

        # Get the matrix in CSR format
        old = old.get_matrix()
        new = new.get_matrix()
        global_matrix = global_graph.get_matrix()

        # Compute the new global_graph
        global_matrix = ( global_matrix - old ) + new

        # Add the matrix to the global graph
        global_graph.update_matrix(global_matrix)

    def process_window(self, metric, n_periods, n_bins):
        """
        Process the actual window adding the new observations and eliminates the old data
        """
        if self.__current_window is not None:

            obs_discretized = self.__mb._ModelBuilder__le.discretize(self.__mb._ModelBuilder__obs)
            n_obs = int(len(obs_discretized) / n_periods)
            # Get the actual global graph and the old data
            old = self.__graphs.pop(0)

            # Generate the new graph data
            period = len(self.__graphs)
            obs_discr_period = obs_discretized[period * n_obs:(period + 1) * n_obs]
            df = pd.DataFrame({'Period': period, 'DP': obs_discr_period})
            new = Graph(np.arange(n_bins, dtype=int), np.zeros((n_bins), dtype=int), np.zeros((n_bins, n_bins), dtype=int))     
            new.generate_graph(df)
            self.__graphs.append(new)

            self.__update_data(old,new, self.__global_graph)

            gdc = GraphDistanceCollector(n_periods)
            baseline = gdc.compute_graphs_dist(self.__graphs, self.__global_graph, metric)

            # Update data from the model
            self.__mb.update_data(self.__global_graph, self.__graphs, baseline)
        else:
            raise ValueError("There is no window specified")


class AnomalyDetector:
    """
    Make predictions and compute outliers 
    """

    def __init__(self, model, n_bins):
        self.__model = model
        self.__tegg = None
        self.__n_bins = n_bins

    def get_tegg(self):
        return self.__tegg

    def make_prediction(self, metric, observations, n_periods):
        """
        Make and return the predictions of the "observations" based on the "baseline" distribution 
        of the dissimilarities, "global_graph", the dissimilarity "metric" and "n_periods"
        """

        # Gets consumption levels (consumption discretization)
        obs_discretized = self.__model.get_level_extractor().discretize(observations)

        # Generates the time-evolving graphs
        self.__tegg = TEGGenerator(obs_discretized, n_periods, len(self.__n_bins))
        graphs = self.__tegg.get_teg()
 
        # Computes the distance between each graph and the global graph        
        global_graph = self.__model.get_global_graph()
        gdc = GraphDistanceCollector(n_periods)
        graph_dist = gdc.compute_graphs_dist(graphs, global_graph, metric)
        
        return graph_dist

    def compute_outliers(self, prediction, sigLevel):
        """
        Compute the outliers based on the baseline graph dissimilarity distribution,
        the "prediction" and the significance level "sigLevel"
        """
        baseline = self.__model.get_baseline()
        perc = np.percentile(baseline, sigLevel)

        # Sets a counter vector to zero
        n_out = np.zeros(prediction.size)
        # Dissimilarity tests
        n_out = np.where(prediction > perc, n_out + 1, n_out)

        return n_out




