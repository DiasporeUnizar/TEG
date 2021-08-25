# TEG detectors implementation

The ```tegdet``` package includes the  modules that implements the TEG detectors. The class diagram below  shows the structure of the implementation, there are three Python modules:
- ```TEG.py```: it is the main module including the yellow classes;
- ```graph_discovery.py```: it is responsible of graph generation and includes the green classes; and
- ```graph_comparison.py```: it is responsible to compute the difference between two graphs according to a given metric (variant of strategy pattern). It includes the blue classes.
 
 <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/tegdet.png" width="1000">

## TEG module
TEG module includes three classes:

- ```TEG``` is the API class to be used from the user point of view.

| attribute  (public)       | description                                                                                       |
|--------------------- |------------------------------------------------------------------------------------------------   |
| metric            | Dissimilarity metric used to compare two graphs.<br>Input parameter.                              |
| n_bins            | Level of discretization of real valued observations (number of levels). <br>Input parameter. Default value= _N_OBS_PER_PERIOD    |
| alpha             | Level of discretization of real valued observations (number of levels). <br>Input parameter. Default value= _N_BINS     |
| n_obs_per_period  | Number of observation per period.<br>Input parameter. Default value= _ALPHA                                             |

| attribute  (private)       | description                                                                                       |
|---------------------- |------------------------------------------------------------------------------------------------   |
| _N_OBS_PER_PERIOD  | Number of observations per period (e.g., number of observation per week).<br>Value=336    |
| _N_BINS            | Level of discretization of real valued observations (number of levels). <br>Value=30.     |
| _ALPHA             | Level of discretization of real valued observations (number of levels). <br>Value=5.      |
| _baseline         | Baseline distribution of the training period. Private attribute				|
| _global_graph	   | Global graph associated to the training period. Private attribute.				|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(metric, n_bins=_N_BINS, alpha=_ALPHA, n_obs_per_period=_N_OBS_PER_PERIOD)	| Constructor that initializes the TEG detector input parameters		|
| get_training_dataset(train_ds_path): DataFrame 	| Loads the training dataset from ```train_ds_path``` csv file and returns it as a   ```pandas``` Dataframe			|
| build_model(training_dataset): TEGdetector, float	|  Builds the prediction model based on the ```training_dataset``` (Dataframe type)  and returns it as ```TEGdetector``` together with the time to build the model (float type)         |
| get_testing_dataset(test_ds_path): DataFrame	| Loads the testing dataset from ```test_ds_path``` csv file and returns it as a   ```pandas``` Dataframe			|
| predict(testing_dataset, model): int, int, float		| Makes predictions on the ```testing_dataset``` using the model (TEGdetector type). It returns three values: number of outliers and total number of observations (int type), and the time to make predictions (float type)		|
| compute_confusion_matrix(testing_len, predictions, is_attack_behavior): dict |	 Computes the confusion matrix based on the total number of observations ```testing_len```, number of outliers ```predictions```and the type of scenario (boolean parameter indicating whether the testing dataset represents an attack scenario or not). It returns the confusion matrix as a dictionary type. __NOTE: The testing dataset can be either a normal scenario (i.e., no attacks) or an attack scenario (all the observations are attacks)__		|
| print_metrics(detector, attack, perf, cm)		|  Prints the performance metrics  ```perf```(dict type including the time to build the model and the time to make predictions) and the confusion matrix ```cm``` (dict type) print on the standard output. The first two parameters to be provided are:  the  dissimilarity metric ```detector``` (used to create the TEG detector, string type) and the name of the scenario ```attack``` (string type)		|
| metrics_to_csv(detector, attack, perf, cm, results_csv_path)	| Save the performance metrics  ```perf```(dict type including the time to build the model and the time to make predictions) and the confusion matrix ```cm``` (dict type) print on the csv file ```results_csv_path```. The first two parameters to be provided are:  the  dissimilarity metric ```detector``` (used to create the TEG detector, string type) and the name of the scenario ```attack``` (string type)				|

 
- ```TEGdetector```  class 

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| le 			| LevelExtractor instance	|
 
| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(usages, n_bins) |	|
| sumGraph(gr1, gr2)	|	|
| getGlobalGraph(graphs) | 	|
| generateTEG(usagesClassified, n_periods) |	|
| computeGraphDist(gr1, gr2, metric) |	|
| buildModel(metric, usages, n_periods) |	|
| makePrediction(baseline, global_graph, metric, usages, n_periods) |	|
| computeOutliers(model, test, sigLevel) |	|

- ```LevelExtractor``` class

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| level 			| 	|
| step			| 	|
| minValue			|	|
 
| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(minValue, step, n_bins) |		|
| getLevel(usages)	|	|
                                                                                
## Graph discovery module
This module includes two classes that enable to generate a causal graph (node frequency list, adjacency-frequency matrix)
from the dataset:


- ```Graph```

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| nodes 			| 	|
| nodesFreq		| 	|
| matrix			|	|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__( ) 	|		|

- ```GraphGenerator```

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| graph 			| 	|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__( ) 	|		|
| getIndex(element)	|	|
| generateGraph(eventlog)		|	|

## Graph comparison module
This modules include classes enable to compare two graphs and compute the "difference" between them according to a 
given measure. 

- ```GraphComparator``` is the superclass (actually never instantiated)

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| graph1 			| 	|
| graph2 			| 	|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(gr1, gr2)	|		|
| expandGraph(graph, position, vertex)	|	|
| normalizeGraphs( )	|	|
| compareGraphs( )	|	|

- The rest of the classes are subclasses of ```GraphComparator``` that override the method ```compareGraphs()```. Each subclass implements a different similarity
metric according to the following Table.

