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
|--------------------- |--------------------------------------------------------------------------------------------------------------------------------   |
| metric: string            | Dissimilarity metric used to compare two graphs. Input parameter.                              |
| n_bins: int            | Level of discretization of real valued observations (number of levels). Input parameter. Default value=  _N_BINS    |
| alpha: int             |  Significance level. Input parameter. Default value=  _ALPHA   |
| n_obs_per_period: int  | Number of observation per period. Input parameter. Default value= _N_OBS_PER_PERIOD                                            |


| attribute  (private)       | description                                                                                       |
|---------------------- |------------------------------------------------------------------------------------------------   |
| _N_OBS_PER_PERIOD: int  | Number of observations per period. Value=336    |
| _N_BINS: int            | Level of discretization of real valued observations (number of levels). Value=30.     |
| _ALPHA: int             | Significance level. Value=5.      |
| _baseline:  numpy array of float        | Baseline distribution of the training period. 				|
| _global_graph: Graph	   | Global graph associated to the training period. 				|


| method            					  |    description														|
|----------------------------------------------------------------------------------------------- |----------------------------------------------------------------------------------   |
| \_\_init__(metric: string, n_bins: int =_N_BINS, alpha: int =_ALPHA, n_obs_per_period: int =_N_OBS_PER_PERIOD)	| Constructor that initializes the TEG input parameters		|
| get_training_dataset(train_ds_path: string): DataFrame 	| Loads the training dataset from ```train_ds_path``` csv file and returns it as a   ```pandas``` Dataframe			|
| build_model(training_dataset: Dataframe): TEGdetector, float	|  Builds the prediction model based on the ```training_dataset``` and returns it together with the time to build the model          |
| get_testing_dataset(test_ds_path: string): DataFrame	| Loads the testing dataset from ```test_ds_path``` csv file and returns it as a   ```pandas``` Dataframe			|
| predict(testing_dataset: Dataframe, model: TEGDetector): int, int, float		| Makes predictions on the ```testing_dataset``` using the model. It returns three values: number of outliers and total number of observations (int type), and the time to make predictions (float type)		|
| compute_confusion_matrix(testing_len: int, predictions: int, is_attack_behavior: bool): dict |	 Computes the confusion matrix based on the total number of observations ```testing_len```, number of outliers ```predictions```and the type of scenario (boolean parameter indicating whether the testing dataset represents an attack scenario or not). It returns the confusion matrix as a dictionary type. __NOTE: The testing dataset can be either a normal scenario (i.e., no attacks) or an attack scenario (all the observations are attacks)__		|
| print_metrics(detector: string, scenario: string, perf: dict, cm: dict)		|  Prints the performance metrics  ```perf```(dict type including the time to build the model and the time to make predictions) and the confusion matrix ```cm```  on the standard output. The first two parameters to be provided are  the names of the  ```detector``` and the ```scenario```, respectively.		|
| metrics_to_csv(detector: string, scenario: string, perf: dict, cm: dict, results_csv_path: string)	| Save the performance metrics  ```perf```(dict type including the time to build the model and the time to make predictions) and the confusion matrix ```cm``` (dict type) print on the csv file ```results_csv_path```. The first two parameters to be provided are  the names of the  ```detector``` and the ```scenario```, respectively.
 
- ```TEGdetector```  class 

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| le 			| LevelExtractor instance	|
 
| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(usages: Dataframe, n_bins: int) | Constructor that initializes the TEGdetector based on the training dataset ```usage``` and ```n_bins``` |
| sumGraph(gr1: Graph, gr2: Graph)	| Adds to graph ```gr1``` the graph ```gr2```. __Pre-Condition: ```gr1``` nodes set includes the ```gr2```node set__ 	|
| getGlobalGraph(graphs: list of Graph): Graph | Creates and returns a *global graph* as the sum of a list of ```graphs```  	|
| generateTEG(usagesClassified: numpy array of int, n_periods: int): list of Graph | Generates and returns the time evolving graph series from the discretized observations ```usagesClassified```  and the number of periods ```n_period```	|
| computeGraphDist(gr1: Graph, gr2: Graph, metric: string): float | Computes and returns the distance  between two graphs ```gr1``` and ```gr2``` using the dissimilarity ```metric```	|
| buildModel(metric: string, usages: Dataframe, n_periods: int): numpy array of float, Graph | Builds the distribution of the dissimilarities based on the ```metric```, observation set ```usages``` and number of periods ```n_periods```. It returns the distribution of the dissimilarities  and the global graph	|
| makePrediction(baseline: numpy array of float, global_graph: Graph, metric: string, usages: Dataframe, n_periods): numpy array of float | Makes the predictions of the observation set ```usages```  based on the ```baseline``` distribution of the dissimilarities, the ```global graph```, the dissimilarity ```metric``` and the number of periods ```n_periods``` 	|
| computeOutliers(baseline: numpy array of float, prediction: numpy array of float, sigLevel: int): int | Computes the number of outliers based on the ```baseline``` distribution of the dissimilarities, the ```prediction```  and the significance level ```sigLevel``` |

- ```LevelExtractor``` class

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| level: numpy array of int 		| Discretization levels	|
| step: int					| Discretization step	|
| minValue	: float			| Minimum value of the training observation set	|
 
| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(minValue: float, step: int, n_bins: int) |	Constructor that initializes the LevelExtractor 	|
| getLevel(usages: Dataframe): numpy array of int	| Discretizes the real valued observations of ```usages``` according to the discretization levels and returns the discretized usages	|
                                                                                
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

 