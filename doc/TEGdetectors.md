# TEG detectors implementation

The ```tegdet``` package includes the  modules that implements the TEG detectors:
- ```teg.py```: it is the main module that includes the API;
- ```graph_comparison.py```: it includes the graph class and it is responsible to compute the difference between two graphs according to a given metric (variant of strategy pattern). 

The ```tegdet``` package depends on several well-known ```Python``` packages as shown in the diagram below:

 <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/packageOverview_v2.png" width="500">

Each module includes a set of classes, which are detailed in the class diagram below,
where the colour is used to map the classes to the module they belong to:

 <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/tegdetCD_v2.png" width="1000">


## teg module
The teg module includes the following classes:

- ```TEGDetector```: the API class to be used from the user point of view.

| attributes  (class)       | description                                                                                       |
|---------------------- |------------------------------------------------------------------------------------------------   |
| \_\_N_BINS: int            | Level of discretization of real valued observations (number of levels). Value=30.     |
| \_\_N_OBS_PER_PERIOD: int   | Number of observations per period. Value=336    |
| \_\_ALPHA: int             | Significance level 100-_ALPHA. Value=5.      |


| attributes  | description                                                                                       |
|--------------------- |--------------------------------------------------------------------------------------------------------------------------------   |
| \_\_metric: string            | Dissimilarity metric used to compare two graphs. Input parameter.                              |
| \_\_n_bins: int            | Level of discretization of real valued observations (number of levels). Input parameter. Default value=  \_\_N_BINS    |
| \_\_n_obs_per_period: int 	 | Number of observation per period. Input parameter. Default value= \_\_N_OBS_PER_PERIOD                                            |
| \_\_alpha: int             |  Significance level 100-alpha. Input parameter. Default value=  \_\_ALPHA   |


| method            					  |    description														|
|----------------------------------------------------------------------------------------------- |----------------------------------------------------------------------------------   |
| \_\_init__(metric: string, n_bins: int =\_\_N_BINS,  n_obs_per_period: int =\_\_N_OBS_PER_PERIOD, alpha: int =\_\_ALPHA)	| Constructor that initializes the TEGDetector input parameters	(see the attribute table)	|
| get_dataset(ds_path: string): DataFrame 	| Loads the dataset from ```ds_path``` file (comma-separated value format), renames the columns and returns it as a   ```pandas Dataframe```			|
| build_model(training_dataset: Dataframe): ModelBuilder, float	|  Builds the prediction model based on the ```training_dataset``` and returns it together with the time to build the model (```float``` type)        |
| predict(testing_dataset: Dataframe, model: ModelBuilder): numpy array of int, int, float		| Makes predictions on the ```testing_dataset``` using the ```model```. It returns: the outliers (```numpy``` array of {0,1} values) and total number of observations (```int``` type), and the time to make predictions (```float``` type)		|
| compute_confusion_matrix(ground_true: numpy array of int, predictions: numpy array of int): dict |	 Computes the confusion matrix based on the ground true values and predicted values (```numpy``` array of {0,1} values). It returns the confusion matrix as a dictionary (```dict```) type |
| print_metrics(detector: dict, testing_set: string, perf: dict, cm: dict)		|  Prints on the stdout:  the  ```detector```  (```dict```type including the metric and the input parameters setting), and the ```testing_set```, the performance metrics  ```perf```(```dict``` type including the time to build the model and the time to make predictions) and the confusion matrix ```cm``` 		|
| metrics_to_csv(detector: dict, testing_set: string, perf: dict, cm: dict, results_csv_path: string)	| Saves in the file with pathname ```results_csv_path``` (comma-separated values format):  the  ```detector``` (```dict```type), the ```testing_set```,  the performance metrics  ```perf``` (```dict``` type) and the confusion matrix ```cm``` (```dict``` type)
 
- ```ModelBuilder```: the builder of the prediction model, based on TEG and baseline graph dissimilarity distribution 


| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   | 
| \_\_obs: Dataframe    | Training set  |
| \_\_le: LevelExtractor 			| LevelExtractor instance	|
| \_\_baseline:  numpy array of float        | Baseline distribution of the training period.        |
| \_\_global_graph: Graph    | Global graph associated to the training period.        |

 
| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(observations: Dataframe, n_bins: int) | Constructor that initializes the ModelBuilder based on the training dataset ```observations``` and ```n_bins``` |
| \_\_sum_graphs(gr1: Graph, gr2: Graph)	| Adds to graph ```gr1``` the graph ```gr2```. __Pre-Condition: ```gr1``` nodes set includes the ```gr2```node set__ 	|
| \_\_compute_global_graph(graphs: list of Graph): Graph | Creates and returns a *global graph* as the sum of a list of ```graphs```    |
| get_level_extractor(): Dataframe | Returns ```__le``` |
| get_baseline(self): numpy array of float | Returns ```__baseline``` |
| get_global_graph(): Graph  | Returns ```__global_graph``` |
| build_model(metric: string, n_periods: int) | Computes and sets ```__global_graph``` and ```__baseline``` based on the ```metric```, number of periods ```n_periods``` and ```__obs```. 	|


- ```AnomalyDetector```: it makes predictions and computes outliers

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   | 
| \_\_model: ModelBuilder | ModelBuilder instance |

| method                        |    description                            |
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(model: ModelBuilder) | Constructor that initializes  ```model``` with the reference ModelBuilder|
| make_prediction(metric: string, observations: Dataframe, n_periods): numpy array of float | Makes the predictions of the ```observations```  based on the dissimilarity ```metric```, the number of periods ```n_periods``` and the reference ```__model```  |
| compute_outliers(prediction: numpy array of float, sigLevel: int): numpy array of int | Computes the outliers based on the ```prediction```,  the significance level ```sigLevel``` and the reference ```model``` (concretely, the ```__baseline``` distribution) |



- ```LevelExtractor```: extractor of levels and univariate time-series discretizer

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| \_\_level: numpy array of int 		| Discretization levels	|
| \_\_step: int					| Discretization step	|
| \_\_minValue	: float			| Minimum value of the training observation set	|
 
| method            								  |    description														|
|------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------------------------   |
| \_\_init__(minValue: float, step: int, n_bins: int) |	Constructor that initializes the LevelExtractor attributes	|
| get_levels(): numpy array of int | Returns  ```level``` |
| discretize(observations: Dataframe): numpy array of int	| Discretizes the real valued ```observations``` according to the discretization levels and returns the discretized observations	|

- ```TEGGenerator```: Time-Evolving-Graph generator

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| \_\_teg: list of Graph     | Time evolving graph |

| method                              |    description                            |
|------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------------------------   |
| \_\_init__(observation_discretized: numpy array of int, n_periods: int) | Generates and sets the ```__teg``` from the discretized observations ```observation_discretized``` and the number of periods ```n_periods``` |
| get_teg(): list of Graph | Returns the generated ```__teg``` |


- ```GraphDistanceCollector```: Collector of distances between graphs in a TEG and the global graph

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| \_\_distance: numpy array of float | Graph distances |

| method                              |    description                            |
|------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------------------------   |
| \_\_init__(n_periods: int) | Constructor, sets the ```__distance``` attributes as an  ```n_periods``` length empty array |
| compute_graphs_dist(teg: list of Graph, global_graph: Graph, metric: string): numpy array of float | Computes and returns the distances  between each graph in  ```teg``` and ```global_graph``` using the dissimilarity ```metric```  |


## graph comparison module
This module includes the following classes:


- ```Graph```: Graph generator (empty graph, graph from an univariate time-series) and manipulator (graph expansion)

| attribute 						        | description                                                                                       |
|------------------------------------------------------ |------------------------------------------------------------------------------------------------   |
| \_\_nodes: numpy array of int 			|  nodes of the graph	|
| \_\_nodes_freq: numpy array of int			|  node frequencies	|
| \_\_matrix: numpy array of int			|  adjacency-frequency matrix	|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(nodes=None, nodes_freq=None, matrix=None ) 		|	Constructor that initializes the Graph attributes (possible empty graph) 	|
| \_\_get_index(element: int): int       | Returns the index (int type) of the matrix row/column based on ```element``` |
| get_nodes() | Returns ```__nodes``` |
| get_nodes_freq() | Returns ```__nodes_freq``` |
| get_matrix() | Returns ```__matrix``` |
| update_node_freq(pos: int, value: int) | Increments by ```value``` the ```pos``` element of ```__nodes_freq``` |
| update_matrix_entry(row: int , col: int, value: int) | Increments by ```value``` the  ```__matrix``` entry in position ```row``` and ```col``` | 
| generate_graph(obs_discretized_: Dataframe)	| Generates the  ```graph``` from the discretized observations ```obs_discretized```  |
| expand_graph(position: int, vertex: int) | Expands the graph by inserting a new node ```vertex``` in ```position```. The new added fictious node has frequency -1. The new added row and column of the adjacency matrix have -1 entries |

- ```GraphComparator```: Graph comparator operator. It is the superclass (actually never instantiated).

| attribute        	 | description                                                                                       |
|-------------------------|------------------------------------------------------------------------------------------------   |
| \_graph1: Graph 	|  first operator	|
| \_graph2: Graph 	|  second operator	|

| method            					  |    description														|
|------------------------------------------------------------- |---------------------------------------------------------------------------------------------   |
| \_\_init__(gr1: Graph, gr2: Graph)				| Constructor that initializes the attributes as ```gr1``` and ```gr2```, respectively	|
| \_normalize\_matrices( )   | Converts the incidence matrices of graph1 and graph2 into one-dimensional array, and normalizes the entries (i.e., relative frequencies). |
| resize_graphs( )							| Compares the nodes of the two graphs and possibly expand them	|
| compare_graphs( )							| Signature only (it is overriden)	|

The rest of the classes are subclasses of ```GraphComparator``` that override the method ```compare_graphs()```. Each subclass ```Graph```__Metric__```Dissimilarity``` implements the dissimilarity metric included in the following Table. 

- The __Hamming__ metric is computed considering two vectors *P* and *Q* obtained by flattening the incidence matrices of the two graphs
- The __Cosine__ metric is computed considering two vectors *P* and *Q* obtained by flattening the node-frequency and the incidence matrices of the two graphs
- The rest of the metrics are computed considering two vectors *P* and *Q* obtained by normalizing the incidence matrices of the two graphs.

  <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/metricsTable.png" width="700">
