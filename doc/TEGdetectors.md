# TEG detectors implementation

The ```tegdet``` package includes the  modules that implements the TEG detectors. The class diagram below  shows the structure of the implementation, there are three Python modules:
- ```TEG.py```: it is the main module including the yellow classes;
- ```graph_discovery.py```: it is responsible of graph generation and includes the green classes; and
- ```graph_comparison.py```: it is responsible to compute the difference between two graphs according to a given metric (variant of strategy pattern). It includes the blue classes.
 
 <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/tegdet.png" width="1000">

## TEG module
TEG module includes three classes:

- ```TEG``` is the main class to be used from the user point of view.

| attribute         | description                                                                                       |
|------------------ |------------------------------------------------------------------------------------------------   |
| N_OBS_PER_PERIOD  | Number of observations per period (e.g., number of observation per week).<br>Default value=336    |
| N_BINS            | Level of discretization of real valued observations (number of levels). <br>Default value=30.     |
| ALPHA             | Level of discretization of real valued observations (number of levels). <br>Default value=30.     |
| metric            | Dissimilarity metric used to compare two graphs.<br>Input parameter.                              |
| n_bins            | Level of discretization of real valued observations (number of levels). <br>Input parameter.      |
| alpha             | Level of discretization of real valued observations (number of levels). <br>Input parameter.      |
| n_obs_per_period  | Number of observation per period.<br><br>Input parameter.                                         |

## Graph discovery module


## Graph comparison module