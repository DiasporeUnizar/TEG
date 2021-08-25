# TEG detectors implementation

The ```tegdet``` package includes the  modules that implements the TEG detectors. The class diagram below  shows the structure of the implementation, there are three Python modules:
- ```TEG.py```: it is the main module including the yellow classes;
- ```graph_discovery.py```: it is responsible of graph generation and includes the green classes; and
- ```graph_comparison.py```: it is responsible to compute the difference between two graphs according to a given metric (variant of strategy pattern). It includes the blue classes.
 
 <img src="https://github.com/DiasporeUnizar/TEG/blob/master/doc/tegdet.png" width="1000">
