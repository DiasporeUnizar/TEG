"""
@Author: Simona Bernardi
@Date: 23/08/2021


Graph discovery module:
Classes that enable to generate a causal graph (node frequency list, adjacency-frequency matrix)
from the dataset

"""

import numpy as np

DEBUG = True


class Graph:
    """Graph structure"""

    def __init__(self):
        # List of nodes 
        self.nodes = []
        # List of node frequencies
        self.nodesFreq = []
        # Adjacency-frequency matrix
        self.matrix = [[]]

 
class GraphGenerator:
    """Generator of a causal graph from the dataset"""

    def __init__(self):
        # Graph data structure
        self.graph = Graph()

    def getIndex(self, element):
        idx = -1  # not assigned
        i = 0
        nodes = self.graph.nodes
        while i < len(nodes) and idx == -1:
            if element == nodes[i]:
                idx = i
            i += 1

        return idx

    #Pre: the eventlog is already filtered for a given ID
    def generateGraph(self, eventlog):
        grouped = eventlog.groupby('Usage').count()
        # Sets vertices: they are ordered according to the usage levels 
        self.graph.nodes = grouped.index.to_numpy()          
        self.graph.nodesFreq = grouped.to_numpy()
        dim = len(self.graph.nodes)

        # Initializes the adjacent matrix
        self.graph.matrix = np.zeros((dim, dim), dtype=int)
        usage = eventlog.Usage.to_numpy()
        # Sets the adjacent matrix with the frequencies
        for i in range(usage.size - 1):
            row = self.getIndex(usage[i])
            col = self.getIndex(usage[i + 1])
            self.graph.matrix[row][col] += 1

