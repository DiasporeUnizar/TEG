"""
@Author: Simona Bernardi
@Date: 28/02/2022


Graph discovery module:
Classes that enable to generate a causal graph (node frequency list, adjacency-frequency matrix)
from the dataset

"""

import numpy as np

DEBUG = True


class Graph:
    """Graph structure"""

    def __init__(self):
        """
        Constructor that initializes the graph attributes to empty lists
        """
        # List of nodes 
        self.nodes = []
        # List of node frequencies
        self.nodesFreq = []
        # Adjacency-frequency matrix
        self.matrix = [[]]

 
class GraphGenerator:
    """Generator of a causal graph from the dataset"""

    def __init__(self):
        """
        Constructor that creates an empty graph
        """
        self.graph = Graph()

    def getIndex(self, element):
        """
        Returns the index of the matrix row (column) based on "element"
        """
        idx = -1  # not assigned
        i = 0
        nodes = self.graph.nodes
        while i < len(nodes) and idx == -1:
            if element == nodes[i]:
                idx = i
            i += 1

        return idx

    
    def generateGraph(self, obsClassified):
        """
        Generates the "graph" from the classified observations "obsClassified"
        """
        grouped = obsClassified.groupby('DP').count()
        # Sets vertices: they are ordered according to the levels 
        self.graph.nodes = grouped.index.to_numpy()          
        self.graph.nodesFreq = grouped.to_numpy()
        dim = len(self.graph.nodes)

        # Initializes the adjacent matrix
        self.graph.matrix = np.zeros((dim, dim), dtype=int)
        attr = obsClassified.DP.to_numpy()
        # Sets the adjacent matrix with the frequencies
        for i in range(attr.size - 1):
            row = self.getIndex(attr[i])
            col = self.getIndex(attr[i + 1])
            self.graph.matrix[row][col] += 1

