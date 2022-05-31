"""
@Author: Simona Bernardi
@Date: 31/05/2022

graph_comparison module Version 2.0.0

- Graph class enables to generate a causal graph (node frequency list, adjacency-frequency matrix)
from the dataset
- GraphComparator and sub-classes enable to compare two graphs and compute the "difference" between them according to a 
given measure

"""

import numpy as np

from abc import ABC, abstractmethod
from math import sqrt
from scipy.stats import entropy #it is used to compute the KLD measure
from scipy.spatial import distance #it is used to compute several distances

class Graph:
    """
    Graph generator and manipulator (graph expansion)
    """

    def __init__(self, nodes=None, nodes_freq=None, matrix=None):
        """
        Constructor that initializes the graph attributes
        """       
        self.__nodes = nodes          
        self.__nodes_freq = nodes_freq
        self.__matrix = matrix

    def get_nodes(self):
        return self.__nodes

    def get_nodes_freq(self):
        return self.__nodes_freq

    def get_matrix(self):
        return self.__matrix

    def update_node_freq(self, pos, value):
        self.__nodes_freq[pos] += value

    def update_matrix_entry(self, row, col, value):
        self.__matrix[row][col] += value

    def __get_index(self, element):
        """
        Returns the index of the matrix row (column) based on "element"
        """
        idx = -1  # not assigned
        i = 0
        while i < len(self.__nodes) and idx == -1:
            if element == self.__nodes[i]:
                idx = i
            i += 1

        return idx
   
    def generate_graph(self, obs_discretized):
        """
        Generates the graph from the discretized observations "obs_discretized"
        """
        grouped = obs_discretized.groupby('DP').count()
        # Sets vertices: they are ordered according to the levels 
        self.__nodes = grouped.index.to_numpy()          
        self.__nodes_freq = grouped.to_numpy()
        dim = len(self.__nodes)

        # Initializes the adjacent matrix
        self.__matrix = np.zeros((dim, dim), dtype=int)
        attr = obs_discretized.DP.to_numpy()
        # Sets the adjacent matrix with the frequencies
        for i in range(attr.size - 1):
            row = self.__get_index(attr[i])
            col = self.__get_index(attr[i + 1])
            self.__matrix[row][col] += 1

    def expand_graph(self, position, vertex):
        """
        Expands the graph by inserting a new node "vertex" in "position". The new added fictious node
        has frequency -1. The new added row and column of the adjacency matrix have -1 entries
        """
        # Different from zero to differentiate from the absence of arc, but presence of the node
        wildcard = '-1'
        # Insert the new vertex in the list of nodes
        self.__nodes = np.insert(self.__nodes, position, vertex)
        self.__nodes_freq = np.insert(self.__nodes_freq, position, wildcard)
        # Insert the new column in the matrix
        self.__matrix = np.insert(self.__matrix, position, wildcard, axis=1)
        # Insert the new row in the matrix
        self.__matrix = np.insert(self.__matrix, position, wildcard, axis=0)


class GraphComparator(ABC):
    """ 
    Graphs comparator operator (abstract class)
    """
    def __init__(self, first_graph, second_graph):
        """
        Constructor that initializes the two operands
        """
        self._graph1 = first_graph
        self._graph2 = second_graph

    def _normalize_matrices(self):
        """
        Flatten the matrices of the two graphs and normalize them
        """
        # Get the two matrices and convert them into arrays
        edges1 = self._graph1.get_matrix().flatten()
        edges2 = self._graph2.get_matrix().flatten()

        # Set -1 entries to zero
        edges1 = np.where((edges1 < 0), edges1 * 0, edges1)
        edges2 = np.where((edges2 < 0), edges2 * 0, edges2)

        # Normalizes the matrices (PDF)
        edges1 = edges1 / (edges1.sum())
        edges2 = edges2 / (edges2.sum())

        return edges1, edges2

    def resize_graphs(self):
        """
        Compare the nodes of the two graphs and possibly expand them
        """

        # Union of the nodes
        union = np.union1d(self._graph1.get_nodes(), self._graph2.get_nodes())

        # Compare the node list and possibly extend the graph(s)
        for i in range(union.size):
            nodes = self._graph1.get_nodes()
            if (nodes.size > i) and (nodes[i] != union[i]) or (nodes.size <= i):
                self._graph1.expand_graph(i, union[i])

        for i in range(union.size):
            nodes = self._graph2.get_nodes()
            if (nodes.size > i) and (nodes[i] != union[i]) or (nodes.size <= i):
                self._graph2.expand_graph(i, union[i])

    @abstractmethod
    def compare_graphs(self):  # signature only because it is overriden
        pass



#######################################################################
# Graph Edit Distance (GED) family --> this family mostly used in TEG
#######################################################################
class GraphHammingDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: Structural-based distance (Hamming)
        The two matrix arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self._graph1.get_matrix().flatten()
        second = self._graph2.get_matrix().flatten()
        # Just the adjacency matrix
        dim = min(len(first), len(second))
        # Setting a counter vector to zero
        counter = np.zeros(dim)
        # Count if both elements are either positive or zero
        counter = np.where(((first > 0) & (second > 0)) |
                           ((first == 0) & (second == 0)), counter + 1, counter)

        distance = 1.0 - np.sum(counter) / float(dim)

        return distance  #returns the dissimilarity (distance)

'''
Reference of the following implemented dissimilarity metrics:
- Cha, "Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions", 2007
- See also URL: https://cran.r-project.org/web/packages/philentropy/vignettes/Introduction.html
'''

####################################################################
# Inner product family (3 different distance metrics)
####################################################################

class GraphCosineDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: Frequency-based dissimilarity (1-cosine)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Convert into arrays the node frequencies and matrices
        first = np.concatenate(
            (self._graph1.get_nodes_freq(), self._graph1.get_matrix().flatten()), axis=None)
        second = np.concatenate(
            (self._graph2.get_nodes_freq(), self._graph2.get_matrix().flatten()), axis=None)
    
        # Normalization factor
        nfactor = 1.0
        sp = first * second / nfactor
        # Frobenius norm (L2-norm Euclidean)
        norm1 = np.linalg.norm(first)
        norm2 = np.linalg.norm(second)
        den = np.sum(sp)
        
        if den > 0:
            # Compute the product
            cosinus =  den / (norm1 * norm2)
        else:
            cosinus = 0

        #Since some entries of the matrices can be -1 the cosinus maybe be negative!
        return 1.0 - cosinus #returns the dissimilarity (distance)

class GraphJaccardDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Jaccard)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """
        # Matrices normalization
        first, second = self._normalize_matrices()
 
        # Compute the Jaccard similarity (equal to Peak Correlation Energy)
        sumprod = (first * second).sum()
        quadnorm1 = (first*first).sum()
        quadnorm2 = (second*second).sum()
        jac = sumprod / (quadnorm1 + quadnorm1 - sumprod)

        return 1.0 - jac #returns the dissimilarity (distance)

class GraphDiceDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Dice)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """
        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Dice similarity 
        sumprod = (first * second).sum()
        quadnorm1 = (first*first).sum()
        quadnorm2 = (second*second).sum()
        dice = (2 * sumprod) / (quadnorm1 + quadnorm1)

        return 1.0 - dice #returns the dissimilarity (distance)


####################################################################
# Shannon's entropy family 
####################################################################
class GraphKLDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Kullback–Leibler)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """
        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the KLD of first  w.r.t second 
        kld = entropy(first,second,base=2)

        return kld  #returns the dissimilarity (distance)

class GraphJeffreysDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Jeffreys)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """
        # Matrices normalization
        first, second = self._normalize_matrices()


        # Compute the Jeffreys of first  w.r.t second 
        
        # Check possible case of division by 0 
        almostzero = np.finfo(float).eps
        second = np.where( (second < almostzero), almostzero, second)

        # Check possible case of log 0 
        jef = np.where((first < almostzero), almostzero, first / second)
        jef = np.log(jef) * (first - second)

        return jef.sum() #returns the dissimilarity (distance)

class GraphJSDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic:  PDF-based dissimilarity (Jensen-Shannon)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """
        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the JSD 
        jsd = distance.jensenshannon(first,second,base=2)

        return jsd  #returns the dissimilarity (distance)

###################################################################
# Minkowski family (4 different distance metrics)
###################################################################
class GraphEuclideanDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Euclidean)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Euclidean distance 
        eucl = distance.euclidean(first,second)

        return eucl #returns the dissimilarity (distance)

class GraphCityblockDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Cityblok)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Cityblock distance 
        city = distance.cityblock(first,second)

        return city  #returns the dissimilarity (distance)

class GraphChebyshevDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Chebyshev) 
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Chebyshev distance 
        cheb = distance.chebyshev(first,second)

        return cheb  #returns the dissimilarity (distance)

class GraphMinkowskiDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Minkowski distance with p=3)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Minkowski distance 
        mink = distance.minkowski(first,second,3)

        return mink  #returns the dissimilarity (distance)

###################################################################
# L_1 family (6 different distance metrics) 
###################################################################
class GraphBraycurtisDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Bray-Curtis, also called Sorensen)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
        graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Bray-Curtis distance 
        bray = distance.braycurtis(first,second)

        return bray  #returns the dissimilarity (distance)

class GraphGowerDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Gower)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Gower distance (=Cityblock divided by the number of elements)
        gower = distance.cityblock(first,second) / len(first)

        return gower  #returns the dissimilarity (distance)

class GraphSoergelDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Soergel)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Soergel distance (=Cityblock divided by the 
        # sum of the pairwise_max_elements)
        soergel = distance.cityblock(first,second) / np.maximum(first,second).sum()

        return soergel  #returns the dissimilarity (distance)

class GraphKulczynskiDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Kulczynski)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Kulczynski distance (=Cityblock divided by the 
        # sum of the pairwise_min_elements)
        num = distance.cityblock(first,second)
        den = np.minimum(first,second).sum()
        
        #Check possible division by zero 
        almostzero = np.finfo(float).eps
        if den > almostzero:
            kulc =  num / den
        else:    
            kulc = num / almostzero

        return kulc  #returns the dissimilarity (distance)

class GraphCanberraDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Canberra)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Canberra distance 
        canb = distance.canberra(first,second)

        return canb  #returns the dissimilarity (distance)

class GraphLorentzianDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Lorentzian)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Lorentzian distance 
        lore = np.log(1 + (first - second))

        return lore.sum()  #returns the dissimilarity (distance)

###################################################################
# Fidelity or Squared-chord family (4 different distance metrics) 
###################################################################
class GraphBhattacharyyaDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Bhattacharyya)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Bhattacharyya distance 
        bhatta = (np.sqrt(first * second)).sum()
        
        #Check possible case of log(0) 
        almostzero = np.finfo(float).eps
        if bhatta > almostzero:
            bhatta = -1.0 * np.log(bhatta)
        else:   
            bhatta = -1.0 * np.log(almostzero)

        return  bhatta #returns the dissimilarity (distance)

class GraphHellingerDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Hellinger)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Hellinger distance 
        helli = 1 - (np.sqrt(first * second)).sum()

        return  2.0 * sqrt(helli) #returns the dissimilarity (distance)

class GraphMatusitaDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Matusita)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Matusita distance 
        matu = (np.sqrt(first * second)).sum()

        return  sqrt(2.0 - 2.0 * matu) #returns the dissimilarity (distance)

class GraphSquaredchordDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Squared-chord)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Squared-chord distance (= Matusita without the square root)
        sqchord = (np.sqrt(first * second)).sum()

        return  2.0 - 2.0 * sqchord  #returns the dissimilarity (distance)

###################################################################
# Squared L_2 (or Chi^2) family (7 different distance metrics)
###################################################################

class GraphPearsonDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Pearson Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        num =  (first - second) 
        num = num * num
        
        # Check possible division by zero
        almostzero = np.finfo(float).eps
        zero = 0.0
        second = np.where((second < almostzero), almostzero, second)       

        # Compute the Pearson Chi^2 distance
        pearson = np.where((num < almostzero), zero, num / second)
        
        
        return  pearson.sum() #returns the dissimilarity (distance)

class GraphNeymanDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Neyman Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        num =  (first - second) 
        num = num * num

        # Check possible division by zero
        almostzero = np.finfo(float).eps
        zero = 0.0
        first = np.where((first < almostzero),  almostzero, first)
  
        # Compute the Neyman Chi^2 distance
        neyman = np.where((num < almostzero), zero, num / first)
               
        return  neyman.sum() #returns the dissimilarity (distance)

class GraphSquaredDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Squared Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Squared Chi^2 distance
        num =  (first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0 
        squared = np.divide(num*num, den, out=np.zeros_like(den), where=(den!=0))
                
        return  squared.sum() #returns the dissimilarity (distance)

class GraphProbsymmetricDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Probabilistic symmetric Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Probabistic symmetric Chi^2 distance
        num =  (first - second) 
        den = (first + second)

        # Check possible case of 0 / 0 
        probsym = np.divide(num*num, den, out=np.zeros_like(den), where=(den!=0))
        
        return  2.0 * probsym.sum() #returns the dissimilarity (distance)

class GraphDivergenceDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic:  PDF-based dissimilarity (Divergence)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Divergence
        num =  (first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0
        diverg = np.divide(num*num, den*den, out=np.zeros_like(den), where=(den!=0))
                
        return  2.0 * diverg.sum() #returns the dissimilarity (distance)

class GraphClarkDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Clark)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Clark distance
        num =  abs(first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0
        clark = np.divide(num,den, out=np.zeros_like(den), where=(den!=0))
        clark = (clark*clark).sum()
        
        return  sqrt(clark) #returns the dissimilarity (distance)

class GraphAdditivesymmetricDissimilarity(GraphComparator):

    def compare_graphs(self):
        """
        Heuristic: PDF-based dissimilarity (Additive symmetric Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph resizing has to be done before!
        """

        # Matrices normalization
        first, second = self._normalize_matrices()

        # Compute the Additive symmetric Chi^2 distance
        term1 =  (first - second) 
        term2 =  (first + second)
        num = term1 * term1 * term2
        den = (first * second)
        # Check possible case of 0 / 0
        addsym = np.divide(num,den, out=np.zeros_like(den), where=(den!=0))
        
        return  addsym.sum() #returns the dissimilarity (distance)
