"""
@Author: Simona Bernardi
@Date: 1/03/2023

graph_comparison module 

- Graph class enables to generate a causal graph (node frequency list, adjacency-frequency matrix) from the dataset
- GraphComparator and sub-classes enable to compare two graphs and compute the "difference" between them according to a 
given measure

--> v1.0.1: Graph expansion: changed wildcard from -1 to 0 

"""

import numpy as np

from abc import ABC, abstractmethod
from math import sqrt
from scipy.stats import entropy #it is used to compute the KLD measure
from scipy.spatial import distance #it is used to compute several distances
from scipy.sparse import lil_matrix # it is used for sparse matrix

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
        if matrix is None:
            self.__matrix = lil_matrix((0, 0), dtype=int)  # Use sparse LIL matrix for initialization
        else:
            self.__matrix = lil_matrix(matrix)

    def get_nodes(self):
        return self.__nodes

    def get_nodes_freq(self):
        return self.__nodes_freq

    def get_matrix(self):
        # Convert the LIL matrix to CSR format for efficient operations
        return self.__matrix.tocsr()

    def update_node_freq(self, pos, value):
        self.__nodes_freq[pos] += value

    def update_matrix_entry(self, row, col, value):
        self.__matrix[row, col] += value

    def __get_index(self, element):
        """
        Returns the index of the matrix row (column) based on "element"
        """
        idx = -1  # not assigned
        for i, node in enumerate(self.__nodes):
            if element == node:
                idx = i
                break
        return idx

    def generate_graph(self, obs_discretized):
        """
        Generates the graph from the discretized observations "obs_discretized"
        """
        grouped = obs_discretized.groupby('DP').count()
        # Sets vertices: they are ordered according to the levels 
        self.__nodes = grouped.index.to_numpy()
        self.__nodes_freq = grouped.to_numpy().flatten()
        dim = len(self.__nodes)

        # Initialize the adjacency matrix as a sparse LIL matrix for efficient row operations
        self.__matrix = lil_matrix((dim, dim), dtype=int)
        attr = obs_discretized.DP.to_numpy()

        # Sets the adjacency matrix with the frequencies
        for i in range(len(attr) - 1):
            row = self.__get_index(attr[i])
            col = self.__get_index(attr[i + 1])
            self.update_matrix_entry(row, col, 1)

        # Once matrix construction is done, convert to a more memory-efficient format (CSR)
        self.__matrix = self.__matrix.tocsr()


    def expand_graph(self, position, vertex):
        """
        Expands the graph by inserting a new node "vertex" in "position".
        """
        wildcard = '0'
        # Insert the new vertex
        self.__nodes = np.insert(self.__nodes, position, vertex)
        self.__nodes_freq = np.insert(self.__nodes_freq, position, wildcard)

        # Resize the sparse matrix using lil_matrix
        new_size = len(self.__nodes)
        new_matrix = lil_matrix((new_size, new_size), dtype=int)
        new_matrix[:self.__matrix.shape[0], :self.__matrix.shape[1]] = self.__matrix
        self.__matrix = new_matrix

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
        Normalize the sparse matrices without converting to dense arrays.
        """
        edges1 = self._graph1.get_matrix()
        edges2 = self._graph2.get_matrix()

        # Normalize in sparse domain
        edges1_sum = edges1.sum()
        edges2_sum = edges2.sum()

        if edges1_sum > 0:
            edges1 = edges1.multiply(1 / edges1_sum)
        if edges2_sum > 0:
            edges2 = edges2.multiply(1 / edges2_sum)

        return edges1, edges2

    def resize_graphs(self):
        """
        Compare the nodes of the two graphs and possibly expand them
        """

        # Union of the nodes
        union = np.union1d(self._graph1.get_nodes(), self._graph2.get_nodes())

        for i, node in enumerate(union):
            if i >= len(self._graph1.get_nodes()) or self._graph1.get_nodes()[i] != node:
                self._graph1.expand_graph(i, node)
            if i >= len(self._graph2.get_nodes()) or self._graph2.get_nodes()[i] != node:
                self._graph2.expand_graph(i, node)

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

        # Get the matrices and ensure they're sparse
        first_matrix = self._graph1.get_matrix()
        second_matrix = self._graph2.get_matrix()

        # Sparse matrix element-wise comparison
        diff_matrix = first_matrix != second_matrix

        # Calculate the total number of differing elements
        distance = diff_matrix.nnz / first_matrix.shape[0]

        return distance  # Returns the dissimilarity (distance)

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
