"""
@Authors: Simona Bernardi, Ángel Villanueva
@Date: 13/09/2024

graph_comparison module 

- Graph class enables to generate a causal graph (node frequency list, adjacency-frequency matrix) from the dataset
- GraphComparator and sub-classes enable to compare two graphs and compute the "difference" between them according to a 
given measure

--> v1.0.1: Graph expansion: changed wildcard from -1 to 0 
--> v1.1.0: Major updates:
    - Code refactored to work with sparse matrices using `lil_array` for graph construction and `csr` format for comparison operations.
    - Removed the `get_index` and `expand_graph` methods from the `Graph` class.
    - Removed the `resize_graphs` method from the `GraphComparator` class

"""

import numpy as np

from abc import ABC, abstractmethod
from math import sqrt
from scipy.stats import entropy #it is used to compute the KLD measure
from scipy.spatial import distance #it is used to compute several distances
from scipy.sparse import lil_array # it is used for sparse matrix

class Graph:
    """
    Optimized Graph generator and manipulator (graph expansion)
    """

    def __init__(self, nodes=None, nodes_freq=None, matrix=None):
        """
        Constructor that initializes the graph attributes efficiently.
        """
        self.__nodes = nodes          
        self.__nodes_freq = nodes_freq
        self.__matrix = lil_array(matrix) if matrix is not None else lil_array((len(self.__nodes), len(self.__nodes)), dtype=int)

    def get_nodes(self):
        return self.__nodes

    def get_nodes_freq(self):
        return self.__nodes_freq

    def get_matrix(self):
        return self.__matrix

    def update_node_freq(self, pos, value):
        """
        Update the frequency of a node at a given position.
        """
        self.__nodes_freq[pos] += value

    def update_matrix(self, matrix):
        """
        Update the structure of the matrix
        """
        self.__matrix = matrix

    def generate_graph(self, obs_discretized):
        """
        Generates the graph from the discretized observations "obs_discretized".
        This version reduces unnecessary operations and optimizes the process of matrix filling.
        """
        # Extract node data and their frequency counts
        attr = obs_discretized['DP'].to_numpy()
        # Get unique values and counts
        values, counts = np.unique(attr, return_counts=True)
        self.__nodes = values
        self.__nodes_freq = counts

        dim = max(values)+1
        # Precompute row and column indices in one pass
        transitions = np.array([attr[i] * dim + attr[i + 1]
                                for i in range(len(attr) - 1)])

        # Count the transitions efficiently using bincount
        bincounts = np.bincount(transitions) #, minlength=dim * dim)
        
        nonzero_indices = np.nonzero(bincounts)[0]

        # Update the matrix using the counted transitions
        for idx in nonzero_indices:
            row, col = divmod(idx, dim)            
            self.__matrix[row, col] = bincounts[idx]

        # Convert to CSR after matrix construction
        self.__matrix.tocsr()

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
        # Get the two matrices in CSR format (sparse matrices)
        edges1 = self._graph1.get_matrix()  # CSR format
        edges2 = self._graph2.get_matrix()  # CSR format

        # Sum the entries of each matrix
        sum1 = edges1.sum()
        sum2 = edges2.sum()

        # Normalize the matrices by dividing each entry by the sum
        edges1 = edges1.multiply(1.0 / sum1)
        edges2 = edges2.multiply(1.0 / sum2)

        return edges1, edges2

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
        first = self._graph1.get_matrix().toarray().flatten()
        second = self._graph2.get_matrix().toarray().flatten()
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

        # Get node frequencies and matrices as sparse arrays
        freq1 = self._graph1.get_nodes_freq()
        freq2 = self._graph2.get_nodes_freq()
        
        mat1 = self._graph1.get_matrix()
        mat2 = self._graph2.get_matrix()

        # Ensure both graphs have the same number of nodes and resize them if needed
        min_size = min(len(freq1), len(freq2))

        # Truncate or pad node frequencies to the same size
        freq1 = freq1[:min_size]
        freq2 = freq2[:min_size]

        # Truncate or pad matrices to the same size and keep them sparse
        mat1 = mat1[:min_size, :min_size]
        mat2 = mat2[:min_size, :min_size]

        # Convert sparse matrices to arrays and flatten them
        first = np.concatenate((freq1, mat1.toarray().flatten()), axis=None)
        second = np.concatenate((freq2, mat2.toarray().flatten()), axis=None)

        # Normalization factor
        nfactor = 1.0

        # Compute dot product
        sp = first * second / nfactor

        # Frobenius norm (L2-norm Euclidean)
        norm1 = np.linalg.norm(first)
        norm2 = np.linalg.norm(second)
        den = np.sum(sp)

        if den > 0:
            # Compute the cosine similarity
            cosinus = den / (norm1 * norm2)
        else:
            cosinus = 0

        # Since some entries of the matrices can be -1, the cosine may be negative!
        return 1.0 - cosinus  # returns the dissimilarity (distance)

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

        # Compute the Euclidean distance 
        eucl = distance.euclidean(first, second)

        return eucl  # returns the dissimilarity (distance)

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

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

        # Convert into a dense format
        first = first.toarray().flatten()
        second = second.toarray().flatten()

        # Compute the Additive symmetric Chi^2 distance
        term1 =  (first - second) 
        term2 =  (first + second)
        num = term1 * term1 * term2
        den = (first * second)
        # Check possible case of 0 / 0
        addsym = np.divide(num,den, out=np.zeros_like(den), where=(den!=0))
        
        return  addsym.sum() #returns the dissimilarity (distance)
