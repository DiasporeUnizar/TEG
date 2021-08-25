"""
@Author: Simona Bernardi
@Date: updated 25/04/2021

Graph comparison module:
Classes that enable to compare two graphs and compute the "difference" between them according to a 
given measure


"""

import numpy as np
from math import sqrt
from scipy.stats import entropy #it is used to compute the KLD measure
from scipy.spatial import distance #it is used to compute several distances

DEBUG = True


class GraphComparator:
    """Operator that compares two graphs"""

    def __init__(self,gr1,gr2):
        # First operand
        self.graph1 = gr1
        # Second operand
        self.graph2 = gr2
 
    def expandGraph(self, graph, position, vertex):
        # Different from zero to differentiate from the absence of arc,
        # but presence of the node
        wildcard = '-1'
        # Insert the new vertex in the list of nodes
        graph.nodes = np.insert(graph.nodes, position, vertex)
        graph.nodesFreq = np.insert(graph.nodesFreq, position, wildcard)
        # Insert the new column in the matrix
        graph.matrix = np.insert(graph.matrix, position, wildcard, axis=1)
        # Insert the new row in the matrix
        graph.matrix = np.insert(graph.matrix, position, wildcard, axis=0)

    def normalizeGraphs(self):
        first = self.graph1
        second = self.graph2

        # Union of the nodes
        nodesU = np.union1d(first.nodes, second.nodes)

        # Compare the node list and possibly extend the model(s)
        for i in range(nodesU.size):
            if (first.nodes.size > i) and (first.nodes[i] != nodesU[i]) or (
                    first.nodes.size <= i):
                self.expandGraph(first, i, nodesU[i])

        for i in range(nodesU.size):
            if (second.nodes.size > i) and (second.nodes[i] != nodesU[i]) or (
                    second.nodes.size <= i):
                self.expandGraph(second, i, nodesU[i])


    def compareGraphs(self):  # signature only because it is overriden
        return 0


# Strategy pattern (variant)


#######################################################################
# Graph Edit Distance (GED) family --> this family mostly used in TEG
#######################################################################
class GraphHammingDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: Structural-based distance (Hamming)
        The two matrix arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
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

    def compareGraphs(self):
        """
        Heuristic: Frequency-based dissimilarity (1-cosine)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Convert into arrays the node frequencies and matrices
        first = np.concatenate(
            (self.graph1.nodesFreq, self.graph1.matrix.flatten()), axis=None)
        second = np.concatenate(
            (self.graph2.nodesFreq, self.graph2.matrix.flatten()), axis=None)
        
        # Normalization factor
        nfactor = 1.0
        sp = first * second / nfactor
        # Frobenius norm (L2-norm Euclidean)
        norm1 = np.linalg.norm(first)
        norm2 = np.linalg.norm(second)
        # Compute the product
        cosinus = np.sum(sp) / (norm1 * norm2)

        #Since some entries of the matrices can be -1 the cosinus maybe be negative!
        return 1.0 - abs(cosinus) #returns the dissimilarity (distance)

class GraphJaccardDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Jaccard)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Jaccard similarity (equal to Peak Correlation Energy)
        sumprod = (first * second).sum()
        quadnorm1 = (first*first).sum()
        quadnorm2 = (second*second).sum()
        jac = sumprod / (quadnorm1 + quadnorm1 - sumprod)

        return 1.0 - jac #returns the dissimilarity (distance)

class GraphDiceDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Dice)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

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

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Kullback–Leibler)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the KLD of first  w.r.t second 
        kld = entropy(first,second,base=2)

        return kld  #returns the dissimilarity (distance)

class GraphJeffreysDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Jeffreys)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1  to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())


        # Compute the Jeffreys of first  w.r.t second 
        
        # Check possible case of division by 0 
        almostzero = np.finfo(float).eps
        zero = np.ones(len(second))*almostzero
        second = np.where((second == 0.0), zero, second)
        jef = np.divide(first, second)

        # Check possible case of log 0 
        jef = np.where((jef == 0.0), zero, jef)
        jef = np.log(jef) * (first - second)

        return jef.sum() #returns the dissimilarity (distance)

class GraphJSDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic:  PDF-based dissimilarity (Jensen-Shannon)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the JSD 
        jsd = distance.jensenshannon(first,second,base=2)

        return jsd  #returns the dissimilarity (distance)

###################################################################
# Minkowski family (4 different distance metrics)
###################################################################
class GraphEuclideanDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Euclidean)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Euclidean distance 
        eucl = distance.euclidean(first,second)

        return eucl #returns the dissimilarity (distance)

class GraphCityblockDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Cityblok)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Cityblock distance 
        city = distance.cityblock(first,second)

        return city  #returns the dissimilarity (distance)

class GraphChebyshevDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Chebyshev) 
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Chebyshev distance 
        cheb = distance.chebyshev(first,second)

        return cheb  #returns the dissimilarity (distance)

class GraphMinkowskiDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Minkowski distance with p=3)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Minkowski distance 
        mink = distance.minkowski(first,second,3)

        return mink  #returns the dissimilarity (distance)

###################################################################
# L_1 family (6 different distance metrics) 
###################################################################
class GraphBraycurtisDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Bray-Curtis, also called Sorensen)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
        graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Bray-Curtis distance 
        bray = distance.braycurtis(first,second)

        return bray  #returns the dissimilarity (distance)

class GraphGowerDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Gower)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Gower distance (=Cityblock divided by the number of elements)
        gower = distance.cityblock(first,second) / len(first)

        return gower  #returns the dissimilarity (distance)

class GraphSoergelDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Soergel)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Soergel distance (=Cityblock divided by the 
        # sum of the pairwise_max_elements)
        soergel = distance.cityblock(first,second) / np.maximum(first,second).sum()

        return soergel  #returns the dissimilarity (distance)

class GraphKulczynskiDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Kulczynski)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

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

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Canberra)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Canberra distance 
        canb = distance.canberra(first,second)

        return canb  #returns the dissimilarity (distance)

class GraphLorentzianDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Lorentzian)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Lorentzian distance 
        lore = np.log(1 + (first - second))

        return lore.sum()  #returns the dissimilarity (distance)

###################################################################
# Fidelity or Squared-chord family (4 different distance metrics) 
###################################################################
class GraphBhattacharyyaDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Bhattacharyya)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

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

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Hellinger)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Hellinger distance 
        helli = 1 - (np.sqrt(first * second)).sum()

        return  2.0 * sqrt(helli) #returns the dissimilarity (distance)

class GraphMatusitaDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Matusita)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Matusita distance 
        matu = (np.sqrt(first * second)).sum()

        return  sqrt(2.0 - 2.0 * matu) #returns the dissimilarity (distance)

class GraphSquaredchordDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Squared-chord)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Squared-chord distance (= Matusita without the square root)
        sqchord = (np.sqrt(first * second)).sum()

        return  2.0 - 2.0 * sqchord  #returns the dissimilarity (distance)

###################################################################
# Squared L_2 (or Chi^2) family (7 different distance metrics)
###################################################################

class GraphPearsonDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Pearson Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Check possible division by zero
        almostzero = np.finfo(float).eps
        zero = np.ones(len(second))*almostzero
        second = np.where((second == 0.0), second * almostzero, second)
  
        # Compute the Pearson Chi^2 distance
        num =  (first - second) 
        pearson = np.divide(num*num,second)
        
        
        return  pearson.sum() #returns the dissimilarity (distance)

class GraphNeymanDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Neyman Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Check possible division by zero
        almostzero = np.finfo(float).eps
        zero = np.ones(len(first))*almostzero
        first = np.where((first == 0.0), first * almostzero, first)
  
        # Compute the Neyman Chi^2 distance
        num =  (first - second) 
        neyman = np.divide(num*num,first)
        
        
        return  neyman.sum() #returns the dissimilarity (distance)

class GraphSquaredDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Squared Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        
        # Set -1 entries to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Squared Chi^2 distance
        num =  (first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0 
        squared = np.divide(num*num, den, out=np.zeros_like(den), where=(den!=0))
                
        return  squared.sum() #returns the dissimilarity (distance)

class GraphProbsymmetricDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Probabilistic symmetric Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        
        # Set -1 entries to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

       # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Probabistic symmetric Chi^2 distance
        num =  (first - second) 
        den = (first + second)

        # Check possible case of 0 / 0 
        probsym = np.divide(num*num, den, out=np.zeros_like(den), where=(den!=0))
        
        return  2.0 * probsym.sum() #returns the dissimilarity (distance)

class GraphDivergenceDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic:  PDF-based dissimilarity (Divergence)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        
        # Set -1 entries to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

       # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Divergence
        num =  (first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0
        diverg = np.divide(num*num, den*den, out=np.zeros_like(den), where=(den!=0))
                
        return  2.0 * diverg.sum() #returns the dissimilarity (distance)

class GraphClarkDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Clark)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        
        # Set -1 entries to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Clark distance
        num =  abs(first - second) 
        den = (first + second)
        
        # Check possible case of 0 / 0
        clark = np.divide(num,den, out=np.zeros_like(den), where=(den!=0))
        clark = (clark*clark).sum()
        
        return  sqrt(clark) #returns the dissimilarity (distance)

class GraphAdditivesymmetricDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic: PDF-based dissimilarity (Additive symmetric Chi^2)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        
        
        # Set -1 entries to  zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (PDF)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the Additive symmetric Chi^2 distance
        term1 =  (first - second) 
        term2 =  (first + second)
        num = term1 * term1 * term2
        den = (first * second)
        # Check possible case of 0 / 0
        addsym = np.divide(num,den, out=np.zeros_like(den), where=(den!=0))
        
        return  addsym.sum() #returns the dissimilarity (distance)
