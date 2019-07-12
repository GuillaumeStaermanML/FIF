""" Functional Isolation Forest

    Author : Guillaume Staerman
"""


"""Functional Isolation Forest Algorithm

This is the implementation of The Functional Isolation Forest which is an
extension of the original Isolation Forest applied to functional data.

It return the anomaly score of each sample using the FIF algorithm.
The Functional Isolation Forest 'isolates' observations by 
randomly selecting a curve among a dictionary
and then randomly selecting a split value between the maximum 
and minimum values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a
measure of normality.

Random partitioning produces noticeably shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path
lengths for particular samples, they are highly likely to be anomalies.

Since the probability distribution nu defined in the paper is (in the interesting case)
continuous on a infinite dimensional space we do not represent it in this implementation. 
Instead, lot of dictionaries are already defined as Brownian dictionaries, Brownian bridges..
 where the input of the Wiener measure would be more difficult for the user. If one want to use
 a discrete measure nu, one have to 'replace it' with an appropriate dictionary.
 Example : if nu is discrete measure with ten values with different weight of probability
 and you have a dictionary D of size 10. Then you build a larger dictionaries with the 
 with the ten functions w.r.t. to their weights.




"""
import numpy as np


def derivate(X, step):
    """Compute de derivative of each function in the matrix X w.r.t vector time."""
    step = step.astype(dtype=float)
    A = np.zeros((X.shape[0], X.shape[1] - 1))
    for i in range(X.shape[0]):
        A[i] = np.diff(X[i]) / step
    return A
def derivate_piecewise(X, step):
    """Compute de derivative of each piecewise function in the matrix X w.r.t vector time."""
    A = np.zeros((X.shape[0], X.shape[1] - 1))
    for i in range(X.shape[0]):
        a = np.where(X[i] != 0)[0]
        b = a[0 : (a.shape[0] - 1)]
        A[i, b] = np.diff(X[i,a]) / step[b]
    return A
    
def c_factor(n_samples_leaf) :
    """
    Average path length of unsuccesful search in a binary search tree given n points
    
    Parameters
    ----------
    n_samples_lead : int
        Number of curves for the BST.
    Returns
    -------
    float
        Average path length of unsuccesful search in a BST
        
    """
    return 2.0 * (np.log(n_samples_leaf - 1) + np.euler_gamma) - (2. * (
        n_samples_leaf - 1.) / (n_samples_leaf * 1.0))


class FIForest(object):
    """
    Functional Isolation Forest
    
    Creates an FIForest object. This object holds the data as well as the trained trees (iTree objects).
    
    Attributes
    ----------
    X : Array-like
        Data used for training.
        
    nobjs: int
        Size of the dataset.
        
    sample: int
        Size of the sample to be used for tree creation.
        
    Trees: list
        A list of tree objects.
        
    limit: int
        Maximum depth a tree can have.
        
    c: float
        Multiplicative factor used in computing the anomaly scores.

    step : array
        Vector of the length of intervals of discretization.

    D : Array-like
        Dictionnary of functions used as directions.

    Dsize : int
        The size of the dictionary. It is the number of curves that we will use in our 
        dictionary to build the forest.
    
    innerproduct : str or function  
        An inner product that we use for the construction of the tree. The innerproduct in the paper
        is already implemented, call it with 'auto' and fixe and alpha. If a function is given by 
        the user, it should have three argument : (x, y, step) where x and y are curve (represented
        by a vector of length of the discretization). "step" is a vector of length len(time)-1 which
        represents the vector of length of step between the discretization.
                
    alpha : float
        a float number between [0,1] used in the innerproduct of the paper.
            
    deriv_X : Array like
        A matrix of derivate of X if needed for the scalar product.
        
    deriv_dictionary : Array like
        A matrix of derivate of D if needed for the scalar product.
        
    Attributes
    -------
    compute_paths(X_in) :
        Computes the anomaly score for data X_in
        
    threshold(score_sample, contamination) :
        Given the score returned by the fit function on training sample and a proportion 
        of anomalies, compute the threshold which separates anomalies and normal data.
        
    predict_label(score, contamination) :
        Given any score (training or testing) and the proportion of anomalies 
        it return the labels predicted. The function return +1 for outliers and
        -1 for inliers.
    
    
    References
    ----------
    
    .. [1] Staerman, G, Mozharovskyi, P, D'Alché-buc, F and Clémençon,S. "Functional Isolation forest."

    
    """

    def __init__(self, 
                 X,  
                 D,
                 time,
                 innerproduct,
                 ntrees=None,
                 subsample_size=None, 
                 Dsize=None, 
                 limit=None, 
                 mean=None, 
                 sd=None, 
                 J_max=None,  
                 alpha=None,
                 param1_min=None,
                 param1_max=None,
                 param2_min=None,
                 param2_max=None):
                #criterion="naive",
      
        self.X = X
        self.nobjs = len(X)
        self.Trees = []
        self.time = time
        #self.criterion = criterion
        self.mean = mean
        self.sd = sd
        self.D = D
        


        if (ntrees == None):
            self.ntrees = 100
        else: self.ntrees = ntrees

        if (subsample_size == None):
            if (self.nobjs > 500):
                self.sample = 256
            else: self.sample = 64
        else : self.sample = subsample_size


        if (Dsize == None):
            self.Dsize = 1000
        else: self.Dsize = Dsize 
        

        if (type(D) == str):
            """Finite dictionaries are pre-implemented.
            """ 
           
            if (D == 'Dyadic_indicator'):
                """ We build a dictionary from the basis of the Haar wavelets using 
                only the father wavelets. We use a discretization on [0,1] since 
                we are interested only in the shape.
                """
                if (J_max == None):
                    J_max = 7
                a =0
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((np.sum(np.power(2, np.arange(J_max))), len(self.time)))
                for J in range(J_max):
                    b = np.power(2, J)
                    for k in range(b):
                        for l in range(len(self.time)):
                            x = b * t[l] - k
                            self.D[a,l] = 1 * (0 <= x < 1)
                        a += 1
                        
            elif (D == 'Multiresolution_linear'):
                """ We build a dictionary from the basis of the Haar wavelets using 
                only the father wavelets. We use a discretization on [0,1] since 
                we are interested only in the shape.
                """
                if (J_max == None):
                    J_max = 7
                a =0
                t = np.linspace(0, 1, len(self.time))
                self.D = np.zeros((np.sum(np.power(2, np.arange(J_max))), len(self.time)))
                for J in range(J_max):
                    b = np.power(2,J)
                    for k in range(b):
                        for l in range(len(self.time)):
                            x = b * t[l] - k
                            self.D[a,l] = t[l] * (0 <= x < 1)
                        a += 1

            elif(D == 'Self_local'):
                """
                """
                self.D = np.zeros((self.Dsize, len(self.time)))
                for i in range(self.Dsize):
                    a = (self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0]
                    b = (self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0]
                    for j in range(len(self.time)):
                        k = np.random.randint(low=0, high=X.shape[0], size=1)
                        self.D[i,j] = self.X.copy()[k,j] * (np.maximum(a, b) > self.time[j] > np.minimum(a, b))

            elif (D == 'Self'):
                self.D = self.X.copy()

        self.alpha = alpha
        self.step = np.diff(self.time)

        if (type(D) == str):

            if (D == 'Self_local' or D == 'Self'):
                self.deriv_dictionary = derivate(self.D, self.step)

            elif(D == 'Multiresolution_linear' or D == 'Dyadic_indicator'):
                self.deriv_dictionary = derivate_piecewise(self.D, self.step)

            else: self.deriv_dictionary = []

        self.deriv_X = None

        if not callable(innerproduct):
            """ Some inner product implemented.
            """
            if (innerproduct == 'auto'):
                
                if (self.alpha == None):
                    self.alpha = 1
                
                if (self.alpha == 0):
                    self.deriv_X = derivate(self.X, self.step)

                    def innerproduct(x, y, xderiv, yderiv):
                        """We build the inner product in the paper with alpha = 0 which corresponds 
                        to L2 of derivate dot product.
                        """ 
                        F1 = x * y
                        F2 = xderiv * yderiv

                        F11 = F1[((np.arange(len(F1)) + 1) % len(F1))[:len(F1)-1]]
                        F12 = F1[((np.arange(len(F1)) + -1) % len(F1))[1:len(F1)]]
                        F21 = F2[((np.arange(len(F2)) + 1) % len(F2))[:len(F2)-1]]
                        F22 = F2[((np.arange(len(F2)) + -1) % len(F2))[1:len(F2)]]

                        return  (self.alpha *np.sum(( self.step * (F11 + F12) / 2))
                                +(1-self.alpha) * np.sum((self.step[0:(len(self.step) - 1)]
                                                          * (F21 + F22) / 2)))

                elif (self.alpha == 1):
                    def innerproduct(x, y, xderiv=None, yderiv=None ):
                        """We build the inner product in the paper with alpha = 1 which corresponds 
                        to L2 dot product.
                        """ 
                        F1 = x * y                        
                        return  np.sum((self.step * (F1[((np.arange(len(F1)) + 1) % len(F1))[:len(F1)-1]]
                                + F1[((np.arange(len(F1)) + -1) % len(F1))[1:len(F1)]]) / 2)) 
                                 
                    
                else:
                    self.deriv_X = derivate(self.X, self.step)
                    def innerproduct(x, y, xderiv, yderiv):
                        """We build the inner product in the paper which is a compromise between 
                        L2 scalar product and the L2 scalar product of derivate.
                        The function that we use work only with if we have the observations 
                        of curves at constant steps.
                        """ 
                        F1 = x * y
                        F2 = xderiv * yderiv
                        
                        F11 = F1[((np.arange(len(F1)) + 1) % len(F1))[:len(F1) - 1]]
                        F12 = F1[((np.arange(len(F1)) - 1) % len(F1))[1:len(F1)]]
                        F21 = F2[((np.arange(len(F2)) + 1) % len(F2))[:len(F2) - 1]]
                        F22 = F2[((np.arange(len(F2)) - 1) % len(F2))[1:len(F2)]]
                        
                        x11 = x[((np.arange(len(x)) + 1) % len(x))[:len(x) - 1]]
                        x12 = x[((np.arange(len(x)) - 1) % len(x))[1:len(x)]]
                        x21 = xderiv[((np.arange(len(xderiv)) + 1) % len(xderiv))[:len(xderiv) - 1]]
                        x22 = xderiv[((np.arange(len(xderiv)) - 1) % len(xderiv))[1:len(xderiv)]]
                        
                        y11 = y[((np.arange(len(y)) + 1) % len(y))[:len(y) - 1]]
                        y12 = y[((np.arange(len(y)) - 1) % len(y))[1:len(y)]]
                        y21 = yderiv[((np.arange(len(yderiv)) + 1) % len(yderiv))[:len(yderiv) - 1]]
                        y22 = yderiv[((np.arange(len(yderiv)) - 1) % len(yderiv))[1:len(yderiv)]]
                        return (self.alpha * np.sum(F11 + F12) / (np.sqrt(np.sum(x11 ** 2 + x12 ** 2)) * np.sqrt(np.sum(y11 ** 2 + y12 ** 2)))
                               + (1 - self.alpha) * np.sum(F21 + F22) / (np.sqrt(np.sum(x21 ** 2 + 
                                x22 ** 2)) * np.sqrt(np.sum(y21 ** 2 + y22 ** 2))))
                        
                    
            else: raise TypeError('This inner product is not pre-defined') 
        else: self.alpha = 1 

        self.innerproduct = innerproduct
        self.limit = limit
        self.c = c_factor(self.sample)


        if limit is None:
            """Set limit to the default as specified by the original paper
            (average depth of unsuccesful search through a binary tree).
            """ 
            self.limit = int(np.ceil(np.log2(self.sample))) 
            
        self.Subsample_index = []
        self.Subsample_index_complementary = []

        if (self.alpha == 1):
            for i in range(self.ntrees): 
                """This loop builds an ensemble of f-itrees (the forest).
                """
                ixx = np.random.choice(np.arange(self.nobjs), size=self.nobjs, replace=False)
                
                ix = ixx[:self.sample]

                self.Subsample_index.append(ix)
                self.Subsample_index_complementary.append(ixx[self.sample:])

                
                self.Trees.append(iTree(X[ix], self.time, self.step,  
                                            0, self.limit, 
                                            self.D, self.innerproduct, 
                                            self.alpha, self.deriv_X, 
                                            None, self.sample, 
                                             self.mean, self.sd,param1_min,
                                             param1_max, param2_min, param2_max))

        else:
            for i in range(self.ntrees): 
                """This loop builds an ensemble of f-itrees (the forest).
                """
                ixx = np.random.choice(np.arange(self.nobjs), size=self.nobjs, replace=False)
                
                ix = ixx[:self.sample]           
                
                self.Subsample_index.append(ix)
                self.Subsample_index_complementary.append(ixx[self.sample:])
                
                self.Trees.append(iTree(X[ix], self.time, self.step, 
                                            0, self.limit, 
                                            self.D, self.innerproduct, 
                                            self.alpha, self.deriv_X[ix], 
                                            self.deriv_dictionary, 
                                            self.sample, 
                                            self.mean, self.sd, param1_min=param1_min,
                                             param1_max=param1_max, param2_min=param2_min,
                                             param2_max=param2_max))

                    


    def compute_paths(self, X_in=None):
        """
        compute_paths(X_in = None) 

        Compute the anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        Parameters
        ----------
        X_in : Array-like
                Data to be scored. FIForest.Trees are used for computing the depth reached in 
                each tree by each data curve.
        Returns
        -------
        float
            Anomaly score for a given data curve.
        """
        if X_in is None:
            X_in = self.X           
            if(self.alpha != 1):
                deriv_X_in = self.deriv_X
        else: 
            if(self.alpha != 1):
                deriv_X_in = derivate(X_in, self.step)

        S = np.zeros(len(X_in))
        
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                # Compute path length for each curve
                if(self.alpha != 1):
                    h_temp += PathFactor(X_in[i], self.step, 
                                         self.Trees[j], self.alpha,
                                         deriv_X_in[i]).path * 1.0  
                else:
                    h_temp += PathFactor(X_in[i], self.step, 
                                         self.Trees[j], 
                                         self.alpha).path * 1.0  
                
            # Average of path length travelled by the point in all trees.
            Eh = h_temp / self.ntrees
            
             # Anomaly Score
            S[i] = 2.0 ** (- Eh / self.c)                                           
        return S

    def threshold(self, score_samples, contamination=0.1):
        """Compute the treshold to declare curves as anomalies or not.
           The choice of 'lower' interpolation in the percentile function come from
           the fact that it should be a little gap between the score of anomalies and the normal score. 
           This choice could be different depending on the problem given.
           
        Parameters
        ----------
        
        score_samples : Array
            The score array for a dataset of curves.
            
        contamination : float, optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the decision function.
            
        """
        return np.percentile(score_samples, 100 * (1 - contamination), interpolation='lower')
    
    def predict_label(self, score, contamination=0.1):
         
        """Compute the label vector of curves.  
        
        Parameters
        ----------
        
        score : Array
            The score array for a dataset of curves (training or testing).
            
        contamination : float, optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the decision function.
            
        Returns
        -------
        
        y_label : array
            An array of predict label, -1 if the curve is considered as normal and +1 if not.
        """
        y_label = np.zeros((len(score)))
        return 1- 2.0 * (score > self.threshold(score, contamination))

    def dictionary_selection(self):
        a = 0
        for i in range(self.ntrees):
            a += self.Trees[i].count
        return a
"""
    def importance_feature(self):
        IF = np.zeros((self.D.shape[0]))

        
        for i in range(self.ntrees):
            IF += self.Trees[i].IF
    

        return IF
"""
"""
    def get_Dictionary_information(self):

        Dic = []        
        IF = []

        if (self.D == 'cosinus' or self.D == 'gaussian_wavelets'):
            Param = []

            for i in range(self.ntrees):
                for j in range(len(self.Trees[i].Dictionary)):
                    Dic.append(self.Trees[i].Dictionary[j][0])
                    Param.append(self.Trees[i].Dictionary[j][4])
                    IF.append(self.Trees[i].Dictionary[j][2])
            


            return Dic, Param, IF

        else: 

            for i in range(self.ntrees):
                for j in range(len(self.Trees[i].Dictionary)):
                    Dic.append(self.Trees[i].Dictionary[j][0])
                    IF.append(self.Trees[i].Dictionary[j][2])   

            return Dic, IF
"""
"""        
    def out_of_bag(self, liste1, liste2):

        Score = np.zeros((len(liste1)-1,len(liste2)-1))
        for j in range(len(liste1)-1):
            for k in range(len(liste2)-1):

                if (self.alpha == 1):
                    for i in range(self.ntrees): 
                        #This loop builds an ensemble of f-itrees (the forest).
                        
                        # We build the same forest without the region direction choosen,
                        # we do this for each region. We then have 
                        #ntrees * (len(liste1)-1) * (len(liste2)-1) trees

                        self.OOB.append(iTree(self.X[self.Subsample_index[i]], 
                                                    self.time, self.step,  
                                                    0, self.limit, 
                                                    self.D, self.innerproduct, 
                                                    self.alpha, self.deriv_X, 
                                                    None, self.sample, self.criterion,
                                                     self.mean, self.sd, 
                                                     oob_param1_min=liste1[j],
                                                     oob_param1_max=liste1[j+1],
                                                     oob_param2_min=liste2[k],
                                                     oob_param2_max=liste2[k+1]))


                        if(self.alpha != 1):
                            deriv_X_in = derivate(self.X[self.Subsample_index_complementary[i]], 
                                                    self.step)
                        n0 = len(self.X[self.Subsample_index_complementary[i]])

                        S = np.zeros(n0)
                        

                        for l in  range(n0):
                            
                            
                            # Compute path length for each curve
                            if(self.alpha != 1):
                                S[l] = np.absolute(PathFactor(self.X[self.Subsample_index_complementary[i][l]], self.step, 
                                                     self.Trees[i], self.alpha,
                                                     deriv_X_in[l]).path * 1.0 -  PathFactor(self.X[self.Subsample_index_complementary[i][l]], self.step, 
                                                     self.OOB[len(self.OOB)-1], self.alpha, deriv_X_in[l]).path * 1.0 )
                            else:
                                #print(len(self.Trees))
                                #print(len(self.OOB))
                                #print(S[l])
                                S[l] = np.absolute(PathFactor(self.X[self.Subsample_index_complementary[i][l]], self.step, 
                                                     self.Trees[i], self.alpha).path * 1.0 -  PathFactor(self.X[self.Subsample_index_complementary[i][l]], self.step, 
                                                     self.OOB[len(self.OOB)-1], self.alpha).path * 1.0  )
                            
                            
                        Score[j, k] += np.mean(S)
                    Score[j, k] = Score[j, k] / self.ntrees 
        return Score                   
"""                                                            
                        








"""   
                else:
                    for i in range(self.ntrees): 
                        This loop builds an ensemble of f-itrees (the forest).
                        
                        
                        self.OOB.append(iTree(X[self.Subsample_index[i]],
                                                     self.time, self.step, 
                                                    0, self.limit, 
                                                    self.D, self.innerproduct, 
                                                    self.alpha, self.deriv_X[ix], 
                                                    self.deriv_dictionary, 
                                                    self.sample, self.criterion, 
                                                    self.mean, self.sd,
                                                    oob_param1_min=liste1[j],
                                                    oob_param1_max=liste1[j+1],
                                                    oob_param2_min=liste2[k],
                                                    oob_param2_max=liste2[k+1]))
"""





class Node(object): 
    """
    A single node from each tree (each iTree object). Nodes containe information on 
    hyperplanes used for data division, date to be passed to left and right nodes, 
    whether they are external or internal nodes.
    Attributes
    ----------
    e: int
        Depth of the tree to which the node belongs.
        
    size: int
        Size of the dataset present at the node.
        
    X: Array-like
        Data at the node.
        
    d: Array
        Direction function used to build the hyperplane that splits the data in the node.

    dd : int
        The index of the direction chosen at this node.
        
    q: Array
        Intercept point through which the hyperplane passes.
        
    left: Node object
        Left child node.
        
    right: Node object
        Right child node.
        
    ntype: str
        The type of the node: 'exNode', 'inNode'.
    """
    def __init__(self, 
                 X, 
                 d, 
                 dd, 
                 q, 
                 e, 
                 left, 
                 right, 
                 node_type='' ):

        self.e = e
        self.size = len(X)
        self.X = X 
        self.d = d
        self.dd = dd
        self.q = q
        self.left = left
        self.right = right
        self.ntype = node_type

class iTree(object):

    """
    A single tree in the forest that is build using a unique subsample.
    Attributes
    ----------
    e: int
        Depth of tree
        
    X: list
        Data present at the root node of this tree.

    step : array
        Vector of the length of intervals of discretization.
        
    size: int
        Size of the dataset.
        
    dim: int
        Dimension of the dataset.
        
    l: int
        Maximum depth a tree can reach before its creation is terminated.
        
    d: Array
        Normal vector at the root of this tree, which is used in creating hyperplanes for 
        splitting criteria.

    dd : int
        The index of the direction chosen at this node.
        
    q: float
        Intercept point at the root of this tree through which the splitting hyperplane passes.
        
    root: Node object
        At each node create a new tree.
        
    D: Array-like
        Dictionary of functions used as directions.
        
    innerproduct :  str or function  
        An inner product that we use for the construction of the tree.
        
    alpha : float
        A float number between [0,1] used in the innerproduct of the paper.

    deriv_X : Array-like
        A matrix of derivate of X if needed for the scalar product.
        
    deriv_dictionary : Array-like
        A matrix of derivate of D if needed for the scalar product.
        
    Methods
    -------
    make_tree(X, e, l, D, innerproduct)
        Builds the tree recursively from a given node. Returns a Node object.
    """

    def __init__(self, 
                 X,
                 time, 
                 step, 
                 e, 
                 l, 
                 D, 
                 innerproduct, 
                 alpha, 
                 deriv_X=None, 
                 deriv_dictionary=None,
                 subsample_size=None,               
                 mean=None,
                 sd=None,
                 param1_min=None,
                 param1_max=None,
                 param2_min=None,
                 param2_max=None):
                 #criterion=None,
                 #oob_param1_min=None,
                 #oob_param1_max=None,
                 #oob_param2_min=None,
                 #oob_param2_max=None):
        
        self.e = e
        self.X = X
        self.step = step
        self.time = time
        self.size = len(X)
        self.dim = self.X.shape[1]
        self.l = l
        self.q = None                                       
        self.d = None
        self.dd = None
        self.exnodes = 0
        self.D = D    
        self.innerproduct = innerproduct
        self.alpha = alpha
        self.deriv_X = deriv_X
        self.mean = mean
        self.sd = sd
        self.Dictionary = []

        self.deriv_dictionary = deriv_dictionary

        #if (type(self.D) != str):
        #    self.IF = np.zeros((self.D.shape[0]))
        #else: self.IF = [] 

        self.subsample_size = subsample_size
        self.count = 0
        #self.criterion = criterion

        
        self.param1_min = param1_min
        self.param1_max = param1_max
        self.param2_min = param2_min
        self.param2_max = param2_max

        #self.oob_param1_min = oob_param1_min
        #self.oob_param1_max = oob_param1_max
        #self.oob_param2_min = oob_param2_min
        #self.oob_param2_max = oob_param2_max



        #if oob_param1_min is None:
        #        self.oob_param1_min = 0
        #if oob_param1_max is None:
        #    self.oob_param1_max = 0

        #if oob_param2_min is None:
        #    self.oob_param2_min = 0
        #if oob_param2_max is None:
        #    self.oob_param2_max = 0


        if (type(self.D) == str):
            if (self.D == 'cosinus'):

                if param1_min is None:
                    self.param1_min = -1
                if param1_max is None:
                    self.param1_max = 1

                if param2_min is None:
                    self.param2_min = 0
                if param2_max is None:
                    self.param2_max = 10

            elif (self.D == 'gaussian_wavelets'):
                if param1_min is None:
                    self.param1_min = -4
                if param1_max is None:
                    self.param1_max = 4

                if param2_min is None:
                    self.param2_min = 0.1
                if param2_max is None:
                    self.param2_max = 1






        # At each node create a new tree, starting with root node.
        self.root = self.make_tree(self.X, self.e)
        


    def make_tree(self, X, e):
        """
        make_tree(X,e,l,D, innerproduct)
        Builds the tree recursively from a given node. Returns a Node object.
        Parameters
        ----------
        X: Array like
            Subsample of training data. 
            
        e : int
            Depth of the tree as it is being traversed down. Integer. e <= l.
               
        Returns
        -------
        Node object
        """
        
        self.e = e
        # A curve is isolated in training data, or the depth limit has been reached.
        if e >= self.l or len(X) <= 1:                                               
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.d, self.dd, self.q, e, left, right, node_type='exNode')
        
        # Building the tree continues. All these nodes are internal.
        else:
            sample_size = X.shape[0] 
            t = np.linspace(0,1,len(self.step)+1)

            if (type(self.D) != str):
                # For finite dictionaries, we draw direction from them.
                idx = np.random.choice(np.arange((self.D).shape[0]), size=1)
                self.d = self.D[idx[0],:]
                self.dd = idx[0]



            elif (self.D == 'cosinus'):
                """ We draw directions from the cosinus dictionary defined in the paper
                 (with random amplitudes and frequences).
                """
                a = np.random.uniform(self.param1_min, self.param1_max, size=1)
                b = np.random.uniform(self.param2_min, self.param2_max, size=1)

                #while (self.oob_param1_min < a < self.oob_param1_max):
                #    a = np.random.uniform(self.param1_min, self.param1_max, size=1)
                #while (self.oob_param2_min < b < self.oob_param2_max):
                #    b = np.random.uniform(self.param2_min, self.param2_max, size=1)


                self.d = a  * np.cos(2 * np.pi * b * t)

                #info  = [self.d, 'param:', np.array([a,b]), 'Index IF :']
                
                if (self.alpha != 1):
                    self.deriv_dictionary.append(np.diff(self.d) / self.step)
                    self.dd = len(self.deriv_dictionary) - 1

            elif (self.D == 'Brownian'):
                """ We draw directions from the Brownian motion dictionary defined in the paper"""

                                                                         
                if (self.mean == None):
                    self.mean = 0
                
                if (self.sd == None):
                    self.sd = 1
                    
                self.d = np.zeros((len(t)))
                self.d[0] = np.random.normal(self.mean, scale=self.sd , size=1) 

                


                for i in range(1,len(t)):
                    self.d[i] += self.sd * np.random.normal(0, scale=np.sqrt(t[2] - t[1])
                                                                , size=1) + self.mean * (t[2] - t[1])

                #info  = [self.d, 'Index IF :']

                if (self.alpha != 1):
                    self.deriv_dictionary.append(np.diff(self.d) / self.step)
                    self.dd = len(self.deriv_dictionary) - 1 

            elif (self.D == 'gaussian_wavelets'):
                """ We draw directions from the gaussian wavelets dictionary.
                 We use a discretization on [-5,5] and add two random parameters 
                 to get an interesting dictionary. 
                The standard deviation sigma and a translation parameter K. The range of these 
                parameters are fixed.
                """
                
                sigma = np.random.uniform(self.param2_min, self.param2_max, size=1)
                K = np.random.uniform(self.param1_min, self.param1_max, size=1)

                #while (self.oob_param1_min < K < self.oob_param1_max):
                #    K = np.random.uniform(self.param1_min, self.param1_max, size=1)
                #while (self.oob_param2_min < sigma < self.oob_param2_max):
                #    sigma = np.random.uniform(self.param2_min, self.param2_max, size=1)

                t = np.linspace(-5,5,len(self.step)+1)

                self.d = (-(2 / (np.power(np.pi,0.25) * np.sqrt(3 * sigma)) ) 
                             * ((t - K) ** 2 / (sigma ** 2) -1) * (
                             np.exp(-(t - K) ** 2 / (2 * sigma ** 2))))

                #info  = [self.d, 'param:', np.array([K, sigma]), 'Index IF :']

                if (self.alpha != 1):
                    self.deriv_dictionary.append(np.diff(self.d) / self.step)
                    self.dd = len(self.deriv_dictionary) - 1 

            elif (self.D == 'Brownian_bridge'):
                """ We draw directions from the Brownian bridge dictionary defined in the paper"""

                self.d = np.zeros((len(t)))
                for i in range(1,(len(t)-1)):
                    self.d[i] +=  np.random.normal(0, np.sqrt(t[2] - t[1])
                                  , size=1) - self.d[i-1] * (t[2] - t[1]) / (1 - t[i])

                #info  = [self.d, 'Index IF :']

                if (self.alpha != 1):
                    self.deriv_dictionary.append(np.diff(self.d) / self.step)
                    self.dd = len(self.deriv_dictionary) - 1 


            elif (self.D == 'indicator_uniform'):
                """ We draw directions from the indicator uniform dictionary defined in the paper"""

                self.d = np.zeros((len(t)))
                a = ((self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0])
                b = (self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0]
                for j in range(len(self.time)):
                    self.d[j] = 1. * (np.maximum(a, b) > self.time[j] > np.minimum(a, b))

                #nfo  = [self.d, 'Index IF :']


            elif (self.D == 'linear_indicator_uniform'):
                """ We draw directions from the Linear indicator uniform dictionary defined in the paper"""


                self.d = np.zeros((len(t)))
                a = (self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0]
                b = (self.time[len(self.time) - 1] - self.time[0]) * np.random.random() + self.time[0]
                for j in range(len(self.time)):
                    self.d[j] = self.time[j] * (np.maximum(a,b) > self.time[j] > np.minimum(a,b))

                #info  = [self.d, 'Index IF :']

                if (self.alpha != 1):
                    self.deriv_dictionary.append(np.diff(self.d) / self.step)
                    self.dd = len(self.deriv_dictionary) - 1 

            else: raise TypeError('This Dictionary is not pre-defined')
    


           




            Z = np.zeros((sample_size))

            if (self.alpha != 1):
                for i in range(sample_size):


                    Z[i] = self.innerproduct(X[i,:], self.d, self.deriv_X[i], 
                                                  self.deriv_dictionary[self.dd])
            
            else: 
                for i in range(sample_size): 
                    Z[i] = self.innerproduct(X[i,:], self.d)
                    
            # Picking a random threshold for the hyperplane splitting data.

            self.q = np.random.uniform(np.min(Z), np.max(Z)) 

            # Criteria that determines if a curve should go to the left or right child node.

            w = Z - self.q < 0

            summ = np.sum(w)

            if (sample_size >2):
                if (summ == 1 or summ == sample_size - 1):
                    self.count += 1#sample_size / self.subsample_size 

            #info.append((np.maximum(summ, sample_size - summ) / np.minimum(summ, sample_size - summ)) * np.exp(-(e+1))) 

            
            '''
            if (type(self.D) == str):
                if (sample_size >2):
                    if (np.sum(w) == 1 or np.sum(w) == sample_size - 1): 
                        info.append(sample_size / self.subsample_size )
                    else: info.append(0)
                        #if (self.criterion == "naive"):
                            #self.IF[idx[0]] += 1
                        #elif(self.criterion == "sample"):
                            #self.IF[idx[0]] += sample_size / self.subsample_size 

                        #else:
                            #self.IF[idx[0]] += 1 / (e + 1) 
                else: info.append(0)
            '''

            #self.Dictionary.append(info)


            return Node(self.X, self.d, self.dd, self.q, e,\
            left=self.make_tree(X[w], e+1),\
            right=self.make_tree(X[~w], e+1),\
            node_type = 'inNode' )

class PathFactor(object):
    """
    Given a single tree (iTree objext) and a curve x , compute the length of the path traversed
    by the point on the tree when it reaches an external node.
    
    Attributes
    ----------
    path_list: list
        A list of strings 'L' or 'R' which traces the path a data curve travels down a tree.
        
    x: list
        A single function, which is represented as an array floats.
        
    e: int
        The depth of a given node in the tree.
        
    deriv_x : Array
        The derivative of the new function if needed for the scalar product.

    step : array
        Vector of the length of intervals of discretization.

    D: Array-like
        Dictionary of functions used as directions.
        
    innerproduct :  str or function  
        An inner product that we use for the construction of the tree.
        
    alpha : float
        A float number between [0,1] used in the innerproduct of the paper.

    deriv_X : Array-like
        A matrix of derivate of X if needed for the scalar product.
        
    deriv_dictionary : Array-like
        A matrix of derivate of D if needed for the scalar product.
        
    Methods
    -------
    find_path(T)
        Given a tree, it finds the path a single data curves takes.
    """
    def __init__(self, 
                 x, 
                 step, 
                 itree, 
                 alpha, 
                 deriv_x=None):

        self.path_list=[]
        self.x = x
        self.deriv_x = deriv_x
        self.e = 0
        self.alpha = alpha
        self.step = step
        self.D = itree.D
        self.deriv_dictionary = itree.deriv_dictionary
        self.innerproduct = itree.innerproduct
        self.path  = self.find_path(itree.root)

    def find_path(self, T):
        """
        find_path(T)
        Given a tree, find the path for a single curve based on the splitting criteria 
        stored at each node.
        
        Parameters
        ----------
        T : Node object
        
        innerproduct : str or function
            The innerproduct use in the Forest.
        
        
        Returns
        -------
        int
            The depth reached by the data curve.
        """
        if T.ntype == 'exNode':
            
            if T.size <= 1: return self.e
            
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:

            q = T.q 
            d = T.d                                                             
            self.e += 1            
            if (self.alpha != 1):
                if self.innerproduct(self.x, d, self.deriv_x, self.deriv_dictionary[T.dd]) - q < 0:
                    self.path_list.append('L')
                    return self.find_path(T.left)
                else:
                    self.path_list.append('R')
                    return self.find_path(T.right)
            else:
                if self.innerproduct(self.x, d, self.step) - q < 0:
                    self.path_list.append('L')
                    return self.find_path(T.left)
                else:
                    self.path_list.append('R')
                    return self.find_path(T.right)

