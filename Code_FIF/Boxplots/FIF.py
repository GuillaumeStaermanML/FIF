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
    step = step.astype(dtype = float)
    A = np.zeros((X.shape[0],X.shape[1]-1))
    for i in range(X.shape[0]):
        A[i] = np.diff(X[i]) / step
    return A
def derivate_piecewise(X, step):
    """Compute de derivative of each piecewise function in the matrix X w.r.t vector time."""
    A = np.zeros((X.shape[0],X.shape[1]-1))
    for i in range(X.shape[0]):
        a = np.where(X[i] != 0)[0]
        b = a[0:(a.shape[0]-1)]
        A[i,b] = np.diff(X[i,a]) / step[b]
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
                 innerproduct,
                 ntrees=None,
                 time=None,
                 subsample_size=None, 
                 Dsize=None, 
                 limit=None, 
                 mean=None, 
                 sd=None, 
                 J_max=None, 
                 amplitude_min=None, 
                 amplitude_max=None, 
                 alpha=None):
      
        self.X = X
        self.nobjs = len(X)
        self.Trees = []

        if (ntrees == None):
            self.ntrees = 100
        else: self.ntrees = ntrees

        if (time == None):
            self.time = np.linspace(0,1,self.X.shape[1])
        else: self.time = time

        if (subsample_size == None):
            if (len(X)>500):
                self.sample = 256
            else: self.sample = 64
        else : self.sample = subsample_size


        if (Dsize == None):
            self.Dsize = 1000
        else: self.Dsize = Dsize 
        

        if (type(D) == str):
            """Some dictionary pre-implemented.
            """ 
            
                    
            if (D == 'Brownian'):
                """ We build a dictionary from brownian motion (standard or drift).
                We use a discretization on [0,1] since we are interested only by the shape
                of curves.
                """
                if (mean == None):
                    mean = 0
                
                if (sd == None):
                    sd = 1
                    
                self.D = np.zeros((self.Dsize,len(self.time)))
                t = np.linspace(0, 1, len(self.time))
                self.D[:,0] = np.random.normal(mean, scale = sd , size = self.Dsize) 
                for i in range(1,np.size(self.time)):
                    self.D[:,i] = self.D[:, i-1] + sd * np.random.normal(0, scale = np.sqrt(t[2] - t[1])
                                                                , size = self.Dsize) + mean * (t[2] - t[1]) 

            elif (D == 'Brownian_bridge'):
                """ We build a dictionary from Brownian bridge.
                """
                    
                self.D = np.zeros((self.Dsize,len(self.time)))
                t = np.linspace(0, 1, len(self.time))
                for i in range(1,(len(self.time)-1)):
                    self.D[:,i] = self.D[:, i-1] +  np.random.normal(0, np.sqrt(t[2] - t[1])
                                  , self.Dsize) - self.D[:,i-1] * (t[2] - t[1]) / (1 - t[i])
                    
            elif (D == 'gaussian_wavelets'):  
                """ We build a dictionary from gaussian wavelets. We use a discretization on [-5,5]
                and add two random parameters to get an interesting dictionary. 
                The standard deviation sigma and a translationparameter K. The range of these 
                parameters are fixed.
                """
                t = np.linspace(-5,5,len(self.time))
                self.D = np.zeros((self.Dsize,len(self.time)))
                for i in range(self.Dsize):
                    sigma = np.random.uniform(0.2,1)
                    K = np.random.uniform(-4,4)
                    for l in range(len(self.time)):
                        self.D[i,l] = (-(2 / (np.power(np.pi,0.25) * np.sqrt(3 * sigma)) ) 
                                 * ((t[l] - K) ** 2 / (sigma ** 2) -1) * (
                                 np.exp(-(t[l] - K) ** 2 / (2 * sigma ** 2))))
                        
            elif (D == 'Dyadic_indicator'):
                """ We build a dictionary from the basis of the Haar wavelets using 
                only the father wavelets. We use a discretization on [0,1] since 
                we are interested only in the shape.
                """
                if (J_max == None):
                    J_max = 7
                a =0
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((np.sum(np.power(2,np.arange(J_max))),len(self.time)))
                for J in range(J_max):
                    b = np.power(2,J)
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
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((np.sum(np.power(2,np.arange(J_max))),len(self.time)))
                for J in range(J_max):
                    b = np.power(2,J)
                    for k in range(b):
                        for l in range(len(self.time)):
                            x = b * t[l] - k
                            self.D[a,l] = t[l] * (0 <= x < 1)
                        a += 1
                        
            elif (D == 'linear_indicator_uniform'):
                """ 
                """
                self.D = np.zeros((self.Dsize,len(self.time)))
                

                for i in range(self.Dsize):
                    a = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    b = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    for j in range(len(self.time)):
                        self.D[i,j] = self.time[j] * (b > self.time[j] > a)
                        
                        
            elif (D == 'indicator_uniform'):
                """ 
                """
                self.D = np.zeros((self.Dsize,len(self.time)))

                for i in range(self.Dsize):
                    a = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    b = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    for j in range(len(self.time)):
                        self.D[i,j] = 1. * (b > self.time[j] > a)

            elif(D == 'Self_local'):
                """
                """
                self.D = np.zeros((self.Dsize,len(self.time)))
                for i in range(self.Dsize):
                    a = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    b = (self.time[len(self.time)-1]-self.time[0]) * np.random.random() + self.time[0]
                    for j in range(len(self.time)):
                        k = np.random.randint(low = 0, high = X.shape[0], size = 1)
                        self.D[i,j] = self.X.copy()[k,j] * (b > self.time[j] > a)
            elif ( D == 'Self'):
                self.D = self.X.copy()
                
            elif (D == 'cosinus'):
                """ We build a cosinus dictionary with random amplitudes and frequences.
                Amplitudes are fixed by the user while freq are fixed by the algorithm 
                with a large range to avoid overloading parameters.
                """
                if (amplitude_min == None):
                    amplitude_min = -1
                    
                if (amplitude_max == None):
                    amplitude_max = 1
                    
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((self.Dsize,len(self.time)))
                for i in range(self.Dsize):
                    freq = np.random.uniform(0, 10, 1)
                    amp = np.random.uniform(amplitude_min, amplitude_max, 1)
                    self.D[i,:] = amp * np.cos(2 * np.pi * freq * t)

            else: raise TypeError('This Dictionary is not pre-defined')
        else: self.D = D 
 
                

        self.alpha = alpha
        self.step = np.diff(self.time)
        self.deriv_dictionary = None
        self.deriv_X = None
        if not callable(innerproduct):
            """ Some inner product implemented.
            """
            if (innerproduct == 'auto'):
                
                if (self.alpha == None):
                    self.alpha =1
                
                if (self.alpha == 0):
                    self.deriv_X = derivate(self.X, self.step)
                    if (type(D) == str):
                        if (D == 'linear_indicator_uniform' or D == 'Multiresolution_linear'):
                            self.deriv_dictionary = derivate_piecewise(self.D, self.step)
                            
                        else:
                            self.deriv_dictionary = derivate(self.D, self.step)
                            self.deriv_X = derivate(self.X, self.step)
                    else:
                            self.deriv_dictionary = derivate(self.D, self.step)
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
                    def innerproduct(x, y, xderiv = None, yderiv = None ):
                        """We build the inner product in the paper with alpha = 1 which corresponds 
                        to L2 dot product.
                        """ 
                        F1 = x * y                        
                        return  np.sum((self.step * (F1[((np.arange(len(F1)) + 1) % len(F1))[:len(F1)-1]]
                                + F1[((np.arange(len(F1)) + -1) % len(F1))[1:len(F1)]]) / 2)) 
                                 
                    
                else:
                    self.deriv_X = derivate(self.X, self.step)
                    if (type(D) == str):
                        if (D == 'linear_indicator_uniform' or D == 'Multiresolution_linear'):
                            self.deriv_dictionary = derivate_piecewise(self.D, self.step)
                        
                        else:
                            self.deriv_dictionary = derivate(self.D, self.step)
                            self.deriv_X = derivate(self.X, self.step)
                    else:
                            self.deriv_dictionary = derivate(self.D, self.step)
                            self.deriv_X = derivate(self.X, self.step)

                    def innerproduct(x, y, xderiv, yderiv):
                        """We build the inner product in the paper which is a compromise between 
                        L2 scalar product and the L2 scalar product of derivate.
                        The function that we use work only with if we have the observations 
                        of curves at constant steps.
                        """ 
                        F1 = x * y
                        F2 = xderiv * yderiv
                        
                        F11 = F1[((np.arange(len(F1)) + 1) % len(F1))[:len(F1)-1]]
                        F12 = F1[((np.arange(len(F1)) + -1) % len(F1))[1:len(F1)]]
                        F21 = F2[((np.arange(len(F2)) + 1) % len(F2))[:len(F2)-1]]
                        F22 = F2[((np.arange(len(F2)) + -1) % len(F2))[1:len(F2)]]
                        
                        x11 = x[((np.arange(len(x)) + 1) % len(x))[:len(x)-1]]
                        x12 = x[((np.arange(len(x)) + -1) % len(x))[1:len(x)]]
                        x21 = xderiv[((np.arange(len(xderiv)) + 1) % len(xderiv))[:len(xderiv)-1]]
                        x22 = xderiv[((np.arange(len(xderiv)) + -1) % len(xderiv))[1:len(xderiv)]]
                        
                        y11 = y[((np.arange(len(y)) + 1) % len(y))[:len(y)-1]]
                        y12 = y[((np.arange(len(y)) + -1) % len(y))[1:len(y)]]
                        y21 = yderiv[((np.arange(len(yderiv)) + 1) % len(yderiv))[:len(yderiv)-1]]
                        y22 = yderiv[((np.arange(len(yderiv)) + -1) % len(yderiv))[1:len(yderiv)]]
                        return (self.alpha * np.sum(F11 + F12) / (np.sqrt(np.sum(x11 ** 2 + x12 ** 2)) * np.sqrt(np.sum(y11**2 + y12**2)))
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
            
        

        
        if (self.alpha == 1):
            for i in range(self.ntrees): 
                """This loop builds an ensemble of iTrees (the forest).
                """
                ix = np.random.choice(np.arange(self.nobjs), size = self.sample, replace = False)
                
                self.Trees.append(iTree(X[ix], self.step,  
                                        0, self.limit, 
                                        self.D, self.innerproduct, 
                                        self.alpha, self.deriv_X, 
                                        self.deriv_dictionary))
        else:
            for i in range(self.ntrees): 
                """This loop builds an ensemble of iTrees (the forest).
                """
                ix = np.random.choice(np.arange(self.nobjs), size = self.sample, replace = False)
                
                self.Trees.append(iTree(X[ix], self.step, 
                                        0, self.limit, 
                                        self.D, self.innerproduct, 
                                        self.alpha, self.deriv_X[ix], 
                                        self.deriv_dictionary))


    def compute_paths(self, X_in = None):
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
    def threshold(self, score_samples, contamination = 0.1):
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
        return np.percentile(score_samples, 100 * (1 - contamination), interpolation = 'lower')
    
    def predict_label(self, score, contamination = 0.1):
         
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
                 step, 
                 e, 
                 l, 
                 D, 
                 innerproduct, 
                 alpha, 
                 deriv_X=None, 
                 deriv_dictionary=None):
        
        self.e = e
        self.X = X
        self.step = step
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
        self.deriv_dictionary = deriv_dictionary
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
            return Node(X, self.d, self.dd, self.q, e, left, right, node_type = 'exNode')
        
        # Building the tree continues. All these nodes are internal.
        else:                                                                   
            sample_size = X.shape[0] 
            idx = np.random.choice(np.arange((self.D).shape[0]), size=1)
            self.d = self.D[idx[0],:]
            self.dd = idx[0]
            Z = np.zeros((sample_size))

            if (self.alpha != 1):
                for i in range(sample_size):
                        Z[i] = self.innerproduct(X[i,:], self.d, self.deriv_X[i], 
                                                 self.deriv_dictionary[idx[0]] )
      
            else: 
                for i in range(sample_size): 
                        Z[i] = self.innerproduct(X[i,:], self.d)
                    
            # Picking a random threshold for the hyperplane splitting data.
            #print(np.min(Z))
            self.q = np.random.uniform(np.min(Z), np.max(Z)) 
            # Criteria that determines if a curve should go to the left or right child node.
            w = Z - self.q < 0                                                    
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

