""" Multivariate Functional Isolation Forest

    Author : Guillaume Staerman
"""


"""Multivariate Functional Isolation Forest Algorithm

This is the implementation of The Multivariate Functional Isolation Forest which is an
extension of the original Isolation Forest applied to functional data.

It return the anomaly score of each sample using the FIF algorithm.
The Functional Isolation Forest 'isolates' observations by 
randomly selecting a multivariate curve among a dictionary
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

"""
import numpy as np
#from version import __version__



def derivateM(X, step):
    """Compute de derivative of a multivariate function X on each dimension.
    """
    step = step.astype(dtype = float)
    A = np.zeros((X.shape[0],X.shape[1]-1))
    for i in range(X.shape[0]):
            A[i,:] = np.diff(X[i,:]) / step
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


class MFIForest(object):
    """
    Multivariate Functional Isolation Forest
    
    Creates an MFIForest object. This object holds the data as well as the trained trees (iTree objects).
    
    Attributes
    ----------
    X : Array-like (n_samples, dimension, discretization)
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
        
    mean : float or None, optional (Default=None)
        The mean of the stochastic process used to build a stochastic dictionary. 
        This is set to zero by default. If a stochastic dictionary is called 
        and no mean is given, it is set to 0.

    sd : float or None, optional (Default=None)
        The standard deviation of the stochastic process used to build a stochastic dictionary.
        This is set to one by default. If a stochastic dictionary is called 
        and no standard deviation is given, it is set to 1.

    J_max : int or None, optional (Default=None)
        This parameter fix the size of the dictionary of Haar_wavelet_father. 
        It will build 2 power J_max functions.

    amplitude_min : float or None, optional (Default=None)
        This parameter is used for cosinus dictionary. 
        It is the minimum amplitude with which one draws the amplitude.

    amplitude_max : float or None, optional (Default=None)
        This parameter is used for cosinus dictionary. 
        It is the minimum amplitude with which one draws the amplitude.
        
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
                 time,
                 ntrees=None,
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
        self.time = time


        if (ntrees == None):
            self.ntrees = 100
        else: self.ntrees = ntrees


        if (subsample_size == None):
            if (len(X)>800):
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
                We use a discretization on [0,1] since we are interested only in the shape.
                """
                if (mean == None):
                    mean = np.zeros(((self.X).shape[1]))
                
                if (sd == None):
                    sd = np.eye((self.X).shape[1],(self.X).shape[1])
                    
                self.D = np.zeros((self.Dsize,(self.X).shape[1], len(self.time)))
                t = np.linspace(0, 1, len(self.time))
                self.D[:,:,0] = np.random.multivariate_normal(mean = mean, cov = sd, size = self.Dsize) 
                for i in range(self.Dsize):
                    for j in range(1,np.size(self.time)):
                        self.D[i,:,j] = self.D[i,:, j-1] + np.dot(sd, np.random.multivariate_normal(mean = mean, 
                                    cov = np.eye((self.X).shape[1],(self.X).shape[1]) * np.sqrt( t[2] - t[1])
                                                                , size = 1).T).T + mean * (t[2] - t[1]) 

            elif (D == 'Brownian_bridge'):
                """ We build a dictionary from Brownian bridge.
                """
                mean = np.zeros(((self.X).shape[1]))
                sd = np.eye((self.X).shape[1],(self.X).shape[1])
                self.D = np.zeros((self.Dsize,(self.X).shape[1],len(self.time)))
                t = np.linspace(0, 1, len(self.time))
                for i in range(self.Dsize):
                    for k in range((self.X).shape[1]):
                        for j in range(1,(len(self.time)-1)):
                             self.D[i,k,j] = self.D[i,k, j-1] +  np.random.normal(0, np.sqrt(t[2]-t[1])
                                                        , size = 1) - self.D[i,k,j-1] * (t[2]-t[1]) / (1 - t[j])
                    
            elif (D == 'gaussian_wavelets'):  
                """ We build a dictionary from gaussian wavelets. We use a discretization on [-5,5]
                and add two random parameters to get an interesting dictionary. 
                The standard deviation sigma and a translationparameter K. The range of these 
                parameters are fixed.
                """
                t = np.linspace(-5,5,len(self.time))
                self.D = np.zeros((self.Dsize, (self.X).shape[1], len(self.time)))
                for i in range(self.Dsize):
                    for j in range((self.X).shape[1]): 
                        sigma = np.random.uniform(0.2,1)
                        K = np.random.uniform(-4,4)
                        for l in range(len(self.time)):
                            self.D[i,j,l] = (-(2 / (np.power(np.pi,0.25) * np.sqrt(3 * sigma)) ) 
                            * ((t[l]-K) ** 2 / (sigma ** 2) -1) * (np.exp(-(t[l] - K) ** 2 / (2 * sigma ** 2))))
                        
            elif (D == 'Dyadic_indicator'):
                """ We build a dictionary from the basis of the Haar wavelets using 
                only the father wavelets. We use a discretization on [0,1] since 
                we are interested only in the shape of the curves.
                """
                if (J_max == None):
                    J_max = 7
                a =0
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((np.sum(np.power(2,np.arange(J_max))) ** 2, (self.X).shape[1], len(self.time)))
                for J in range(J_max):
                    for j in range((self.X).shape[1]):
                        b = np.power(2,J)
                        for k in range(0,b):
                            for l in range(0,len(self.time)):
                                x = b * t[l] - k
                                self.D[a,j,l] = a*(0 <= x < 1)
                            a += 1
                        
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
                self.D = np.zeros((self.Dsize,(self.X).shape[1],len(self.time)))
                for i in range(self.Dsize):
                    for j in range((self.X).shape[1]):
                        freq = np.random.uniform(0, 10, 1)
                        amp = np.random.uniform(amplitude_min, amplitude_max, 1)
                        self.D[i,j,:] = amp * np.cos(2 * np.pi * freq * t)
                        
            elif (D == 'SinusCosinus'):
                """ We build a cosinus dictionary with random amplitudes and frequences.
                Amplitudes are fixed by the user while freq are fixed by the algorithm 
                with a large range to avoid overloading parameters.
                """
                if (amplitude_min == None):
                    amplitude_min = -1
                    
                if (amplitude_max == None):
                    amplitude_max = 1
                    
                t = np.linspace(0,1,len(self.time))
                self.D = np.zeros((self.Dsize,(self.X).shape[1],len(self.time)))
                for i in range(self.Dsize):
                    for j in range((self.X).shape[1]):
                        freq = np.random.uniform(0, 10, 1)
                        amp = np.random.uniform(amplitude_min, amplitude_max, 1)
                        choice = np.random.choice(np.array([0,1,]))
                        if (choice == 0):
                            self.D[i,j,:] = amp * np.cos(2 * np.pi * freq * t)
                        else:
                            self.D[i,j,:] = amp * np.sin(2 * np.pi * freq * t)
            elif ( D == 'Self'):
                self.D = self.X.copy()
            else: raise TypeError('This Dictionary is not pre-defined')
        else: self.D = D
            
        self.alpha = alpha
        self.step = np.diff(self.time)
        self.deriv_D = None
        self.deriv_X = None
        
        if not callable(innerproduct):
            """ Some inner product implemented.
            """                
            if (innerproduct == 'auto1'):
               
                
                if (self.alpha == None):
                    self.alpha = 1
                if (self.alpha == 1):
                    def innerproduct(x, y, xderiv = None, yderiv = None ):
                        """We build the inner product in the paper with alpha = 1 which corresponds 
                        to L2 dot product.
                        """ 
                        F1 = x * y 
                        A = 0
                        for i in range(F1.shape[0]):
                            A += np.sum((self.step * (F1[i][((np.arange(len(F1[i])) + 1) % len(F1[i]))[:len(F1[i])-1]]
                                + F1[i][((np.arange(len(F1[i])) + -1) % len(F1[i]))[1:len(F1[i])]]) / 2))
                        return A   
                        
                        
                        
                elif (self.alpha == 0):
                    self.deriv_X = np.zeros((self.X.shape[0], self.X.shape[1],self.X.shape[2]-1))
                    self.deriv_D = np.zeros((self.D.shape[0], self.D.shape[1],self.D.shape[2]-1))
                    for i in range(self.X.shape[0]):
                        self.deriv_X[i] = derivateM(self.X[i], self.step)
                    for i in range(self.D.shape[0]):
                        self.deriv_D[i] = derivateM(self.D[i], self.step)
                    def innerproduct(x,y, xderiv, yderiv):
                        """We build the inner product in the paper with alpha = 0 which corresponds 
                        to L2 of derivate dot product.
                        """ 
                        A = 0
                        F1 = x * y
                        F2 = xderiv * yderiv
                        F11 = np.zeros((F1.shape[0],F1.shape[1]-1))
                        F12 = np.zeros((F1.shape[0],F1.shape[1]-1))
                        F21 = np.zeros((F2.shape[0],F2.shape[1]-1))
                        F22 = np.zeros((F2.shape[0],F2.shape[1]-1))
                        for i in range(F1.shape[0]):
                            F11[i] = F1[i][((np.arange(len(F1[i])) + 1) % len(F1[i]))[:len(F1[i])-1]]
                            F12[i] = F1[i][((np.arange(len(F1[i])) + -1) % len(F1[i]))[1:len(F1[i])]]
                            F21[i] = F2[i][((np.arange(len(F2[i])) + 1) % len(F2[i]))[:len(F2[i])-1]]
                            F22[i] = F2[i][((np.arange(len(F2[i])) + -1) % len(F2[i]))[1:len(F2[i])]]
                            
                        for i in range(F1.shape[0]):
                            A += (self.alpha *np.sum(( self.step * (F11[i] + F12[i]) / 2))
                            +(1-self.alpha) * np.sum((self.step[0:(len(self.step) - 1)]
                                                      * (F21[i] + F22[i]) / 2)))
                        return A
                        
                else:
                    self.deriv_X = np.zeros((self.X.shape[0], self.X.shape[1],self.X.shape[2]-1))
                    self.deriv_D = np.zeros((self.D.shape[0], self.D.shape[1],self.D.shape[2]-1))
                    for i in range(X.shape[0]):
                        self.deriv_X[i] = derivateM(self.X[i], self.step)
                        self.deriv_D[i] = derivateM(self.D[i], self.step)

                    def innerproduct(x,y, xderiv, yderiv):
                        """We build the inner product in the paper which is a compromise between 
                        L2 scalar product and the L2 scalar product of derivate.
                        The function that we use work only with if we have the observations 
                        of curves at constant steps.
                        """ 
                        A = 0
                        F1 = x * y
                        F2 = xderiv * yderiv
                        F11 = np.zeros((F1.shape[0],F1.shape[1]-1))
                        F12 = np.zeros((F1.shape[0],F1.shape[1]-1))
                        F21 = np.zeros((F2.shape[0],F2.shape[1]-1))
                        F22 = np.zeros((F2.shape[0],F2.shape[1]-1))
                        x11 = np.zeros((x.shape[0],x.shape[1]-1))
                        x12 = np.zeros((x.shape[0],x.shape[1]-1))
                        x21 = np.zeros((xderiv.shape[0],xderiv.shape[1]-1))
                        x22 = np.zeros((xderiv.shape[0],xderiv.shape[1]-1))
                        y11 = np.zeros((y.shape[0],y.shape[1]-1))
                        y12 = np.zeros((y.shape[0],y.shape[1]-1))
                        y21 = np.zeros((yderiv.shape[0],yderiv.shape[1]-1))
                        y22 = np.zeros((yderiv.shape[0],yderiv.shape[1]-1))
                        for i in range(F1.shape[0]):
                            F11[i] = F1[i][((np.arange(len(F1[i])) + 1) % len(F1[i]))[:len(F1[i])-1]]
                            F12[i] = F1[i][((np.arange(len(F1[i])) + -1) % len(F1[i]))[1:len(F1[i])]]
                            F21[i] = F2[i][((np.arange(len(F2[i])) + 1) % len(F2[i]))[:len(F2[i])-1]]
                            F22[i] = F2[i][((np.arange(len(F2[i])) + -1) % len(F2[i]))[1:len(F2[i])]]
                            x11[i] = x[i][((np.arange(len(x[i])) + 1) % len(x[i]))[:len(x[i])-1]]
                            x12[i] = x[i][((np.arange(len(x[i])) + -1) % len(x[i]))[1:len(x[i])]]
                            x21[i] = xderiv[i][((np.arange(len(xderiv[i])) + 1) % len(xderiv[i]))[:len(xderiv[i])-1]]
                            x22[i] = xderiv[i][((np.arange(len(xderiv[i])) + -1) % len(xderiv[i]))[1:len(xderiv[i])]]                             
                            y11[i] = y[i][((np.arange(len(y[i])) + 1) % len(y[i]))[:len(y[i])-1]]
                            y12[i] = y[i][((np.arange(len(y[i])) + -1) % len(y[i]))[1:len(y[i])]]
                            y21[i] = yderiv[i][((np.arange(len(yderiv[i])) + 1) % len(yderiv[i]))[:len(yderiv[i])-1]]
                            y22[i] = yderiv[i][((np.arange(len(yderiv[i])) + -1) % len(yderiv[i]))[1:len(yderiv[i])]]





                        for i in range(F1.shape[0]):
                            A += (self.alpha * np.sum(F11[i] + F12[i]) / (np.sqrt(np.sum(x11[i] ** 2 +
                                x12[i] ** 2)) * np.sqrt(np.sum(y11[i] ** 2 + y12[i] ** 2)))
                               + (1 - self.alpha) * np.sum(F21[i] + F22[i]) / (np.sqrt(np.sum(x21[i] ** 2 + 
                                x22[i] ** 2)) * np.sqrt(np.sum(y21[i] ** 2 + y22[i] ** 2))))
                        return A
            elif (innerproduct == 'auto2'):
                if (self.alpha == None or self.alpha !=1):
                    self.alpha = 1
                def innerproduct(x, y, xderiv = None, yderiv = None):
                    """ We build the second type of generalization of the dot product in 
                         multivariate setting.
                    """
                    A = 0
                    for i in range(x.shape[1]):
                        A += np.inner(x[:,i],y[:,i])
                    return A 
                    
                            
            else: raise TypeError('This inner product is not pre-defined')
        else: self.alpha = 1 

        self.innerproduct = innerproduct
        self.limit = limit
        if limit is None:
            """Set limit to the default as specified by the paper
            (average depth of unsuccesful search through a binary tree).
            """ 
            self.limit = int(np.ceil(np.log2(self.sample)))  
            
        self.c = c_factor(self.sample)
        
        if (self.alpha == 1):
            for i in range(self.ntrees): 
                """This loop builds an ensemble of iTrees (the forest).
                """
                ix = np.random.choice(np.arange(self.nobjs), size = self.sample, replace = False)
                
                self.Trees.append(iTree(X[ix], self.step,  
                                        0, self.limit, 
                                        self.D, self.innerproduct, 
                                        self.alpha, self.deriv_X, 
                                        self.deriv_D))
        else:
            for i in range(self.ntrees): 
                """This loop builds an ensemble of iTrees (the forest).
                """
                ix = np.random.choice(np.arange(self.nobjs), size = self.sample, replace = False)
                
                self.Trees.append(iTree(X[ix], self.step, 
                                        0, self.limit, 
                                        self.D, self.innerproduct, 
                                        self.alpha, self.deriv_X[ix], 
                                        self.deriv_D))
    
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
                deriv_X_in = np.zeros((X_in.shape[0],X_in.shape[1],X_in.shape[2]-1))
                for i in range(X_in.shape[0]):
                    deriv_X_in[i] = derivateM(X_in[i], self.step)
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
                                         self.Trees[j],self.alpha).path * 1.0  
                
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
        return np.percentile(score_samples, 100 * (1-contamination), interpolation = 'lower')
    
    def predict_label(self, score, contamination = 0.1):
         
        """Compute the label vector of curves.  
        
        Parameters
        ----------
        
        score : Array
            The score array for a dataset of curves.
            
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
        return -1 + 2.0 * (score > self.threshold(score, contamination))


class Node(object): 
    """
    A single node from each tree (each iTree object). Nodes containe information on hyperplanes used for data division, date to be passed to left and right nodes, whether they are external or internal nodes.
    Attributes
    ----------
    e: int
        Depth of the tree to which the node belongs.
        
    size: int
        Size of the dataset present at the node.
        
    X: Array-like
        Data at the node.
        
    d: Array-like
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
        """
        Node(X, u, q, e, left, right, node_type = '' )
        Create a node in a given tree (iTree objectg)
        Parameters
        ----------
        X : Array-like
            Training data available to each node.
            
        d : Array
            Direction (curve) used to build the hyperplane that splits the data in the node.
            
        q : Array
            Intercept point for the hyperplane used for splitting data.
            
        left : Node object
            Left child node.
            
        right : Node object
            Right child node.
            
        node_type : str
            Specifies if the node is external or internal. Takes two values: 'exNode', 'inNode'.
        """
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
        
    d: list
        Normal vector at the root of this tree, which is used in creating hyperplanes for splitting criteria
        
    dd : int
        The index of the direction chosen at this node.
        
    q: list
        Intercept point at the root of this tree through which the splitting hyperplane passes.
        
    root: Node object
        At each node create a new tree.
        
    D: Array like
        Dictionary of functions used as directions.
        
    innerproduct : function or str  
        An inner product that we use for the construction of the tree.
        
    alpha : float
        A float number between [0,1] used in the innerproduct of the paper.

    deriv_X : Array-like
        A matrix of derivate of X if needed for the scalar product.
        
    deriv_D : Array-like
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
                 deriv_D=None):
        
        self.e = e
        self.X = X
        self.step = step
        self.size = len(X)
        self.l = l
        self.q = None                                       
        self.d = None 
        self.dd = None
        self.exnodes = 0
        self.D = D  
        self.innerproduct = innerproduct
        self.alpha = alpha
        self.deriv_X = deriv_X
        self.deriv_D = deriv_D
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
            idx = np.random.choice(range(0, (self.D).shape[0]), size=1)
            self.d = self.D[idx[0]]
            self.dd = idx[0]
            Z = np.zeros((sample_size))
            if (self.alpha != 1):
                for i in range(sample_size): 
                        Z[i] = self.innerproduct(X[i], self.d, self.deriv_X[i], self.deriv_D[idx[0]])
            else : 
                for i in range(sample_size): 
                        Z[i] = self.innerproduct(X[i], self.d)
            # Picking a random  threshold for the hyperplane splitting data.
            self.q = np.random.uniform(np.min(Z), np.max(Z)) 
            # Criteria that determines if a curve should go to the left or right child node.
            w = Z - self.q < 0                                                    
            return Node(X, self.d, self.dd, self.q, e,\
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
        
    x: Array like (dimension, discretization)
        A single function, which is represented as an matrix of floats.
        
    e: int
        The depth of a given node in the tree.
        
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
        """
        PathFactor(x, itree)
        Given a single tree (iTree objext) and a curve x, compute the legth of the path traversed 
        by the point on the tree when it reaches an external node.
        
        Parameters
        ----------
        x : Array 
            A single function x.
            
        itree : iTree object
            A single tree.
        """
        self.path_list=[]
        self.x = x
        self.deriv_x = deriv_x
        self.e = 0
        self.alpha = alpha
        self.step = step
        self.D = itree.D
        self.deriv_D = itree.deriv_D
        self.innerproduct = itree.innerproduct
        self.path  = self.find_path(itree.root)

    def find_path(self, T):
        """
        find_path(T)
        Given a tree, find the path for a single curve based on the splitting criteria stored at each node.
        
        Parameters
        ----------
        T : Itree object
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
            # Threshold for the hyperplane for splitting data at a given node.
            q = T.q 
            # Direction curve for the hyperplane for splitting data at a given node.
            d = T.d                                                             
            self.e += 1
            
            if (self.alpha != 1):
                if self.innerproduct(self.x, d, self.deriv_x, self.deriv_D[T.dd]) - q < 0:
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
