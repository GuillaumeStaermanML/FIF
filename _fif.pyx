# Cython wrapper for Functional Isolation Forest
# This code is highly inspired from the code of 'Extended Isolation Forest' https://github.com/sahandha/eif.

# distutils: language = C++
# distutils: sources  = fif.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np
from version import __version__

cimport __fif

np.import_array()

cdef class FiForest:
    cdef int size_X
    cdef int dim
    cdef int _ntrees
    cdef int _limit
    cdef int sample
    cdef int tree_index
    cdef int dic_number
    cdef double alpha
    cdef __fif.FiForest* thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__ (self, np.ndarray[double, ndim=2] X not None, np.ndarray[double, ndim=1] time not None,  int sample_size, int ntrees=100, int limit=0,  int seed=-1, int dic_number=1, double alpha=1.0):
        self.thisptr = new __fif.FiForest (ntrees, sample_size, limit, seed, dic_number, alpha)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        if not time.flags['C_CONTIGUOUS']:
            time = time.copy(order='C')
        self.size_X = X.shape[0]
        self.dim = X.shape[1]
        self.sample = sample_size
        self._ntrees = ntrees
        self._limit = self.thisptr.limit
        self.alpha = alpha
        self.dic_number = dic_number
        self.thisptr.fit (<double*> np.PyArray_DATA(X), <double*> np.PyArray_DATA(time), self.size_X, self.dim)

    @property
    def ntrees(self):
        return self._ntrees

    @property
    def limit(self):
        return self._limit

    def __dealloc__ (self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_paths (self, np.ndarray[double, ndim=2] X_in=None):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if X_in is None:
            S = np.empty(self.size_X, dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), NULL, 0)
        else:
            if not X_in.flags['C_CONTIGUOUS']:
                X_in = X_in.copy(order='C')
            S = np.empty(X_in.shape[0], dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X_in), X_in.shape[0])
        return S

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_paths_single_tree (self, np.ndarray[double, ndim=2] X_in=None, tree_index=0):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if X_in is None:
            S = np.empty(self.size_X, dtype=np.float64, order='C')
            self.thisptr.predictSingleTree (<double*> np.PyArray_DATA(S), NULL, 0, tree_index)
        else:
            if not X_in.flags['C_CONTIGUOUS']:
                X_in = X_in.copy(order='C')
            S = np.empty(X_in.shape[0], dtype=np.float64, order='C')
            self.thisptr.predictSingleTree (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X_in), X_in.shape[0], tree_index)
        return S

    def output_tree_nodes (self, int tree_index):
        self.thisptr.OutputTreeNodes (tree_index)
