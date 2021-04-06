cdef extern from "fif.hxx":
    cdef cppclass FiForest:
        int limit
        FiForest (int, int, int, int,  int, double)
        void fit (double*, double*, int, int)
        void predict (double*, double*,  int)
        void predictSingleTree (double*, double*, int, int)
        void OutputTreeNodes (int)
