#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

#define EULER_CONSTANT 0.5772156649
#define PI_CONSTANT 3.1415926535

#define RANDOM_ENGINE std::mt19937_64
#define RANDOM_SEED_GENERATOR std::random_device


/****************************
        Class Node
 ****************************/
class Node
{

    private:

    protected:

    public:
	    int e;
        int size;
        std::vector<double> dic_vector;
        double treshold;
        Node* left;
        Node* right;
        std::string node_type;

        Node (int, int, double*, double, int, Node*, Node*, std::string);
        ~Node ();

};


/****************************
        Class FiTree
 ****************************/
class FiTree
{

    private:
        int e;
        int size;
        double alpha;
        int dic_number;
        int dim;
        int limit;
        int exnodes;

    protected:

    public:
        Node* root;

        FiTree ();
        ~FiTree ();
        void build_tree (double*, double*,  int, int, int, double,int, int, RANDOM_ENGINE&);
        Node* add_node (double*, double*,  int, int, RANDOM_ENGINE&);

};


/*************************
        Class Path
 *************************/
class Path
{

    private:
        int dim;
        double alpha;
        double* time;
        double* x;
        double e;
    protected:

    public:
        std::vector<char> path_list;
        double pathlength;

        Path (double*, int, double, double*, FiTree);
        ~Path ();
        double find_path (Node*);

};


/****************************
        Class FiForest
 ****************************/
class FiForest
{

    private:
        int nobjs;
        int dim;
        int sample;
        int ntrees;
        int dic_number;
        double alpha;
        double* X;
        double * time;
        double c;
        FiTree* Trees;
        unsigned random_seed;

	bool CheckSampleSize ();
    protected:

    public:
        int limit;
        FiForest (int, int, int, int, int, double);
        ~FiForest ();
        void fit (double*, double*, int, int);
        void predict (double*, double*, int);
        void predictSingleTree (double*, double*, int, int);
	    void OutputTreeNodes (int);

};


/********************************
        Utility functions
 ********************************/
inline std::vector<double> derivate (double* , double*, int);
inline std::vector<double> linspace(double, double, int);
inline std::vector<double> dictionary_function (int , int, RANDOM_ENGINE&);
inline std::vector<int> sample_without_replacement (int, int, RANDOM_ENGINE&);
inline double inner_product (double*, double*, double*, double, int);
inline double c_factor (int);
void output_tree_node (Node*, std::string);
void delete_tree_node (Node*);
