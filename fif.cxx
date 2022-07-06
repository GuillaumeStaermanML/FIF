#include "fif.hxx"


/********************************
	Utility functions
 ********************************/

inline std::vector<double> derivate (double* X1, double* time, int dim)
	/* return the derivative of the function X1 whose have been measured at times time.*/
	
{	std::vector<double> derivative (dim-1, 0.0);

	for (int i=1; i<dim; i++) derivative[i-1] = (X1[i] - X1[i-1]) / (time[i] - time[i-1]);

	return derivative;

}
inline std::vector<double> linspace(double  start, double end, int num)
	/* return an vector of 'num' equispaced values between 'start' and 'end'. */ 
{
  std::vector<double> linspaced;
  double delta = (end - start) / (num - 1);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); 

  return linspaced;
}

inline std::vector<double> dictionary_function (int dim, int dic_number, RANDOM_ENGINE& random_engine_in)
	/* return a function sampled from a dictionary. Three choices are possible: 
	*
	* 'dic_number=0' means Brownian motion
	* 'dic_num=1' means gaussian wavelets 
	* 'dic_number=2' means cosine dictionary.
	*/
{	
	std::vector<double> dic_function (dim, 0.0);
	std::vector<double> t (dim, 0.0);
	t = linspace(-5,5,dim);

	if (dic_number == 0) // Standard Brownian motion
		{
		dic_function[0] = std::normal_distribution<double> (0.0, 1.0) (random_engine_in);
		for (int i=1; i<dim; i++)
			{
				dic_function[i] = std::normal_distribution<double> (0.0, std::sqrt(t[i] - t[i-1]))(random_engine_in);
			}
		}
	else if (dic_number == 1) // gaussian wavelets with various mean and std
		{	

		double sigma;
		double K;


		sigma = std::uniform_real_distribution<double> (0.2, 1)(random_engine_in);
		K = std::uniform_real_distribution<double> (-4.0, 4.0)(random_engine_in);
		for (int i=0; i<dim; i++)
			{
			dic_function[i] = -(2 / (std::pow(PI_CONSTANT, 0.25) * std::sqrt(3 * sigma))) * (std::pow(t[i] - K, 2.0) / std::pow(sigma,2.0) - 1) * (std::exp(-(std::pow(t[i]-K, 2.0) / (2 * std::pow(sigma, 2.0)))));
			}
		}
	else if (dic_number == 2) // cosine with various frequencies and amplitude

		{	double ampl = 0.0;
			double freq = 0.0;
			ampl = std::uniform_real_distribution<double> (-1, 1)(random_engine_in);
			freq = std::uniform_real_distribution<double> (0, 10)(random_engine_in);

		for (int i=0; i<dim; i++)
			{


			dic_function[i] = ampl * std::cos(2 * PI_CONSTANT * freq * t[i]);
			}
		}
	else
		{
		std::cout << "this dictionary is not defined";
		}

	return dic_function;


}


inline double inner_product (double* X1, double* X2, double* time, double alpha, int dim)
	/* Return the innerproduct between X1 and X2 as a convex combination
	 *between L2 innerproduct and the L2 innerproduct of derivatives. 
	 *
	 * 'alpha=1' corresponds to L2 innerproduct
	 * 'alpha=0.5' corresponds to the Sobolev innerproduct
	 * 'alpha=0' corresponds to the derivative innerproduct. 
	*/
{
	double result = 0.0;
	std::vector<double> prod (dim, 0.0);

	for (int i=0; i<dim; i++);

	if (alpha == 1)
		{
		prod[0] = X1[0] * X2[0];	
		for (int i=1; i<dim; i++)
			{ 
			prod[i] = X1[i] * X2[i];
			result += (time[i] - time[i-1]) * (prod[i] + prod[i-1]) / 2.0;
			}

		}
	else if (alpha == 0)
		{
		std::vector<double> prod_derivate (dim-1, 0.0);
		std::vector<double> X1_derivate (dim-1, 0.0);
		std::vector<double> X2_derivate (dim-1, 0.0);

		X1_derivate = derivate(X1, time, dim);
		X2_derivate = derivate(X2, time, dim);		
		prod_derivate[0] = X1_derivate[0] * X2_derivate[0];

		for (int i=1; i<dim-1; i++) 
			{
			prod_derivate[i] = X1_derivate[i] * X2_derivate[i];
			result += (time[i] - time[i-1]) * (prod_derivate[i] + prod_derivate[i-1]) / 2.0;
			}
		}

	else
		{
		std::vector<double> prod_derivate (dim-1, 0.0);
		std::vector<double> step_time (dim-1, 0.0);
		std::vector<double> X1_derivate (dim-1, 0.0);
		std::vector<double> X2_derivate (dim-1, 0.0);
		double inner = 0.0;
		double inner_derivate = 0.0;
		double norm_X1 = 0.0;
		double norm_X2 = 0.0;
		double norm_X1_derivate = 0.0;
		double norm_X2_derivate = 0.0;


		prod[0] = X1[0] * X2[0];
		for (int i=1; i<dim; i++)
			{ 
			prod[i] = X1[i] * X2[i];
			step_time[i-1] = time[i] - time[i-1];
			inner += step_time[i-1] * (prod[i] + prod[i-1]) / 2.0;
			norm_X1 += step_time[i-1] * (std::pow (X1[i], 2.0) + std::pow (X1[i-1], 2.0)) / 2.0;
			norm_X2 += step_time[i-1] * (std::pow (X2[i], 2.0) + std::pow (X2[i-1], 2.0)) / 2.0;
			}

		X1_derivate = derivate(X1, time, dim);
		X2_derivate = derivate(X2, time, dim);
		prod_derivate[0] = X1_derivate[0] * X2_derivate[0];

		for (int i=1; i<dim-1; i++) 
			{
			prod_derivate[i] = X1_derivate[i] * X2_derivate[i];
			inner_derivate += step_time[i-1]  * (prod_derivate[i] + prod_derivate[i-1]) / 2.0;
			norm_X1_derivate += step_time[i-1]  * (std::pow (X1_derivate[i], 2.0) + std::pow (X1_derivate[i-1], 2.0)) / 2.0;
			norm_X2_derivate += step_time[i-1]  * (std::pow (X2_derivate[i], 2.0) + std::pow (X2_derivate[i-1], 2.0)) / 2.0;
			}
		result = alpha * inner / (std::sqrt (norm_X1) * std::sqrt (norm_X2)) + (1 - alpha) * inner_derivate / (std::sqrt (norm_X1_derivate) * std::sqrt (norm_X2_derivate));
		}			
	
	return result;

}


inline double c_factor (int N)
	/* Constant factor of the average depth of trees. */ 
{

	double Nd = (double) N;
	double result;
	result = 2.0*(log(Nd-1.0)+EULER_CONSTANT) - 2.0*(Nd-1.0)/Nd;
	return result;

}

inline std::vector<int> sample_without_replacement (int k, int N, RANDOM_ENGINE& gen)
	/* Sample k elements from the range [1, N] without replacement  */
{

    // Create an unordered set to store the samples
    std::unordered_set<int> samples;

    // Sample and insert values into samples
    for (int r=N-k+1; r<N+1; ++r)
    {
        int v = std::uniform_int_distribution<>(1, r)(gen);
        if (!samples.insert(v).second) samples.insert(r);
    }

    // Copy samples into vector
    std::vector<int> result(samples.begin(), samples.end());

    // Shuffle vector
    std::shuffle(result.begin(), result.end(), gen);

    return result;

}

void output_tree_node (Node* node_in, std::string string_in)
{

	std::cout << "==== Node ====" << std::endl;
	std::cout << "path: " 	<< string_in << std::endl;
	std::cout << "e   : " 	<< node_in[0].e << std::endl;
	std::cout << "size: " 	<< node_in[0].size << std::endl;
	std::cout << "n   : [";
	int size_n = node_in[0].dic_vector.size(); 
	for (int i=0; i<size_n; i++)
	{
		std::cout << node_in[0].dic_vector[i];
		if (i<size_n-1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;
	std::cout << node_in[0].treshold;
	std::cout << "]" << std::endl;
	std::cout << "type: " << node_in[0].node_type << std::endl;

	if (node_in[0].node_type == "exNode") return;
	else
	{
		output_tree_node (node_in[0].left, string_in.append(" L"));
		string_in.pop_back();
		output_tree_node (node_in[0].right, string_in.append("R"));
	}

}

void delete_tree_node (Node* node_in)
{

	if (node_in[0].node_type == "exNode") delete node_in;
	else
	{
		delete_tree_node (node_in[0].left);
		delete_tree_node (node_in[0].right);
		delete node_in;
	}

}


/****************************
        Class Node
 ****************************/
Node::Node (int size_in, int dim_in, double* dic_vector_in, double treshold_in, int e_in, Node* left_in, Node* right_in, std::string node_type_in)
{

	e = e_in;
	size = size_in;	
	treshold = treshold_in;
	left = left_in;
	right = right_in;
	node_type = node_type_in;
	for (int i=0; i<dim_in; i++) dic_vector.push_back(dic_vector_in[i]);

}

Node::~Node ()
{

}


/****************************
        Class FiTree
 ****************************/
FiTree::FiTree ()
{
	root = NULL;
}

FiTree::~FiTree ()
{

}

void FiTree::build_tree (double* X_in, double* time_in, int size_in, int e_in, int limit_in, double alpha_in, int dic_number_in, int dim_in, RANDOM_ENGINE& random_engine_in)
{

	e = e_in;
	size = size_in;
	dim = dim_in;
	dic_number = dic_number_in;
	alpha = alpha_in;
	limit = limit_in;
	exnodes = 0;
	root = add_node (X_in, time_in, size_in, e_in, random_engine_in);

}

Node* FiTree::add_node (double* X_in, double* time_in, int size_in, int e_in, RANDOM_ENGINE& random_engine_in)
{

	e = e_in;
	double treshold=0.0;
	std::vector<double> dic_vector (dim, 0.0);

	if (e_in >= limit || size_in <= 1) {

		Node* left = NULL;
		Node* right = NULL;
		exnodes += 1;
		Node* node = new Node (size_in, dim, &dic_vector[0], treshold, e_in, left, right, "exNode");
		return node;

	} else {

		std::vector<double> innerprod (size_in, 0.0);
		std::vector<double> XL, XR;
		int sizeXL = 0;
		int sizeXR = 0;

		dic_vector = dictionary_function(dim, dic_number, random_engine_in);
		for (int i=0; i<size_in; i++)
		{
			int index = i*dim;
			innerprod[i] = inner_product (&X_in[index], &dic_vector[0],time_in, alpha, dim);
		}

		// Pick a random point between min and max of the projections
		double innermin; double innermax; 

		innermin = *std::min_element(std::begin(innerprod), std::end(innerprod));
		innermax = *std::max_element(std::begin(innerprod), std::end(innerprod));
		treshold = std::uniform_real_distribution<double> (innermin, innermax)(random_engine_in);

		// Assign data in left and right leaves.
		for (int i=0; i<size_in; i++)
		{ 	int index = i*dim;
			if (innerprod[i] < treshold) {
				for (int j=0; j<dim; j++) XL.push_back(X_in[j+index]);
				sizeXL += 1;
			} else {
				for (int j=0; j<dim; j++) XR.push_back(X_in[j+index]);
				sizeXR += 1;
			}
		
		}	

		Node* left = add_node (&XL[0], time_in, sizeXL, e_in+1, random_engine_in);
		Node* right = add_node (&XR[0], time_in, sizeXR, e_in+1, random_engine_in);

		Node* node = new Node (size_in, dim, &dic_vector[0], treshold, e_in, left, right, "inNode");
		return node;

	}

}


/*************************
        Class Path
 *************************/
Path::Path (double* time_in, int dim_in, double alpha_in, double* x_in, FiTree fitree_in)
{

	dim = dim_in;
	alpha = alpha_in;
	x = x_in;
	time = time_in;
	e = 0.0;
	pathlength = find_path (fitree_in.root);

}

Path::~Path ()
{

}

double Path::find_path (Node* node_in)
{

	if (node_in[0].node_type == "exNode") {

		if (node_in[0].size <= 1) {
			return e;
		} else {
			e = e + c_factor (node_in[0].size);
			return e;
		}

	} else {

		e += 1.0;

		double xdotn, treshold, plength;
		treshold = node_in[0].treshold;
		xdotn = inner_product (x, &node_in[0].dic_vector[0], time, alpha, dim);
		if (xdotn < treshold) {
			path_list.push_back('L');
			plength = find_path (node_in[0].left);
		} else {
			path_list.push_back('R');
			plength = find_path (node_in[0].right);
		}
		return plength;

	}

}


/****************************
        Class FiForest
 ****************************/
FiForest::FiForest (int sample_in, int ntrees_in=100, int limit_in=0,  int random_seed_in=-1, int dic_number_in=1, double alpha_in=1.0)
{

	ntrees = ntrees_in;
	dic_number = dic_number_in;
	sample = sample_in;
	limit = limit_in;
	alpha = alpha_in;
	if (limit_in <= 0) limit = (int) ceil(log2(sample)); 
	c = c_factor (sample);
	Trees = new FiTree [ntrees];
	if (random_seed_in < 0) {
		RANDOM_SEED_GENERATOR random_seed_generator;
		random_seed = random_seed_generator();
	} else {
		random_seed = (unsigned) random_seed_in;
	}

}

FiForest::~FiForest ()
{

	for (int i=0; i<ntrees; i++)
		if (Trees[i].root != NULL) delete_tree_node (Trees[i].root);
	delete [] Trees;

}


bool FiForest::CheckSampleSize ()
{

	if (sample < 1)
	{
		std::cout << "Subsample size must be an integer between 1 and " << nobjs << "." << std::endl;
		return false;
	}
	if (sample > nobjs)
	{
		std::cout << "No. of data points is " << nobjs << ". Subsample size cannot be larger than " << nobjs << "." << std::endl;
		return false;
	}

	return true;

}

void FiForest::fit (double* X_in, double* time_in,  int nobjs_in, int dim_in)
{
	std::vector<double> Xsubset;

	X = X_in;
	time = time_in;
	nobjs = nobjs_in;
	dim = dim_in;
	if (!CheckSampleSize ()) return;



	for (int i=0; i<ntrees; i++)
	{
		/* Select a random subset of X_in of size sample_in */
		RANDOM_ENGINE random_engine (random_seed+i);
		std::vector<int> sample_index = sample_without_replacement (sample, nobjs, random_engine);
		Xsubset.clear();
		for (int j=0; j<sample; j++)
		{
			for (int k=0; k<dim; k++)
			{
				int index = k+(sample_index[j]-1)*dim;
				Xsubset.push_back(X[index]);
			}
		}

		Trees[i].build_tree (&Xsubset[0], time, sample, 0, limit, alpha, dic_number, dim, random_engine );
	}

}

void FiForest::predict (double* S,  double* X_in=NULL, int size_in=0)
{

	if (X_in == NULL)
	{
		X_in = X;
		size_in = nobjs;
	}

	double htemp, havg;
	for (int i=0; i<size_in; i++)
	{
		htemp = 0.0;
		for (int j=0; j<ntrees; j++)
		{
			Path path (time, dim, alpha, &X_in[i*dim], Trees[j]);
			htemp += path.pathlength;
		}
		havg = htemp/ntrees;
		S[i] = std::pow(2.0, -havg/c);
	}

}
void FiForest::predictSingleTree (double* S, double* X_in=NULL, int size_in=0, int FiTree_index=0)
{

	if (X_in == NULL)
	{
		X_in = X;
		size_in = nobjs;
	}

	double htemp;
	for (int i=0; i<size_in; i++)
	{
		htemp = 0.0;
		Path path (time, dim, alpha,  &X_in[i*dim], Trees[FiTree_index]);
		htemp = path.pathlength;
		S[i] = htemp;
	}

}

void FiForest::OutputTreeNodes (int FiTree_index)
{

	output_tree_node (Trees[FiTree_index].root, "root");

}
