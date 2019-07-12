FIF : Functional Isolation Forest
=========================================

This repository hosts Python code of the Functional Isolation Forest algorithms and its extension to Multivariate functional data. Here we provide the source code for the algorithms as well as example notebooks to help get started.


=========================================


Installation
------------

To get the latest version of the code::

  $ git clone https://github.com/Gstaerman/FIF.git
  
Algorithm
---------
Functional Isolation Forest is an anomaly detection (and anomaly ranking) algorithm for functional data.
It shows a great flexibility to distinguish most of anomaly types of functional data.

Some parameters have to be set by the user : 
                                    - innerproduct (one can fix 'auto' and vary alpha parameter to play with FIF) 
                                    - D (Dictionary)
                                    - time (vector time of discretization points) 
                                    
See the documentation of FIF.py to get more informations on innerproduct and dictionary possibilities.                                 

Quick Start :
------------

Create a toy dataset :

.. code:: python


  import numpy as np 
  np.random.seed(42)
  m =100
  n =100
  tps = np.linspace(0,1,m)
  v = np.linspace(1,1.4,n)
  X = np.zeros((n,m))
  for i in range(n):
      X[i] = 30 * ((1-tps) ** v[i]) * tps ** v[i]


  Z1 = np.zeros((m))
  for j in range(m):
      if (tps[j]<0.2 or tps[j]>0.8):
          Z1[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 
      else:
          Z1[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + np.random.normal(0,0.3,1)
  Z1[0] = 0
  Z1[m-1] = 0


  Z2 = 30 * ((1-tps) ** 1.6) * tps ** 1.6


  Z3 = np.zeros((m))
  for j in range(m):
      Z3[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + np.sin(2*np.pi*tps[j])

  Z4 = np.zeros((m))
  for j in range(m):
      Z4[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2

  for j in range(70,71):
      Z4[j] += 2

  Z5 = np.zeros((m))
  for j in range(m):
      Z5[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + 0.5*np.sin(10*np.pi*tps[j])

  X = np.concatenate((X,Z1.reshape(1,-1),Z2.reshape(1,-1),  
                       Z3.reshape(1,-1), Z4.reshape(1,-1), Z5.reshape(1,-1)), axis = 0)


   
And then use FIF to ranking functional dataset :

.. code:: python

  np.random.seed(42)
  F  = FIForest(X, D="gaussian_wavelets", time=tps, innerproduct="auto", alpha=0.5)
  S  = F.compute_paths()
    
The simulated dataset with the five introduced anomalies (left). The sorted dataset (middle), the darker the color, the more the curves are considered anomalies. The sorted anomaly score of the dataset (right). 

.. image:: anomaly_example.png
.. image:: anomaly_example_rank.png
.. image:: anomaly_example_score.png

Dependencies
------------

These are the dependencies to use FIF:

* numpy 


Cite
----

If you use this code in your project, please cite::

   Functional Isolation Forest   
   Guillaume Staerman, Pavlo Mozharovskyi, Stéphan Clémençon, Florence d'Alché-Buc. 
   (submitted), 2019.
   https://arxiv.org/abs/1904.04573 

  
