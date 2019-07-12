""" Boxplot on the first dataset w.r.t. the size of the sample.

    Author : Guillaume Staerman
"""

from FIF import *
import csv
from multiprocessing import Pool
import pandas as pd


def simul_Brownien_Drift(n = 100, m = 1000, sigma = 1, mu = 0, T = 1):
    tps = np.linspace(0,T,m) # Discrétisation du temps
    B = np.zeros((n,m))
    B[:,0] = np.random.normal(0, scale = 0.5, size = n) # départ du MB à 0
    for i in range(1,np.size(tps)):
        B[:,i] = B[:, i-1] + sigma * np.random.normal(0,np.sqrt(tps[2] - tps[1]), n)+ mu * (tps[2] - tps[1]) 
    return B;


m = 500
#with open('variance_study_score_n_dataset.csv', 'rb') as f:
#    X = np.array([[float(e) for e in row] for row in csv.reader(f, quoting=csv.QUOTE_NONE)])

Y0 = pd.read_csv('variance_study_score_n_dataset.csv', header = None)

X0 = Y0.as_matrix()
X_ano = X0[:4] # four curves where we compute the depth.


times = np.linspace(0,1,m)
l = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])


def boucleA(l):
    np.random.seed(42)
    score10 = []
    score20 = []
    score30 = []
    score40 = []
    for k in range(100):
        Y = simul_Brownien_Drift(n = l-4, m = 500, T = 1)
        Z = np.concatenate((X_ano,Y), axis = 0)
        F = FIForest(Z, ntrees=100, time = times, subsample_size= np.min(np.array([64, l])),
         D= 'Dyadic_indicator', innerproduct= "auto",  alpha = 1)
        S = F.compute_paths()
        score10.append(S[0])
        score20.append(S[1])
        score30.append(S[2])
        score40.append(S[3])
    print('bonjour')
    return score10, score20, score30, score40

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(7)
    resultA = p.map(boucleA, l) # list of length l of 4 list, result[0] give the 4 list for l=10

AA = np.zeros((100,len(l)))
BB = np.zeros((100,len(l)))
CC = np.zeros((100,len(l)))
DD = np.zeros((100,len(l)))

for i in range(len(l)):
	AA[:,i] = resultA[i][0]

for i in range(len(l)):
	BB[:,i] = resultA[i][1]

for i in range(len(l)):
	CC[:,i] = resultA[i][2]  

for i in range(len(l)):
	DD[:,i] = resultA[i][3]     	

with open("variance_study_score_nsize_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(AA)
with open("variance_study_score_nsize_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(BB)
with open("variance_study_score_nsize_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(CC)
with open("variance_study_score_nsize_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(DD)

