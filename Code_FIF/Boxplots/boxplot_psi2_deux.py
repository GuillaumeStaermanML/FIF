""" Boxplot on the second dataset w.r.t. the size of the subsample.

    Author : Guillaume Staerman
"""

from FIF import *
import csv
from multiprocessing import Pool
import pandas as pd



def simul_simulate(n = 100, m = 200, T = 1):
    tps = np.linspace(0,T,m) # Discrétisation du temps
    B = np.zeros((n,m))
    a1 = np.zeros((n))
    a2 = np.zeros((n))
    for i in range(0,n):
        a1[i] = 0.05 * np.random.random()
        #random.seed(42)
        a2[i] = 0.05 * np.random.random()
        #print(a1[i])
        #print(a2[i])
        for j in range(0,m):
                B[i,j] = a1[i] * np.cos(tps[j] *
                 2 * np.pi) + a2[i] * np.sin(tps[j] * 2 * np.pi)
    return B;


Y = pd.read_csv('variance_study_score_n_dataset2.csv', header = None)

X = Y.as_matrix()
x = X[:4]
y = simul_simulate(n = 496, m=200, T=1)

Z = np.concatenate((x,y), axis = 0)

X00 = Z.copy()


m = 200

times = np.linspace(0,1,m)
psi00 = np.array([16, 32, 64, 128, 256, 350, 500])


def boucle00(psi):
    np.random.seed(42)
    score100 = []
    score200 = []
    score300 = []
    score400 = []
    for k in range(100):
        F = FIForest(X00, ntrees=100, limit = 7 , time = times, subsample_size= psi,
         D= 'gaussian_wavelets', innerproduct= "auto", Dsize = 1000, alpha = 1)
        S = F.compute_paths()
        score100.append(S[0])
        score200.append(S[1])
        score300.append(S[2])
        score400.append(S[3])
    print('bonjour')
    return score100, score200, score300, score400

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(7)
    result00 = p.map(boucle00, psi00) # list of length l of 4 list, result[0] give the 4 list for l=10

A00 = np.zeros((100,len(psi00)))
B00 = np.zeros((100,len(psi00)))
C00 = np.zeros((100,len(psi00)))
D00 = np.zeros((100,len(psi00)))

for i in range(len(psi00)):
	A00[:,i] = result00[i][0]

for i in range(len(psi00)):
	B00[:,i] = result00[i][1]

for i in range(len(psi00)):
	C00[:,i] = result00[i][2]  

for i in range(len(psi00)):
	D00[:,i] = result00[i][3]     	

with open("variance_study_score_psi2_deux_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A00)
with open("variance_study_score_psi2_deux_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B00)
with open("variance_study_score_psi2_deux_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(C00)
with open("variance_study_score_psi2_deux_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(D00)

