""" Boxplot on the second dataset w.r.t. the size of the sample.

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
        a2[i] = 0.05 * np.random.random()
        for j in range(0,m):
                B[i,j] = a1[i] * np.cos(tps[j] *
                 2 * np.pi) + a2[i] * np.sin(tps[j] * 2 * np.pi)
    return B;


m = 200
#with open('variance_study_score_n_dataset.csv', 'rb') as f:
#    X = np.array([[float(e) for e in row] for row in csv.reader(f, quoting=csv.QUOTE_NONE)])

x= pd.read_csv('variance_study_score_n_dataset2.csv', header = None)

x = x.as_matrix()
X_ano = x[:4] # four curves where we compute the depth.

times = np.linspace(0,1,m)
l = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])

def boucleX(l):
    np.random.seed(42)
    score10 = []
    score20 = []
    score30 = []
    score40 = []
    for k in range(100):
        Y = simul_simulate(n = l-4, m = 200, T = 1)
        Z = np.concatenate((X_ano,Y), axis = 0)
        F = FIForest(Z, ntrees=100, time = times, subsample_size= np.min(np.array([64, l])),
         D= 'Dyadic_indicator', innerproduct= "auto", alpha = 1)
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
    resultX = p.map(boucleX, l) # list of length l of 4 list, result[0] give the 4 list for l=10

AX = np.zeros((100,len(l)))
BX = np.zeros((100,len(l)))
CX = np.zeros((100,len(l)))
DX = np.zeros((100,len(l)))

for i in range(len(l)):
    AX[:,i] = resultX[i][0]

for i in range(len(l)):
    BX[:,i] = resultX[i][1]

for i in range(len(l)):
    CX[:,i] = resultX[i][2]  

for i in range(len(l)):
    DX[:,i] = resultX[i][3]         

with open("variance_study_score_nsize_deux_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(AX)
with open("variance_study_score_nsize_deux_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(BX)
with open("variance_study_score_nsize_deux_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(CX)
with open("variance_study_score_nsize_deux_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(DX)

