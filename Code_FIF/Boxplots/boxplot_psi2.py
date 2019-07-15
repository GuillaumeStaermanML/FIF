""" Boxplot on the first dataset w.r.t. the size of the subsample.

    Author : Guillaume Staerman
"""


from FIF import *
import csv
from multiprocessing import Pool
import pandas as pd







m = 500
#with open('variance_study_score_n_dataset.csv', 'rb') as f:
#    X = np.array([[float(e) for e in row] for row in csv.reader(f, quoting=csv.QUOTE_NONE)])

Y0 = pd.read_csv('variance_study_score_n_dataset.csv', header = None)

X0 = Y0.as_matrix()
X0 = X0[:500]

times = np.linspace(0,1,m)
psi0 = np.array([16, 32, 64, 128, 256, 350, 500])


def boucle0(psi):
    np.random.seed(42)
    score10 = []
    score20 = []
    score30 = []
    score40 = []
    for k in range(100):
        F = FIForest(X0, ntrees=100, limit = 7 , time = times, subsample_size= psi,
         D= 'gaussian_wavelets', innerproduct= "auto",  alpha = 1)
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
    result0 = p.map(boucle0, psi0) # list of length l of 4 list, result[0] give the 4 list for l=10

A0 = np.zeros((100,len(psi0)))
B0 = np.zeros((100,len(psi0)))
C0 = np.zeros((100,len(psi0)))
D0 = np.zeros((100,len(psi0)))

for i in range(len(psi0)):
	A0[:,i] = result0[i][0]

for i in range(len(psi0)):
	B0[:,i] = result0[i][1]

for i in range(len(psi0)):
	C0[:,i] = result0[i][2]  

for i in range(len(psi0)):
	D0[:,i] = result0[i][3]     	

with open("variance_study_score_psi2_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A0)
with open("variance_study_score_psi2_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B0)
with open("variance_study_score_psi2_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(C0)
with open("variance_study_score_psi2_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(D0)

