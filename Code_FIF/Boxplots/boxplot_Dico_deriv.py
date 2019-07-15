""" Boxplot on the first dataset w.r.t. the different
    dictionaries with L2 scalar product of derivative.

    Author : Guillaume Staerman
"""


from FIF import *
import csv
from multiprocessing import Pool
import pandas as pd







m = 500
#with open('variance_study_score_n_dataset.csv', 'rb') as f:
#    X = np.array([[float(e) for e in row] for row in csv.reader(f, quoting=csv.QUOTE_NONE)])

Y = pd.read_csv('variance_study_score_n_dataset.csv', header = None)

X = Y.as_matrix()
X = X[:500]

times = np.linspace(0,1,m)
score1111 = []
score2222 = []
score3333 = []
score4444 = []
Dico = ["gaussian_wavelets", "Brownian", "cosinus", "Multiresolution_linear", "linear_indicator_uniform" , "Brownian_bridge", "Self"]


def boucle(Dico):
    np.random.seed(42)
    for k in range(100):
        F = FIForest(X, ntrees=100, time = times, subsample_size= 64, 
         D= Dico, innerproduct= "auto",  alpha = 0)
        S = F.compute_paths()
        score1111.append(S[0])
        score2222.append(S[1])
        score3333.append(S[2])
        score4444.append(S[3])
    print('bonjour')
    return score1111, score2222, score3333, score4444

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(7)
    result = p.map(boucle, Dico) # list of length l of 4 list, result[0] give the 4 list for l=10

A = np.zeros((100,len(Dico)))
B = np.zeros((100,len(Dico)))
C = np.zeros((100,len(Dico)))
D = np.zeros((100,len(Dico)))

for i in range(len(Dico)):
	A[:,i] = result[i][0]

for i in range(len(Dico)):
	B[:,i] = result[i][1]

for i in range(len(Dico)):
	C[:,i] = result[i][2]  

for i in range(len(Dico)):
	D[:,i] = result[i][3]     	

with open("variance_study_score_Dico_deriv_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A)
with open("variance_study_score_Dico_deriv_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B)
with open("variance_study_score_Dico_deriv_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(C)
with open("variance_study_score_Dico_deriv_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(D)

