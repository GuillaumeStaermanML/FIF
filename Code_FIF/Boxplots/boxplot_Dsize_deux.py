""" Boxplot on the second dataset w.r.t. the dictionary size.

    Author : Guillaume Staerman
"""


from FIF import *
import csv
from multiprocessing import Pool
import pandas as pd






m = 200
#with open('variance_study_score_n_dataset.csv', 'rb') as f:
#    X = np.array([[float(e) for e in row] for row in csv.reader(f, quoting=csv.QUOTE_NONE)])

Y = pd.read_csv('variance_study_score_n_dataset2.csv', header = None)

X = Y.as_matrix()


times = np.linspace(0,1,m)
score1 = []
score2 = []
score3 = []
score4 = []
Dsize = np.array([50, 100, 200, 500, 750, 1000, 1500, 2000, 5000, 10000])


def boucle(Dsize):
    for k in range(1000):
        F = FIForest(X, ntrees=100, time = times, subsample_size= 64, 
         D= 'gaussian_wavelets', innerproduct= "auto", Dsize = Dsize, alpha = 1)
        S = F.compute_paths()
        score1.append(S[0])
        score2.append(S[1])
        score3.append(S[2])
        score4.append(S[3])
    print('bonjour')
    return score1, score2, score3, score4

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(10)
    result = p.map(boucle, Dsize) # list of length l of 4 list, result[0] give the 4 list for l=10

A = np.zeros((1000,len(Dsize)))
B = np.zeros((1000,len(Dsize)))
C = np.zeros((1000,len(Dsize)))
D = np.zeros((1000,len(Dsize)))

for i in range(len(Dsize)):
	A[:,i] = result[i][0]

for i in range(len(Dsize)):
	B[:,i] = result[i][1]

for i in range(len(Dsize)):
	C[:,i] = result[i][2]  

for i in range(len(Dsize)):
	D[:,i] = result[i][3]     	

with open("variance_study_score_Dsize_deux_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A)
with open("variance_study_score_Dsize_deux_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B)
with open("variance_study_score_Dsize_deux_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(C)
with open("variance_study_score_Dsize_deux_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(D)

