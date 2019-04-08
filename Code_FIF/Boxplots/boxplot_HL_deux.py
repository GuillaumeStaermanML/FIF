""" Boxplot on the second dataset w.r.t. the height limit.

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

l = np.array([3, 4, 5, 6, 7, 8])


def boucleHH(l):
    np.random.seed(42)
    score1 = []
    score2 = []
    score3 = []
    score4 = []
    for k in range(100):
        F = FIForest(X, ntrees=100, time = times, subsample_size= 64, limit = l,
         D= 'Haar_wavelets_father', innerproduct= "auto", Dsize = 1000, alpha = 1)
        S = F.compute_paths()
        score1.append(S[0])
        score2.append(S[1])
        score3.append(S[2])
        score4.append(S[3])
    print('bonjour')
    return score1, score2, score3, score4

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(6)
    resultHH = p.map(boucleHH, l) # list of length l of 4 list, result[0] give the 4 list for l=10

AHH = np.zeros((100,len(l)))
BHH = np.zeros((100,len(l)))
CHH = np.zeros((100,len(l)))
DHH = np.zeros((100,len(l)))

for i in range(len(l)):
	AHH[:,i] = resultHH[i][0]

for i in range(len(l)):
	BHH[:,i] = resultHH[i][1]

for i in range(len(l)):
	CHH[:,i] = resultHH[i][2]  

for i in range(len(l)):
	DHH[:,i] = resultHH[i][3]     	

with open("variance_study_score_HL_deux_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(AHH)
with open("variance_study_score_HL_deux_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(BHH)
with open("variance_study_score_HL_deux_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(CHH)
with open("variance_study_score_HL_deux_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(DHH)

