""" Boxplot on the first dataset w.r.t. the height limit.

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

l = np.array([3, 4, 5, 6, 7, 8])


def boucleHL(l):
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
    resultHL = p.map(boucleHL, l) # list of length l of 4 list, result[0] give the 4 list for l=10

AHL = np.zeros((100,len(l)))
BHL = np.zeros((100,len(l)))
CHL = np.zeros((100,len(l)))
DHL = np.zeros((100,len(l)))

for i in range(len(l)):
	AHL[:,i] = resultHL[i][0]

for i in range(len(l)):
	BHL[:,i] = resultHL[i][1]

for i in range(len(l)):
	CHL[:,i] = resultHL[i][2]  

for i in range(len(l)):
	DHL[:,i] = resultHL[i][3]     	

with open("variance_study_score_HL_0_1.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(AHL)
with open("variance_study_score_HL_0_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(BHL)
with open("variance_study_score_HL_0_3.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(CHL)
with open("variance_study_score_HL_0_4.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(DHL)

