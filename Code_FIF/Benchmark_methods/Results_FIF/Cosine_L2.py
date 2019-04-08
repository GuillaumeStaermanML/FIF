""" Compute FIF score with cosinus dictionary and
    Sobolev scalar product on the thirteen real dataset of the paper.

    Author : Guillaume Staerman
"""
import csv
import pandas as pd
from FIF import *
from multiprocessing import Pool



chinatown = pd.read_csv('China_Train.csv', header = None)
chinatown2 = pd.read_csv('China_Test.csv', header = None)


X1_train = np.array(chinatown, dtype = float)[:,1:]
X1_test = np.array(chinatown2, dtype = float)[:,1:]
y1_train = np.array(chinatown, dtype=float)[:,0]
y1_test = np.array(chinatown2, dtype=float)[:,0]

coffee = pd.read_csv('Coffee_Train.csv', header = None)
coffee2 = pd.read_csv('Coffee_Test.csv', header = None)


X2_train = np.array(coffee, dtype = float)[:,1:]
X2_test = np.array(coffee2, dtype = float)[:,1:]
y2_train = np.array(coffee, dtype=float)[:,0]
y2_test = np.array(coffee2, dtype=float)[:,0]

ecgfivedays = pd.read_csv('ECGFiveDays_Train.csv', header = None)
ecgfivedays2 = pd.read_csv('ECGFiveDays_Test.csv', header = None)


X3_train = np.array(ecgfivedays, dtype = float)[:,1:]
X3_test = np.array(ecgfivedays2, dtype = float)[:,1:]
y3_train = np.array(ecgfivedays, dtype=float)[:,0]
y3_test = np.array(ecgfivedays2, dtype=float)[:,0]

ecg200 = pd.read_csv('ECG200_Train.csv', header = None)
ecg200_2 = pd.read_csv('ECG200_Test.csv', header = None)


X4_train = np.array(ecg200, dtype = float)[:,1:]
X4_test = np.array(ecg200_2, dtype = float)[:,1:]
y4_train = np.array(ecg200, dtype=float)[:,0]
y4_test = np.array(ecg200_2, dtype=float)[:,0]

handoutlines = pd.read_csv('Handoutlines_Train.csv', header = None)
handoutlines_2 = pd.read_csv('Handoutlines_Test.csv', header = None)


X5_train = np.array(handoutlines, dtype = float)[:,1:]
X5_test = np.array(handoutlines_2, dtype = float)[:,1:]
y5_train = np.array(handoutlines, dtype=float)[:,0]
y5_test = np.array(handoutlines_2, dtype=float)[:,0]

SonyRobotAI1 = pd.read_csv('SonyRobotAI1_Train.csv', header = None)
SonyRobotAI1_2 = pd.read_csv('SonyRobotAI1_Test.csv', header = None)


X6_train = np.array(SonyRobotAI1, dtype = float)[:,1:]
X6_test = np.array(SonyRobotAI1_2, dtype = float)[:,1:]
y6_train = np.array(SonyRobotAI1, dtype=float)[:,0]
y6_test = np.array(SonyRobotAI1_2, dtype=float)[:,0]


SonyRobotAI2 = pd.read_csv('SonyRobotAI2_Train.csv', header = None)
SonyRobotAI2_2 = pd.read_csv('SonyRobotAI2_Test.csv', header = None)


X7_train = np.array(SonyRobotAI2, dtype = float)[:,1:]
X7_test = np.array(SonyRobotAI2_2, dtype = float)[:,1:]
y7_train = np.array(SonyRobotAI2, dtype=float)[:,0]
y7_test = np.array(SonyRobotAI2_2, dtype=float)[:,0]

starlightcurves = pd.read_csv('StarLightCurves_Train.csv', header = None)
starlightcurves2  = pd.read_csv('StarLightCurves_Test1.csv', header = None)
starlightcurves3  = pd.read_csv('StarLightCurves_Test2.csv', header = None)

X8_train = np.array(starlightcurves, dtype = float)[:,1:]
y8_train = np.array(starlightcurves, dtype=float)[:,0]

X8_test1 = np.array(starlightcurves2, dtype = float)[:,1:]
X8_test2 = np.array(starlightcurves3, dtype = float)[:,1:]
X8_test = np.concatenate((X8_test1,X8_test2), axis = 0)
y8_test1 = np.array(starlightcurves2, dtype=float)[:,0]
y8_test2 = np.array(starlightcurves3, dtype=float)[:,0]
y8_test = np.concatenate((y8_test1,y8_test2))



twoleadECG = pd.read_csv('TwoLeadECG_Train.csv', header = None)
twoleadECG2  = pd.read_csv('TwoLeadECG_Test.csv', header = None)


X9_train = np.array(twoleadECG , dtype = float)[:,1:]
X9_test = np.array(twoleadECG2 , dtype = float)[:,1:]
y9_train = np.array(twoleadECG , dtype=float)[:,0]
y9_test = np.array(twoleadECG2 , dtype=float)[:,0]

yoga = pd.read_csv('Yoga_Train.csv', header = None)
yoga2  = pd.read_csv('Yoga_Test.csv', header = None)


X10_train = np.array(yoga , dtype = float)[:,1:]
X10_test = np.array(yoga2 , dtype = float)[:,1:]
y10_train = np.array(yoga , dtype=float)[:,0]
y10_test = np.array(yoga2 , dtype=float)[:,0]

y10_train[np.where(y10_train == 1)[0]] = -1
y10_train[np.where(y10_train == 2)[0]] = 1


EOGHorizontal = pd.read_csv('EOGHorizontal_Train.csv', header = None)
EOGHorizontal2  = pd.read_csv('EOGHorizontal_Test.csv', header = None)


X11_train = np.array(EOGHorizontal , dtype = float)[:,1:]
X11_test = np.array(EOGHorizontal2 , dtype = float)[:,1:]
y11_train = np.array(EOGHorizontal , dtype=float)[:,0]
y11_test = np.array(EOGHorizontal2 , dtype=float)[:,0]


CinECGTorso = pd.read_csv('CinECGTorso_Train.csv', header = None)
CinECGTorso2  = pd.read_csv('CinECGTorso_Test.csv', header = None)


X12_train = np.array(CinECGTorso , dtype = float)[:,1:]
X12_test = np.array(CinECGTorso2 , dtype = float)[:,1:]
y12_train = np.array(CinECGTorso , dtype=float)[:,0]
y12_test = np.array(CinECGTorso2 , dtype=float)[:,0]

ECG5000 = pd.read_csv('ECG5000_Train.csv', header = None)
ECG50002  = pd.read_csv('ECG5000_Test.csv', header = None)


X13_train = np.array(ECG5000 , dtype = float)[:,1:]
X13_test = np.array(ECG50002 , dtype = float)[:,1:]
y13_train = np.array(ECG5000 , dtype=float)[:,0]
y13_test = np.array(ECG50002 , dtype=float)[:,0]

l = [[X1_train, X1_test, y1_test], [X2_train, X2_test, y2_test], [X3_train, X3_test, y3_test],
	   [X4_train, X4_test, y4_test], [X5_train, X5_test, y5_test], [X6_train, X6_test, y6_test],
	   [X7_train, X7_test, y7_test], [X8_train, X8_test, y8_test], [X9_train, X9_test, y9_test],
	   [X10_train, X10_test, y10_test], [X11_train, X11_test, y11_test], [X12_train, X12_test, y12_test],
	   [X13_train, X13_test, y13_test]]

def bench5(l):
	np.random.seed(42)
	score5 = np.zeros((l[1].shape[0]))
	times = np.linspace(0,1,l[0].shape[1])
	psi = np.min(np.array((l[0].shape[0], 256)))
	F = FIForest(l[0], ntrees=100, time = times, subsample_size= psi,
	 D= 'cosinus', innerproduct= "auto", Dsize = 1000, alpha = 1)
	score5 = F.compute_paths(X_in = l[1])
	return	score5 

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(13)
    result5 = p.map(bench5, l) # list of length l of 4 list, result[0] give the 4 list for l=10




	

with open("Results_Cosine_L2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(result5)

