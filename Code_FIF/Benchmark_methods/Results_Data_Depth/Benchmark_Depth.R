############################
# This script computes the functional stahel-Donoho-Outlyingness depth from [1] and the 
# random functional depth from [2] for the thirteen dataset in the (Functional Isolation Forest) paper.
############################

############################
# @Author : Guillaume Staerman
############################

###########################
# References :

# Hubert M., Rousseeuw P.J., Segaert P. (2017). Multivariate and functional classification using depth and distance.
# Advances in Data Analysis and Classification, 11, 445–466.

# Cuesta-Albertos, J.A. and Nieto-Reyes, A. (2008). The random Tukey depth. 
# Computational Statis- tics & Data Analysis 52 (11), 4979–4988.

library(readr)
library(roahd)
library(ggplot2)
library(fda)
library(ddalpha)
library(pROC)
library(mrfDepth) 

#####################
rm(list=objects())


#####################
###  Be careful  ####
#####################
# Write the directory where the datasets are :
setwd('~/Desktop/Code_FIF/Datasets')
#####################




#####################
#####
#Importation of datasets
#####
China_TRAIN <- read.delim("China_train.csv", header=FALSE, sep = ',')
China_TRAIN <-data.matrix(China_TRAIN, rownames.force = NA)
China_TEST <- read.delim("China_test.csv", header=FALSE, sep=',')
China_TEST <-data.matrix(China_TEST, rownames.force = NA)

lab_China_TEST <- China_TEST[,1]

ECGFiveDays_TRAIN <- read.delim("ECGFiveDays_train.csv", header=FALSE, sep = ',')
ECGFiveDays_TRAIN <-data.matrix(ECGFiveDays_TRAIN, rownames.force = NA)
ECGFiveDays_TEST <- read.delim("ECGFiveDays_test.csv", header=FALSE, sep=',')
ECGFiveDays_TEST <-data.matrix(ECGFiveDays_TEST, rownames.force = NA)

lab_ECGFiveDays_TEST <- ECGFiveDays_TEST[,1]

Coffee_TRAIN <- read.delim("Coffee_train.csv", header=FALSE, sep = ',')
Coffee_TRAIN <-data.matrix(Coffee_TRAIN, rownames.force = NA)
Coffee_TEST <- read.delim("Coffee_test.csv", header=FALSE, sep=',')
Coffee_TEST <-data.matrix(Coffee_TEST, rownames.force = NA)

lab_Coffee_TEST <- Coffee_TEST[,1]

ECG200_TRAIN <- read.delim("ECG200_train.csv", header=FALSE, sep = ',')
ECG200_TRAIN <-data.matrix(ECG200_TRAIN, rownames.force = NA)
ECG200_TEST <- read.delim("ECG200_test.csv", header=FALSE, sep=',')
ECG200_TEST <-data.matrix(ECG200_TEST, rownames.force = NA)

lab_ECG200_TEST <- ECG200_TEST[,1]

Handoutlines_TRAIN <- read.delim("Handoutlines_train.csv", header=FALSE, sep = ',')
Handoutlines_TRAIN <-data.matrix(Handoutlines_TRAIN, rownames.force = NA)
Handoutlines_TEST <- read.delim("Handoutlines_test.csv", header=FALSE, sep=',')
Handoutlines_TEST <-data.matrix(Handoutlines_TEST, rownames.force = NA)

lab_Handoutlines_TEST <- Handoutlines_TEST[,1]

SonyRobotAI1_TRAIN <- read.delim("SonyRobotAI1_train.csv", header=FALSE, sep = ',')
SonyRobotAI1_TRAIN <-data.matrix(SonyRobotAI1_TRAIN, rownames.force = NA)
SonyRobotAI1_TEST <- read.delim("SonyRobotAI1_test.csv", header=FALSE, sep=',')
SonyRobotAI1_TEST <-data.matrix(SonyRobotAI1_TEST, rownames.force = NA)

lab_SonyRobotAI1_TEST <- SonyRobotAI1_TEST[,1]

SonyRobotAI2_TRAIN <- read.delim("SonyRobotAI2_train.csv", header=FALSE, sep = ',')
SonyRobotAI2_TRAIN <-data.matrix(SonyRobotAI2_TRAIN, rownames.force = NA)
SonyRobotAI2_TEST <- read.delim("SonyRobotAI2_test.csv", header=FALSE, sep=',')
SonyRobotAI2_TEST <-data.matrix(SonyRobotAI2_TEST, rownames.force = NA)

lab_SonyRobotAI2_TEST <- SonyRobotAI2_TEST[,1]

StarLightCurves_TRAIN <- read.delim("StarLightCurves_train.csv", header=FALSE, sep = ',')
StarLightCurves_TRAIN <-data.matrix(StarLightCurves_TRAIN, rownames.force = NA)
StarLightCurves_TEST1 <- read.delim("StarLightCurves_test1.csv", header=FALSE, sep=',')
StarLightCurves_TEST1 <-data.matrix(StarLightCurves_TEST1, rownames.force = NA)
StarLightCurves_TEST2 <- read.delim("StarLightCurves_test2.csv", header=FALSE, sep=',')
StarLightCurves_TEST2 <-data.matrix(StarLightCurves_TEST2, rownames.force = NA)

StarLightCurves_TEST <- rbind(StarLightCurves_TEST1, StarLightCurves_TEST2)
lab_StarLightCurves_TEST <- StarLightCurves_TEST[,1]

TwoLeadECG_TRAIN <- read.delim("TwoLeadECG_train.csv", header=FALSE, sep = ',')
TwoLeadECG_TRAIN <-data.matrix(TwoLeadECG_TRAIN, rownames.force = NA)
TwoLeadECG_TEST <- read.delim("TwoLeadECG_test.csv", header=FALSE, sep=',')
TwoLeadECG_TEST <-data.matrix(TwoLeadECG_TEST, rownames.force = NA)

lab_TwoLeadECG_TEST <- TwoLeadECG_TEST[,1]

Yoga_TRAIN <- read.delim("Yoga_train.csv", header=FALSE, sep = ',')
Yoga_TRAIN <-data.matrix(Yoga_TRAIN, rownames.force = NA)
Yoga_TEST <- read.delim("Yoga_test.csv", header=FALSE, sep=',')
Yoga_TEST <-data.matrix(Yoga_TEST, rownames.force = NA)

lab_Yoga_TEST <- Yoga_TEST[,1]

EOGHorizontal_TRAIN <- read.delim("EOGHorizontal_train.csv", header=FALSE, sep = ',')
EOGHorizontal_TRAIN <-data.matrix(EOGHorizontal_TRAIN, rownames.force = NA)
EOGHorizontal_TEST <- read.delim("EOGHorizontal_test.csv", header=FALSE, sep=',')
EOGHorizontal_TEST <-data.matrix(EOGHorizontal_TEST, rownames.force = NA)

lab_EOGHorizontal_TEST <- EOGHorizontal_TEST[,1]


CinECGTorso_TRAIN <- read.delim("CinECGTorso_train.csv", header=FALSE, sep = ',')
CinECGTorso_TRAIN <-data.matrix(CinECGTorso_TRAIN, rownames.force = NA)
CinECGTorso_TEST <- read.delim("CinECGTorso_test.csv", header=FALSE, sep=',')
CinECGTorso_TEST <-data.matrix(CinECGTorso_TEST, rownames.force = NA)

lab_CinECGTorso_TEST <- CinECGTorso_TEST[,1]

ECG5000_TRAIN <- read.delim("ECG5000_train.csv", header=FALSE, sep = ',')
ECG5000_TRAIN <-data.matrix(ECG5000_TRAIN, rownames.force = NA)
ECG5000_TEST <- read.delim("ECG5000_test.csv", header=FALSE, sep=',')
ECG5000_TEST <-data.matrix(ECG5000_TEST, rownames.force = NA)

lab_ECG5000_TEST <- ECG5000_TEST[,1]







X1 = list(China_TRAIN[,2:ncol(China_TRAIN)],China_TEST[,2:ncol(China_TEST)], lab_China_TEST)
X2 = list(Coffee_TRAIN[,2:ncol(Coffee_TRAIN)],Coffee_TEST[,2:ncol(Coffee_TEST)], lab_Coffee_TEST)
X3 = list(ECGFiveDays_TRAIN[,2:ncol(ECGFiveDays_TRAIN)],ECGFiveDays_TEST[,2:ncol(ECGFiveDays_TEST)], lab_ECGFiveDays_TEST)
X4 = list(ECG200_TRAIN[,2:ncol(ECG200_TRAIN)],ECG200_TEST[,2:ncol(ECG200_TEST)], lab_ECG200_TEST)
X5 = list(Handoutlines_TRAIN[,2:ncol(Handoutlines_TRAIN)],Handoutlines_TEST[,2:ncol(Handoutlines_TEST)], lab_Handoutlines_TEST)
X6 = list(SonyRobotAI1_TRAIN[,2:ncol(SonyRobotAI1_TRAIN)],SonyRobotAI1_TEST[,2:ncol(SonyRobotAI1_TEST)], lab_SonyRobotAI1_TEST)
X7 = list(SonyRobotAI2_TRAIN[,2:ncol(SonyRobotAI2_TRAIN)],SonyRobotAI2_TEST[,2:ncol(SonyRobotAI2_TEST)], lab_SonyRobotAI2_TEST)
X8 = list(StarLightCurves_TRAIN[,2:ncol(StarLightCurves_TRAIN)],StarLightCurves_TEST[,2:ncol(StarLightCurves_TEST)], lab_StarLightCurves_TEST)
X9 = list(TwoLeadECG_TRAIN[,2:ncol(TwoLeadECG_TRAIN)],TwoLeadECG_TEST[,2:ncol(TwoLeadECG_TEST)], lab_TwoLeadECG_TEST)
X10 = list(Yoga_TRAIN[,2:ncol(Yoga_TRAIN)],Yoga_TEST[,2:ncol(Yoga_TEST)], lab_Yoga_TEST)
X11 = list(EOGHorizontal_TRAIN[,2:ncol(EOGHorizontal_TRAIN)],EOGHorizontal_TEST[,2:ncol(EOGHorizontal_TEST)], lab_EOGHorizontal_TEST)
X12 = list(CinECGTorso_TRAIN[,2:ncol(CinECGTorso_TRAIN)],CinECGTorso_TEST[,2:ncol(CinECGTorso_TEST)], lab_CinECGTorso_TEST)
X13 = list(ECG5000_TRAIN[,2:ncol( ECG5000_TRAIN)], ECG5000_TEST[,2:ncol( ECG5000_TEST)], lab_ECG5000_TEST)



#######################################
# The function to compute 12 depth of the set X
########################################
Boucle<-function(X){
set.seed(42)
grid = seq( 0, T, length.out = ncol(X[[1]]) )
dataframe = list(time = grid, vals = X[[1]])
dataframe2 = list(time = grid, vals = X[[2]])
X_f_train = list()
X_f_test = list()

for (i in 1:nrow(dataframe$vals)){
  X_f_train[[i]] <- list(args = dataframe$time, vals = dataframe$vals[i,])
}
for (i in 1:nrow(dataframe2$vals)){
  X_f_test[[i]] <- list(args = dataframe2$time, vals = dataframe2$vals[i,])
}
# Integred with halspace depth and simplicial depth  :
score_fd1 <- depthf.fd1(X_f_test,X_f_train)
Score_Simpl_FD <- score_fd1$Simpl_FD
Score_Half_FD <- score_fd1$Half_FD 
#Score_Simpl_ID <- score_fd1$Simpl_ID 
#Score_Half_ID <- score_fd1$Half_ID

# Modal depth :
score_HM <- depthf.hM(X_f_test,X_f_train)

# Random Projection method with halfspace, simplicial and random halfspace depth :
score_RP1<- depthf.RP1(X_f_test,X_f_train)

Score_RP_HD <- score_RP1$Half_FD
Score_RP_SD <- score_RP1$Simpl_FD
Score_RP_RHD <- score_RP1$RHalf_FD

arr = array(0, dim=c(ncol(X[[1]]),nrow(X[[1]]),1))
arr[,,1] = t(X[[1]] )
arr2 = array(0, dim=c(ncol(X[[2]]),nrow(X[[2]]),1))
arr2[,,1] = t(X[[2]] )
a = fOutl(arr,arr2,type="fAO")
b = fOutl(arr,arr2,type="fDO")
c = fOutl(arr,arr2,type="fbd")
d = fOutl(arr,arr2,type="fSDO")


a1 = roc( X[[3]], Score_Simpl_FD )
a2 = roc( X[[3]], Score_Half_FD )
a3 = roc( X[[3]], score_HM )
a4 = roc( X[[3]], Score_RP_HD  )
a5 = roc( X[[3]], Score_RP_SD  )
a6 = roc( X[[3]], Score_RP_RHD  )
a7 = roc( X[[3]], as.vector(a$fOutlyingnessZ ))
a8 = roc( X[[3]], as.vector(b$fOutlyingnessZ ))
a9 = roc( X[[3]], as.vector(c$fOutlyingnessZ ))
a10 = roc( X[[3]], as.vector(d$fOutlyingnessZ ))
return (list(Simplf_FD = a1$auc, Half_FD = a2$auc,
             modal = a3$auc, Random_projection_HD = a4$auc , Random_projection_SD = a5$auc,
             Random_projection_RHD = a6$auc, fAO = a7$auc, fDO = a8$auc,fbd = a9$auc ,fSDO = a10$auc))
#return (list( HalfRandom_projection_HD = a4$auc , fSDO = a10$auc))
}
#######################################

#######################################
# Results:
#######################################
result1 <- Boucle(X1) # Chinatown
result2 <- Boucle(X2) # Coffee
result3 <- Boucle(X3) # ECGFiveDays
result4 <- Boucle(X4) # ECG200
result5 <- Boucle(X5) # Handoutlines
result6 <- Boucle(X6) # SonyRobotAI1
result7 <- Boucle(X7) # SonyRobotAI1
result8 <- Boucle(X8) # StarLightCurves
result9 <- Boucle(X9) # TwoleadECG
result10 <- Boucle(X10) # Yoga
result11 <- Boucle(X11) # EOGHorizontalSignal
result12 <- Boucle(X12) # CinECGTorso
result13 <- Boucle(X13) # ECG5000

# These two vector represent the two data depths in the section 4.2 (Numerical results) of the paper. Each vector represent 
# one column of the tabular.
Random_projection_Halfspace_Depth = c(result1$Random_projection_HD, result2$Random_projection_HD, 
                                      result3$Random_projection_HD, result4$Random_projection_HD,
                                      result5$Random_projection_HD, result6$Random_projection_HD,
                                      result7$Random_projection_HD, result8$Random_projection_HD,
                                      result9$Random_projection_HD, result10$Random_projection_HD, 
                                      result11$Random_projection_HD, result12$Random_projection_HD, 
                                      result13$Random_projection_HD)
Functional_Stahel_Donoho_utlyingness = c(result1$fSDO, result2$fSDO, result3$fSDO, result4$fSDO,
                                         result5$fSDO, result6$fSDO, result7$fSDO, result8$fSDO,
                                         result9$fSDO, result10$fSDO, result11$fSDO, result12$fSDO,
                                          result13$fSDO)
#######################################
