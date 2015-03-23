
#==============================================================================
# useful link
#http://scikit-learn.org/stable/modules/model_evaluation.html
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
#==============================================================================
import sys
import os
from  time import time
import pandas as pd
import numpy as np
import pickle
from operator import itemgetter
from sklearn import svm, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV,
                                 ParameterGrid, ParameterSampler)
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion

from sparseFiltering import *

#==============================================================================
#set dir
#==============================================================================

curDir,scriptname = os.path.split(sys.argv[0])
os.chdir(curDir)
#==============================================================================
# # Utility function to report best scores
#==============================================================================
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    return top_scores[0]

def myScore(estimater,X,Y):
    y_pred = estimater.predict(X)
    return 1.0-mean_absolute_error(Y,y_pred)

#==============================================================================
# Initial 
#==============================================================================

train_file = './data/training.csv'

#==============================================================================
# read data from files and pre-treatment
#==============================================================================

train = pd.read_csv(train_file)

labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

train['Depth']=train['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtrain = train.values

#==============================================================================
# seprete data to train and validation
#==============================================================================

X_train,X_valid,Y_train,Y_valid = \
train_test_split(xtrain, labels, test_size=0.1, random_state=42)

#==============================================================================
# feature engineering
#==============================================================================
#min_max_scaler = preprocessing.MinMaxScaler()
#
#poly = PolynomialFeatures(2,interaction_only=True,include_bias =False)
##pca = PCA(n_components =5)
#pcapoly = PCA(n_components =200)
#
##selection = SelectKBest(k =10)
#feaPipeline = Pipeline([
#            ("MinMaxScaler",min_max_scaler),\
#            ("pcapoly",pcapoly)
#            ])
##feaPipeline = Pipeline([
##            ("MinMaxScaler",min_max_scaler),\
##            ("pcapoly",pcapoly),\
##            ("poly",poly)
##            ])
#
#feaPipeline.fit(X_train,Y_train)
#
#X_train = np.concatenate((X_train,feaPipeline.transform(X_train)),axis=1)
#X_valid = np.concatenate((X_valid,feaPipeline.transform(X_valid)),axis=1)

print "start sparse filter training"

L1_size = 500 # Increase this for more features
L1 = sparseFiltering(L1_size,  np.transpose(X_train))
print "Layer 1 done."
X_train1 = feedForwardSF(L1,np.transpose(X_train))
X_valid1 = feedForwardSF(L1,np.transpose(X_valid))

X_train2 = np.transpose(X_train1)
X_valid2 = np.transpose(X_valid1)

###################
###################

L2_size = 250 # Increase this for more features
L2 = sparseFiltering(L2_size,  np.transpose(X_train2))
print "Layer 2 done."
X_train3 = feedForwardSF(L2,np.transpose(X_train2))
X_valid3 = feedForwardSF(L2,np.transpose(X_valid2))

X_train4 = np.transpose(X_train3)
X_valid4 = np.transpose(X_valid3)


#X_train = np.concatenate((X_train,X_train2),axis=1)
#X_valid = np.concatenate((X_valid,X_valid2),axis=1)

min_max_scaler = preprocessing.MinMaxScaler()

min_max_scaler.fit(X_train4)

X_train = min_max_scaler.transform(X_train4)
X_valid = min_max_scaler.transform(X_valid4)


#==============================================================================
# tranning and prediction
#==============================================================================
C0 = 10000
verbose0 = 0

CList = [C0,C0,C0,C0,C0]
gammaList = [0.001,0.0,0.0,0.0,0.0]
kernelList = ["rbf","rbf","rbf","rbf","rbf"]

sv =[     svm.SVR(C=CList[0], gamma=gammaList[0],kernel = kernelList[0],verbose = verbose0),\
          svm.SVR(C=CList[1], gamma=gammaList[1],kernel = kernelList[1],verbose = verbose0),\
          svm.SVR(C=CList[2], gamma=gammaList[2],kernel = kernelList[2],verbose = verbose0),\
          svm.SVR(C=CList[3], gamma=gammaList[3],kernel = kernelList[3],verbose = verbose0),\
          svm.SVR(C=CList[4], gamma=gammaList[4],kernel = kernelList[4],verbose = verbose0)
    ]

preds = np.zeros((X_valid.shape[0], 5))

s = np.zeros(5)

#for i in range(5):
#    sv[i].fit(X_train, Y_train[:,i])
#    sup_vec= sv[i]
#    preds[:,i] = sup_vec.predict(X_valid).astype(float)
#    s[i] = mean_absolute_error(Y_valid[:,i],preds[:,i])
#    print i,s[i]

#print "\n\n\n mean err ",s.mean()


#==============================================================================
# grid searching
#==============================================================================
#,"poly","sigmoid"
param_grid ={"kernel":["rbf"],
             "gamma":[0.0,0.001,0.01],
             "C":[1000,2000,5000,10000]}
             
s = np.zeros(5)
print "start grid searching ......"
for i in range(5):

    sup_vec = svm.SVR(verbose = verbose0)
    
    grid_search = GridSearchCV(sup_vec, param_grid=param_grid,cv=5,scoring =myScore)
#    start = time()
    grid_search.fit(X_train, Y_train[:,i])
    
#    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#          % (time() - start, len(grid_search.grid_scores_)))
    print "\n\n i = ",i
    print "\n\n"
    report(grid_search.grid_scores_)
    
    sv[i] = grid_search.best_estimator_
    
    y_true,y_pred = Y_valid[:,i],sv[i].predict(X_valid).astype(float)
    
    print "mean err: ",i,mean_absolute_error(y_true,y_pred)
    
    s[i] = mean_absolute_error(y_true,y_pred)
print "done s = ",s.mean()

#==============================================================================
# save results
#==============================================================================
#outFile1 = open('./pickle_files/est1.pkl','wb')
#outFile2 = open('./pickle_files/est2.pkl','wb')
#pickle.dump(feaPipeline,outFile1)
#pickle.dump(sv,outFile2)
#outFile1.close
#outFile2.close
#
#outFile1 = open('./pickle_files/est1.pkl','U')
#outFile2 = open('./pickle_files/est2.pkl','U')
#feaPipeline = pickle.load(outFile1)
#sv = pickle.load(outFile2)
#outFile1.close()
#outFile2.close()



