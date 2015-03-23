
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
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV,
                                 ParameterGrid, ParameterSampler)
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion


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
def myScore(estimater,X,Y):
    y_pred = estimater.predict(X)
    return 1.0-mean_absolute_error(Y,y_pred)
#==============================================================================
#set dir
#==============================================================================
curDir,scriptname = os.path.split(sys.argv[0])
os.chdir(curDir)

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

X_train0,X_valid0,Y_train,Y_valid = \
train_test_split(xtrain, labels, test_size=0.01, random_state=42)

#==============================================================================
# feature engineering
#==============================================================================
min_max_scaler = preprocessing.MinMaxScaler()

#poly = PolynomialFeatures(2,interaction_only=True,include_bias =False)
#pca = PCA(n_components =5
pcapoly = PCA(n_components =3000)

#selection = SelectKBest(k =10)
feaPipeline = Pipeline([
            ("MinMaxScaler",min_max_scaler),\
            ("pcapoly",pcapoly)
            ])
#feaPipeline = Pipeline([
#            ("MinMaxScaler",min_max_scaler),\
#            ("pcapoly",pcapoly),\
#            ("poly",poly)
#            ])

#feaPipeline.fit(X_train0,Y_train)
#min_max_scaler.fit(X_train0,Y_train)
#
#X_train = np.concatenate((min_max_scaler.transform(X_train0),feaPipeline.transform(X_train)),axis=1)
#X_valid = np.concatenate((min_max_scaler.transform(X_valid0),feaPipeline.transform(X_valid)),axis=1)

#X_train =feaPipeline.transform(X_train)
#X_valid =feaPipeline.transform(X_valid)

#==============================================================================
# tranning and prediction
#==============================================================================
C0 = 10000
verbose0 = 2

CList = [900,11000,5000,1000,3000]
gammaList = [0.001,0.0,0.0,0.0,0.0]
kernelList = ["rbf","rbf","rbf","poly","poly"]

sv =[     svm.SVR(C=CList[0], gamma=gammaList[0],kernel = kernelList[0],verbose = verbose0),\
          svm.SVR(C=CList[1], gamma=gammaList[1],kernel = kernelList[1],verbose = verbose0),\
          svm.SVR(C=CList[2], gamma=gammaList[2],kernel = kernelList[2],verbose = verbose0),\
          svm.SVR(C=CList[3], gamma=gammaList[3],kernel = kernelList[3],verbose = verbose0),\
          svm.SVR(C=CList[4], gamma=gammaList[4],kernel = kernelList[4],verbose = verbose0)
    ]

f =[feaPipeline,feaPipeline,feaPipeline,feaPipeline,feaPipeline]
m=[min_max_scaler,min_max_scaler,min_max_scaler,min_max_scaler,min_max_scaler]
preds = np.zeros((X_valid0.shape[0], 5))

s = np.zeros(5)

#for i in range(5):
#    sv[i].fit(X_train, Y_train[:,i])
#    sup_vec= sv[i]
#    preds[:,i] = sup_vec.predict(X_valid).astype(float)
#    s[i] = mean_absolute_error(Y_valid[:,i],preds[:,i])
#    print i,s[i]

print "\n\n\n mean err ",s.mean()
#==============================================================================
# grid searching
#==============================================================================
# "poly","sigmoid"

param_grid ={"kernel":["rbf"],
"gamma":[0.0,0.01],
"C":[500,1000,5000,10000]}
#"C":[1000,2000,5000,10000]}
s = np.zeros(5)
print "start grid searching ......"
for i in range(5):
    feaPipeline.fit(X_train0,Y_train[:,i])
    min_max_scaler.fit(X_train0,Y_train[:,i])
    f[i] = feaPipeline
    m[i] = min_max_scaler
    X_train = np.concatenate((min_max_scaler.transform(X_train0),feaPipeline.transform(X_train0)),axis=1)
    X_valid = np.concatenate((min_max_scaler.transform(X_valid0),feaPipeline.transform(X_valid0)),axis=1)
    
    sup_vec = svm.SVR(verbose = verbose0)
    
    grid_search = GridSearchCV(sup_vec, param_grid=param_grid,cv=3,scoring =myScore)
    start = time()
    grid_search.fit(X_train, Y_train[:,i])
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
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
print "saving resutls ...."
outFile1 = open('./pickle_files/est1.pkl','wb')
outFile2 = open('./pickle_files/est2.pkl','wb')
outFile3 = open('./pickle_files/est3.pkl','wb')

pickle.dump(f,outFile1)
pickle.dump(sv,outFile2)
pickle.dump(m,outFile3)
outFile1.close
outFile2.close
outFile3.close

outFile1 = open('./pickle_files/est1.pkl','U')
outFile2 = open('./pickle_files/est2.pkl','U')
outFile3 = open('./pickle_files/est3.pkl','U')
f = pickle.load(outFile1)
sv = pickle.load(outFile2)
m = pickle.load(outFile3)

outFile1.close()
outFile2.close()
outFile3.close()



