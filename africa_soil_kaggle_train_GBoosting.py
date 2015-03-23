
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
from sklearn import preprocessing
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV,
                                 ParameterGrid, ParameterSampler)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report


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

X_train,X_valid,Y_train,Y_valid = \
train_test_split(xtrain, labels, test_size=0.1, random_state=42)

#==============================================================================
# feature engineering
#==============================================================================
min_max_scaler = preprocessing.MinMaxScaler()

poly = PolynomialFeatures(2,interaction_only=True,include_bias =False)
#pca = PCA(n_components =5)
pcapoly = PCA(n_components =100)

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

feaPipeline.fit(X_train,Y_train)

X_train = np.concatenate((X_train,feaPipeline.transform(X_train)),axis=1)
X_valid = np.concatenate((X_valid,feaPipeline.transform(X_valid)),axis=1)

#==============================================================================
# tranning and prediction
#==============================================================================
C0 = 10000
verbose0 = 2

sv =[ svm.SVR(C=C0, verbose = verbose0),\
          svm.SVR(C=C0, verbose = verbose0),\
          svm.SVR(C=C0, verbose = verbose0),\
          svm.SVR(C=C0, verbose = verbose0),\
          svm.SVR(C=C0, verbose = verbose0)]

preds = np.zeros((X_valid.shape[0], 5))

s = np.zeros(5)

for i in range(1):
    sv[i].fit(X_train, Y_train[:,i])
    sup_vec= sv[i]
    preds[:,i] = sup_vec.predict(X_valid).astype(float)
    s[i] = mean_squared_error(Y_valid[:,i],preds[:,i])
    print i,s[i]

print "\n\n\n mean err ",s.mean()
#==============================================================================
# GBoost
#==============================================================================

# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls','verbose':1}
clf = ensemble.GradientBoostingRegressor(**params)
i=0
clf.fit(X_train, Y_train[:,i])
mse = mean_squared_error(Y_valid[:,i], clf.predict(X_valid))
print("MSE: %.4f" % mse)

#==============================================================================
# grid searching 
#==============================================================================
#param_grid ={"kernel": ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')}
#
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
#          'learning_rate': 0.01, 'loss': 'ls'}
#
#grid_search = GridSearchCV(sup_vec, param_grid=param_grid)
#start = time()
#i=0
#grid_search.fit(X_train, Y_train[:,i])
#
#print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#      % (time() - start, len(grid_search.grid_scores_)))
#report(grid_search.grid_scores_)
#
#print "done"

#==============================================================================
# save results
#==============================================================================
outFile1 = open('./pickle_files/est1.pkl','wb')
outFile2 = open('./pickle_files/est2.pkl','wb')
pickle.dump(feaPipeline,outFile1)
pickle.dump(sv,outFile2)
outFile1.close
outFile2.close

outFile1 = open('./pickle_files/est1.pkl','U')
outFile2 = open('./pickle_files/est2.pkl','U')
feaPipeline = pickle.load(outFile1)
sv = pickle.load(outFile2)
outFile1.close()
outFile2.close()



