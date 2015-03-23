
#==============================================================================
# useful link
#http://scikit-learn.org/stable/modules/model_evaluation.html
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
#==============================================================================
import sys
import os
import scipy
from scipy.spatial.distance import euclidean as dis
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

from sklearn import linear_model

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

#train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

train.drop(['PIDN'], axis=1, inplace=True)

train['Depth']=train['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtrain = train.values

X_train0,X_valid0,Y_train,Y_valid = \
train_test_split(xtrain, labels, test_size=0.1, random_state=42)


lenOfFre = 3578  # +16 = 3594



def get_pred(x0,X_train0,n=20):

    xs= [np.insert(x[lenOfFre:lenOfFre+21],0,dis(x0-x0[10],x[0:lenOfFre]-x[10])) for x in X_train0]

    xs1 = np.asarray(xs)
    
    #dis2= (dis(xtrain[1,0:lenOfFre],x[0:lenOfFre]) for x in xtrain[0:5,:])
    
    ind = np.lexsort((xs1[:,1],xs1[:,0]))
    
    xs_sort = xs1[ind]
    
    xtrain_1 = xs_sort[0:n]
    
    #xtrain_1 = xs1
    
    #s_weight = [1/x for x in xs1[0,:]]
    
    xx= xtrain_1[:,1:17]
    #    clf = linear_model.LinearRegression()
    xp5 = np.zeros(5)
    for i in range(1,6):
        yy = xtrain_1[:,16+i]
        #clf = svm.SVR(C=10000)
        clf = linear_model.LinearRegression()
        clf.fit(xx,yy)
        xp0 = X_valid0[nx,lenOfFre:lenOfFre+16]
        xp = clf.predict(xp0)
        xp5[i-1] = xp
        #print xp ,X_valid0[nx,lenOfFre+15+i]
    return xp5
    
nx =40

x0 =  X_valid0[nx,0:lenOfFre]  
print "\n"
print X_valid0[nx,lenOfFre+15+1:lenOfFre+15+6]
print "\n"
print get_pred(x0,X_train0,n=10)
print get_pred(x0,X_train0,n=20)
print get_pred(x0,X_train0,n=30)
print get_pred(x0,X_train0,n=40)
print get_pred(x0,X_train0,n=50)
print get_pred(x0,X_train0,n=100)
print get_pred(x0,X_train0,n=150)
print get_pred(x0,X_train0,n=200)
print get_pred(x0,X_train0,n=500)
print get_pred(x0,X_train0,n=1000)
##xs_sort = np.sort(xs1,axis=1,order=0)

##==============================================================================
## seprete data to train and validation
##==============================================================================
#
#X_train0,X_valid0,Y_train,Y_valid = \
#train_test_split(xtrain, labels, test_size=0.1, random_state=42)
#
##==============================================================================
## feature engineering
##==============================================================================
#min_max_scaler = preprocessing.MinMaxScaler()
#
##poly = PolynomialFeatures(2,interaction_only=True,include_bias =False)
##pca = PCA(n_components =5
#pcapoly = PCA(n_components =3000)
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
##feaPipeline.fit(X_train0,Y_train)
##min_max_scaler.fit(X_train0,Y_train)
##
##X_train = np.concatenate((min_max_scaler.transform(X_train0),feaPipeline.transform(X_train)),axis=1)
##X_valid = np.concatenate((min_max_scaler.transform(X_valid0),feaPipeline.transform(X_valid)),axis=1)
#
##X_train =feaPipeline.transform(X_train)
##X_valid =feaPipeline.transform(X_valid)
#
##==============================================================================
## tranning and prediction
##==============================================================================
#C0 = 10000
#verbose0 = 2
#
#CList = [900,11000,5000,1000,3000]
#gammaList = [0.001,0.0,0.0,0.0,0.0]
#kernelList = ["rbf","rbf","rbf","poly","poly"]
#
#sv =[     svm.SVR(C=CList[0], gamma=gammaList[0],kernel = kernelList[0],verbose = verbose0),\
#          svm.SVR(C=CList[1], gamma=gammaList[1],kernel = kernelList[1],verbose = verbose0),\
#          svm.SVR(C=CList[2], gamma=gammaList[2],kernel = kernelList[2],verbose = verbose0),\
#          svm.SVR(C=CList[3], gamma=gammaList[3],kernel = kernelList[3],verbose = verbose0),\
#          svm.SVR(C=CList[4], gamma=gammaList[4],kernel = kernelList[4],verbose = verbose0)
#    ]
#
#preds = np.zeros((X_valid0.shape[0], 5))
#
#s = np.zeros(5)
#
##for i in range(5):
##    sv[i].fit(X_train, Y_train[:,i])
##    sup_vec= sv[i]
##    preds[:,i] = sup_vec.predict(X_valid).astype(float)
##    s[i] = mean_absolute_error(Y_valid[:,i],preds[:,i])
##    print i,s[i]
#
#print "\n\n\n mean err ",s.mean()
##==============================================================================
## grid searching
##==============================================================================
## "poly","sigmoid"
#
#param_grid ={"kernel":["rbf"],
#"gamma":[0.0],
#"C":[1000,10000]}
##"C":[1000,2000,5000,10000]}
#s = np.zeros(5)
#print "start grid searching ......"
#for i in range(5):
#    feaPipeline.fit(X_train0,Y_train)
#    min_max_scaler.fit(X_train0,Y_train)
#    
#    X_train = np.concatenate((min_max_scaler.transform(X_train0),feaPipeline.transform(X_train0)),axis=1)
#    X_valid = np.concatenate((min_max_scaler.transform(X_valid0),feaPipeline.transform(X_valid0)),axis=1)
#    
#    sup_vec = svm.SVR(verbose = verbose0)
#    
#    grid_search = GridSearchCV(sup_vec, param_grid=param_grid,cv=5,scoring =myScore)
#    start = time()
#    grid_search.fit(X_train, Y_train[:,i])
#    
#    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#          % (time() - start, len(grid_search.grid_scores_)))
#    print "\n\n i = ",i
#    print "\n\n"
#    report(grid_search.grid_scores_)
#    
#    sv[i] = grid_search.best_estimator_
#    
#    y_true,y_pred = Y_valid[:,i],sv[i].predict(X_valid).astype(float)
#    
#    print "mean err: ",i,mean_absolute_error(y_true,y_pred)
#    s[i] = mean_absolute_error(y_true,y_pred)
#print "done s = ",s.mean()
##==============================================================================
## save results
##==============================================================================
#outFile1 = open('./pickle_files/est1.pkl','wb')
#outFile2 = open('./pickle_files/est2.pkl','wb')
#outFile3 = open('./pickle_files/est3.pkl','wb')
#pickle.dump(feaPipeline,outFile1)
#pickle.dump(sv,outFile2)
#pickle.dump(min_max_scaler,outFile3)
#outFile1.close
#outFile2.close
#outFile3.close
#
#outFile1 = open('./pickle_files/est1.pkl','U')
#outFile2 = open('./pickle_files/est2.pkl','U')
#outFile3 = open('./pickle_files/est3.pkl','U')
#feaPipeline = pickle.load(outFile1)
#sv = pickle.load(outFile2)
#min_max_scaler = pickle.load(outFile3)
#
#outFile1.close()
#outFile2.close()
#outFile3.close()
#
#
#
