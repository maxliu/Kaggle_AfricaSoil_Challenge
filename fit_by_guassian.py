
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
import pylab as pl
from scipy import *
from scipy.optimize import leastsq
from scipy.optimize import fmin_slsqp

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

mcols = train.columns[0:3578]

fcols = [float(mcols[i][1:]) for i in range(len(mcols))]

fig1 = pl.figure()
pl.plot(fcols,xtrain[0,0:3578])
pl.plot(fcols,xtrain[1,0:3578])
pl.plot(fcols,xtrain[200,0:3578])
pl.show()

#d0 = [zip(fcols,xtrain[i,0:3578]) for i in range(xtrain.shape[0])]
#base = d0[0][0][1]
#
#pl.plot(d0[0][:,0],d0[0][:,1])
#
#pl.show()

#==============================================================================
# gussian fit
#   http://stackoverflow.com/questions/10880266/robust-algorithm-for-detection-of-peak-widths
#==============================================================================
def gaussian(x, A, x0, sig):
    return A*exp(-(x-x0)**2/(2.0*sig**2))

def fit(p,x):
    return np.sum([gaussian(x, p[i*3],p[i*3+1],p[i*3+2]) 
                   for i in xrange(len(p)/3)],axis=0)

err = lambda p, x, y: fit(p,x)-y

y_vals = xtrain[100,0:3578]

#params are our intitial guesses for fitting gaussians, 
#(Amplitude, x value, sigma):
params = [[2,660,5],
          [1,1225,5],
          [1.8,1629,5],
          [1.0,1870,5],
          [1.7,3403,5],
          [2,3629,5],
          [0.8,3709,5],
          [0.45,4516,5],
          [0.4,5209,5],
          [0.36,7203,5] 
          ]
          # this last one is our noise estimate
params = np.asarray(params).flatten()

x  = fcols # xrange(len(y_vals))

#==============================================================================
# start fitting....
#==============================================================================
#results, value = leastsq(err, params, args=(x, y_vals),maxfev=20000)
results, value = fmin_slsqp(err, params, args=(x, y_vals),iter=20000)

for res in results.reshape(-1,3):
    print "amplitude, position, sigma", res

fig2 = pl.figure()
pl.subplot(211, title='original data')
pl.plot(x,y_vals)
pl.subplot(212, title='guassians fit')
y = fit(results, x)
pl.plot(x, y)
pl.savefig('fig2.png')
pl.show()



