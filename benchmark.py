
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.
import sys
import os
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation

curDir,scriptname = os.path.split(sys.argv[0])
os.chdir(curDir)

train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/sorted_test.csv')
sample = pd.DataFrame();
sample['PIDN'] = test['PIDN']

labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

#xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
train['Depth']=train['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)
test['Depth']=test['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtrain = train.values
xtest = test.values

sup_vec = svm.SVR(C=100.0, verbose = 2)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

#dataFrapd.read_csv('./submission/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('./submission/beating_benchmark.csv', index = False)


