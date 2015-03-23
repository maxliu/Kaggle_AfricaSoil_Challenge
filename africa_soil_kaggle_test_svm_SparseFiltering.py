
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.
import sys
import os
import pandas as pd
import numpy as np
import pickle

curDir,scriptname = os.path.split(sys.argv[0])
os.chdir(curDir)

#==============================================================================
# read data from files
#==============================================================================

outFile1 = open('./pickle_files/est1.pkl','U')
outFile2 = open('./pickle_files/est2.pkl','U')
feaPipeline = pickle.load(outFile1)
sv = pickle.load(outFile2)
outFile1.close()
outFile2.close()


test = pd.read_csv('./data/sorted_test.csv')

sample = pd.DataFrame();
sample['PIDN'] = test['PIDN']

test.drop('PIDN', axis=1, inplace=True)

test['Depth']=test['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtest = test.values

xtest = np.concatenate((xtest,feaPipeline.transform(xtest)),axis=1)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec=sv[i]
    preds[:,i] = sup_vec.predict(xtest).astype(float)

#dataFrapd.read_csv('./submission/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('./submission/beating_benchmark.csv', index = False)

print 'csv file saved'

