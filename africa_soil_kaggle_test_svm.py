
# coding: utf-8
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
outFile3 = open('./pickle_files/est3.pkl','U')
f = pickle.load(outFile1)
sv = pickle.load(outFile2)
m = pickle.load(outFile3)

outFile1.close()
outFile2.close()
outFile3.close()


test = pd.read_csv('./data/sorted_test.csv')

sample = pd.DataFrame();
sample['PIDN'] = test['PIDN']

test.drop('PIDN', axis=1, inplace=True)

test['Depth']=test['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtest0 = test.values



preds = np.zeros((xtest0.shape[0], 5))
for i in range(5):
    feaPipeline = f[i]
    min_max_scaler = m[i]
    xtest = np.concatenate((min_max_scaler.transform(xtest0),feaPipeline.transform(xtest0)),axis=1)
    sup_vec=sv[i]
    preds[:,i] = sup_vec.predict(xtest).astype(float)

#dataFrapd.read_csv('./submission/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('./submission/svm_more_search.csv', index = False)

print 'csv file saved'

