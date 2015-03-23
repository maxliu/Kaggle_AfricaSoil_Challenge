
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.
#import sys
#import os
#import pandas as pd
#import numpy as np
#import pickle

#curDir,scriptname = os.path.split(sys.argv[0])
#os.chdir(curDir)

#==============================================================================
# read data from files
#==============================================================================

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


test = pd.read_csv('./data/sorted_test.csv')
test_slopes = test_file_slope = './Soil_R_code/pSlope_test.csv'

#==============================================================================
# test data
#==============================================================================
sample = pd.DataFrame();
sample['PIDN'] = test['PIDN']

test.drop('PIDN', axis=1, inplace=True)

test['Depth']=test['Depth'].map({'Topsoil':1,'Subsoil':0}).astype(int)

xtest = test.values

#==============================================================================
# slope data
#==============================================================================
test_slope =pd.read_csv(test_slopes)

xtest_slope = test_slope.values

#==============================================================================
# merge data
#==============================================================================
xtest =  np.concatenate((xtest,xtest_slope),axis=1)
#==============================================================================
# 
#==============================================================================
xtest = np.concatenate((min_max_scaler.transform(xtest),feaPipeline.transform(xtest)),axis=1)

#xtest = feaPipeline.transform(xtest)

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

