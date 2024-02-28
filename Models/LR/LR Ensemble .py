#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:03:34 2024

@author: davidteunissen

This set of Python code creates a functioning traditional ensemble for the logistic regression

The code is set up in the following format:
    1. We create a function that allows us to subsample the data according to the EasyEnsemble process
       and splits the data in test and train sets
    3. We create the actual ensemble function

"""

#%% 0. Packages are imported and pre-processed data is loaded
import numpy as np
import math
import time
import pandas as pd
from sklearn.model_selection import train_test_split

#LR packages
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

from sklearn.linear_model import LogisticRegression

#Load data
df_processed=pd.read_csv('/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/processed_nosub_norm.tsv',sep='\t')

#%% 1. We create a function that splits the data in test and train sets and splits the data on relevant variables
def testtrain(df_full, randomstate, n_df, y_col, x_col):
    
    #We split the data in a test and train set using ids, stratifying on conversion
    df_ids = df_full[['id', 'conversion']].drop_duplicates()
    
    train, test = train_test_split(
        df_ids, test_size=0.2, random_state = randomstate, shuffle=True, 
        stratify = df_ids[['conversion']])
    
    #Next, we split the train data sets into distinct subsampled sets. 
    trainsets_ids = []
    
    conv_train_ids = train[train['conversion']==1]['id'].reset_index(drop=True)
    nonconv_train_ids = train[train['conversion']==0]['id'].reset_index(drop=True)
    for i in range(n_df):
        
        #We extract the sequences that we want in this trainset
        conv_sampled = conv_train_ids.sample(n=len(conv_train_ids), replace=True, random_state=i).reset_index(drop=True)
        nonconv_sampled = nonconv_train_ids.sample(n=len(conv_train_ids), replace=False, random_state=i).reset_index(drop=True)
        
        #We concat these sets and append these to the ids list
        trainsets_ids.append(pd.concat([conv_sampled,nonconv_sampled], ignore_index=True))
    
    #Next, we extract the test and training sets split into predictors and outcome variables
    X_test = df_full[df_full['id'].isin(test['id'])][x_col]
    y_test = df_full[df_full['id'].isin(test['id'])][y_col]  
    
    #We create a dictionary in which we store all the subsampled train sets
    train_datasets = {}
    
    #We create a for-loop that creates n_df amount of subsampled balanced train sets
    for i in range(n_df):
        #First we run the EasyEnsemble function to get a subsampled dataset
        df_i = df_full[df_full['id'].isin(trainsets_ids[i])].sort_values(by=['id','timestamp'], ascending=True).reset_index(drop=True)
        
        #Perform a check if data sets are balanced
        n_conv_ee = df_i[df_i['conversion']==1]['id'].nunique()
        n_nonconv_ee = df_i[df_i['conversion']==0]['id'].nunique()
        
        if(n_conv_ee == n_nonconv_ee):
            print("EasyEnsemble dataset is balanced.")
        else: print("EasyEnsemble dataset is unbalanced. Check: Ratio of converted to non-converted is:", n_conv_ee/n_nonconv_ee)
        
        X_train = df_i[x_col]
        y_train = df_i[y_col]
        
        #Creating names and storing these in the dictionary
        X_train_name = f'X_train{i+1}'
        y_train_name = f'y_train{i+1}'
        
        train_datasets[X_train_name] = X_train 
        train_datasets[y_train_name] = y_train 
    
    return X_test, y_test, train_datasets


#%% 2. We create the actual ensemble function
def logistic_ensemble(data, rs_tt, n_models, ycol, xcol):
    
    start_time_data = time.time()
    print("Creating and splitting data sets")
    Xtest, ytest, train_df = testtrain(df_full= data, randomstate= rs_tt, n_df= n_models, 
                                       y_col = ycol, x_col= xcol)
    
    start_time_model = time.time()
        
    #We create a list in which we store our trained models
    ensemble = []
    
    #Extracting the keys for all training sets
    keys = list(train_df.keys())
    
    for i in range(n_models):
        X_train = train_df[keys[i*2]]
        y_train = train_df[keys[i*2+1]]
        
        model = LogisticRegression(solver='sag', C=100, max_iter=1000)
        model.fit(X_train,y_train)
        
        ensemble.append(model)
        print("Fitted model", i+1," - Models left to fit:", n_models - (i+1))
        
    
    #Now that all models are trained, we get all predictions using the test set and store them in a list
    ensemble_results = []
    for i in range(len(ensemble)):
        model = ensemble[i]
        y_pred_test = model.predict_proba(Xtest)[:,1]
        
        ensemble_results.append(y_pred_test)
        print("Retrieved predictions from model", i+1)
    
    #Using all predictions, we use probability averaging for all predictions to get an ensemble prediction
    ens_prob = sum(ensemble_results)/len(ensemble_results)
    ens_test = pd.Series(np.rint(ens_prob).astype(int))
    end_time = time.time()
    
    elapsed_time_data = end_time - start_time_data
    elapsed_time_model = end_time - start_time_model
    print("Total function runtime is:", elapsed_time_data)
    print("Ensemble function runtime is:", elapsed_time_model)
    #Retrieving prediction performance measures
    def result_eval(ypred, ytrue, yprob):
        
        #Retrieve prediction performances
        results = {'log-loss': [log_loss(ytrue, yprob)],
                   'brier score': [brier_score_loss(ytrue, yprob)],
                   'recall': [recall_score(ytrue, ypred, average='weighted')],
                   'precision': [precision_score(ytrue, ypred, average='weighted')],
                   'accuracy': [accuracy_score(ytrue, ypred)],
                   'F-measure': [f1_score(ytrue, ypred, average='weighted')],
                   'AUC': [roc_auc_score(ytrue, ypred, average='weighted')]      
            }
        
        results = pd.DataFrame(results)
        
        return results
    
    results = result_eval(ypred=ens_test, ytrue = ytest, yprob = ens_prob)
    results['elapsed_time_data'] = elapsed_time_data
    results['elapsed_time_model'] = elapsed_time_model
    results['n_models'] = n_models

    print(results)
    
    return results

#%% Aggregating results
lr_ensemble_res = []
for ens_models in np.arange(5,35,5):
    res = logistic_ensemble(data=df_processed, rs_tt=33, n_models=ens_models, ycol='conversion', 
                            xcol= ['cost', 'campaign', 'timestamp','time_since_last_click','cat1', 
                                   'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'])
    lr_ensemble_res.append(res)

lr_ensemble_df = pd.concat(lr_ensemble_res, ignore_index=True)

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/LR_ens_res_5_30_bagged.tsv"
lr_ensemble_df.to_csv(file_path, sep='\t', index=False)


