#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:52:56 2023

@author: davidteunissen
"""
"""
This set of Python code creates a functioning traditional ensemble for the logistic regression

The code is set up in the following format:
    1. We create a function that allows us to subsample the data according to the EasyEnsemble process
       and splits the data in test and train sets
    3. We create the actual ensemble function
    
ADDITIONALLY: This is an amended version which incorporates a methodology to convert observation-wise 
predictions to predictions for whole sequences for the logistic regression
"""

#%% 0. Packages are imported and pre-processed data is loaded
import numpy as np
import pandas as pd
import time
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

df_processed=pd.read_csv('/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/processed_nosub_norm.tsv',sep='\t')


#%% 1. We create a function that splits the data in test and train sets and splits the data on relevant variables
def testtrain(df_full, randomstate, n_df, y_col, x_col):
    
    #We split the dat ain a test and train set using ids, stratifying on conversion
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
        nonconv_sampled = nonconv_train_ids.sample(n=len(conv_train_ids), replace=True, random_state=i).reset_index(drop=True)
        
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
def logistic_ensemble(data, rs, n_models, ycol, xcol, agg_metric_ens, agg_metric_seq):
    
    start_time_data = time.time()
    print("Creating and splitting data sets")
    Xtest, ytest, train_df = testtrain(df_full= data, randomstate= rs, n_df= n_models, 
                                       y_col = ycol, x_col= xcol)
    start_time_model = time.time()
    
    #We create a list in which we store our trained models
    ensemble = []
    
    #Extracting the keys for training sets
    keys = list(train_df.keys())
    
    for i in range(n_models):
        X_train = train_df[keys[i*2]]
        y_train = train_df[keys[i*2+1]]
        
        model = LogisticRegression(solver='sag', C=100, max_iter=1000)
        model.fit(X_train,y_train['conversion'])
        
        ensemble.append(model)
        print("Fitted model", i+1," - Models left to fit:", n_models - (i+1))
        
    
    #Now that all models are trained, we get all predictions using the test set and store them in a list
    ensemble_results = []
    for i in range(len(ensemble)):
        model = ensemble[i]
        y_pred_test = model.predict_proba(Xtest)[:,1]
        
        ensemble_results.append(y_pred_test)
        print("Retrieved predictions from model", i+1)
    
    #Using all predictions, we can use probability aggregation for all model predictions to get an 
    #full ensemble probability prediction
    def pred_aggr(pred_res, metric):
        if metric == 'average':
            aggregate = sum(pred_res)/len(pred_res)
            
        if metric == 'median':
            aggregate = np.rint(np.median(np.array(pred_res),axis=0))
        
        if metric == 'min':
            aggregate = np.minimum.reduce(pred_res)
        
        if metric == 'max':
            aggregate = np.maximum.reduce(pred_res)
        
        return aggregate
    
    y_pred_eval = {
        'id': ytest['id'],
        'y_true': ytest['conversion'],
        'y_pred_proba': pred_aggr(ensemble_results, agg_metric_ens)
        }
    
    y_pred_eval = pd.DataFrame(y_pred_eval)
    
    #We aggregate all predictions within a sequence to get a final sequence aggregated prediction
    def seq_aggr(pred_eval, metric):
        if metric == 'average':
            prob = pred_eval.groupby(['id'])['y_pred_proba'].mean()
            aggregate = np.rint(prob)
        
        if metric == 'median':
            prob = np.median(np.array(pred_eval),axis=0)
            aggregate = np.rint(prob)
        
        if metric == 'min':
            prob = pred_eval.groupby(['id'])['y_pred_proba'].min()
            aggregate = np.rint(prob)
        
        if metric == 'max':
            pred_eval.groupby(['id'])['y_pred_proba'].max()
            aggregate = np.rint(prob)
        
        return prob, aggregate
    
    y_prob_final, y_pred_final = seq_aggr(y_pred_eval, agg_metric_seq)
    
    y_test_eval = ytest.drop_duplicates().sort_values(by='id')['conversion']
    
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
    
    
    results = result_eval(ypred=y_pred_final, ytrue=y_test_eval, 
                          yprob=y_prob_final)
    results['elapsed_time_data'] = elapsed_time_data
    results['elapsed_time_model'] = elapsed_time_model
    results['n_models'] = n_models
    results['agg_metric_ens'] = agg_metric_ens
    results['agg_metric_seq'] = agg_metric_seq
    
    
    print(results)
    
    return results

#%% Aggregating results
lr_ensemble_res_seq = []
for ens_models in np.arange(5,35,5):
    res = logistic_ensemble(data= df_processed, rs=33, n_models=ens_models, ycol = ['id','conversion'], 
                                xcol = ['cost', 'campaign', 'timestamp','time_since_last_click','cat1', 
                                         'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'], 
                                agg_metric_ens = 'average', agg_metric_seq= 'average')
    lr_ensemble_res_seq.append(res)

lr_ensemble_seq_df = pd.concat(lr_ensemble_res_seq, ignore_index=True)

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/LR_ens_seq_res_5_30_bagged.tsv"
lr_ensemble_seq_df.to_csv(file_path, sep='\t', index=False)

