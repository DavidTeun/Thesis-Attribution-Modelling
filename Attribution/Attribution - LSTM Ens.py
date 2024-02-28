#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:59:35 2023

@author: davidteunissen


In this code sequence we build a traditional ensemble using LSTM models with descriptive attribution.
We take the following steps:
    1. We write a function that allows us to utilize the EasyEnsemble methodology for the LSTM setting
    2. We split our data into train sets and a test set using the EasyEnsemble methodology
    3. We create our actual ensemble
    4. We create a function that extracts probabilities which are used for descriptive attribution
"""

#%%Importing packages and raw data
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Performance metric packages
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

#LSTM packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

#Reading data
df_padded = pd.read_csv('/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/padded_data_norm.tsv',sep='\t')

#%% 2. We split our data into train sets and a test set using the EasyEnsemble methodology
def testtrain_LSTM(df, randomstate, n_df, y_col, x_col):
    
    #We split the data in a test and train set using ids, stratifying on conversion
    df_ids = df[['id', 'conversion']].drop_duplicates()
    
    train, test = train_test_split(
        df_ids, test_size=0.2, random_state = randomstate, shuffle=True,
        stratify=df_ids['conversion'])
    
    #Extracting all the test data
    df_test = df[df['id'].isin(test['id'])]
    
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
    
    #Next, we create a function that allows us to transform the data into a 3D format for LSTM
    def batches(X, seq_len, x_col_LSTM):
        
        #Reshape the data in the form of (ID, Sequence, Variables)
        batch_X = X[x_col_LSTM].to_numpy().reshape(X['id'].nunique(),seq_len, len(x_col_LSTM))

        return batch_X
    
    #We transform the test data to our desired format
    x_col_lstm = x_col.copy()
    x_col_lstm.remove('id')
    
    X_test = batches(df_test, seq_len = 20, x_col_LSTM = x_col_lstm)
    y_test = df_test[y_col].drop_duplicates().reset_index(drop=True)['conversion']
    
    #We create a dictionary in which we store all the subsampled train sets
    train_datasets = {}
    
    
    #We create a for-loop that extracts n_df amount of subsampled balanced train sets
    for i in range(n_df):
        #First we run the EasyEnsemble function to get a subsampled dataset
        df_i = df[df['id'].isin(trainsets_ids[i])].reset_index(drop=True)
        
        #Perform a check if data sets are balanced
        n_conv_ee = df_i[df_i['conversion']==1]['id'].nunique()
        n_nonconv_ee = df_i[df_i['conversion']==0]['id'].nunique()
        
        if(n_conv_ee == n_nonconv_ee):
            print("EasyEnsemble dataset is balanced.")
        else: print("EasyEnsemble dataset is unbalanced. Check: Ratio of converted to non-converted is:", n_conv_ee/n_nonconv_ee)
        
        
        X_train = batches(df_i, seq_len = 20, x_col_LSTM = x_col_lstm)
        y_train = df_i[y_col].drop_duplicates().reset_index(drop=True)['conversion']
        
        #Creating names and storing these in the dictionary
        X_train_name = f'X_train{i+1}'
        y_train_name = f'y_train{i+1}'
        
        train_datasets[X_train_name] = X_train 
        train_datasets[y_train_name] = y_train 
    
    return X_test, y_test, train_datasets, df_test

#%% 3. We create our actual ensemble
def LSTM_ensemble(data, rs, n_models, ycol, xcol, n_epoch):
    
    start_time_data = time.time()
    print("Creating and splitting data sets")
    Xtest, ytest, train_df, df_test = testtrain_LSTM(df= data, randomstate= rs, n_df= n_models, 
                                            y_col = ycol, x_col= xcol)
    
    start_time_model = time.time()
    
    #We create a list in which we store our trained models
    ensemble = []
    
    #Extracting the amount of models to train, divide by 2 as we have both X and y variabels stored
    num_models = int((len(train_df)/2))
    
    keys = list(train_df.keys())
    
    for i in range(num_models):
        
        #We extract the train set from our dict object
        X_train = train_df[keys[i*2]]
        y_train = train_df[keys[i*2+1]]
        
        #We build our model
        n_steps, n_features = np.shape(X_train)[1:3]

        model = Sequential()
        model.add(Masking(mask_value=-1))
        #model.add(LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, input_shape=(n_steps, n_features)))
        model.add(LSTM(64, recurrent_dropout=0.1, input_shape=(n_steps, n_features)))
        model.add(Dense(1, activation='sigmoid')) 

        model.compile(keras.optimizers.legacy.Adam(learning_rate=0.07), loss='binary_crossentropy', metrics=['AUC']) 

        model.fit(X_train, y_train, batch_size=1024, epochs=n_epoch, verbose=1, shuffle=True) 
        
        ensemble.append(model)
        print("Fitted model", i+1," - Models left to fit:", num_models - (i+1))
        
    
    #Now that all models are trained, we get all predictions using the test set and store them in a list
    ensemble_results = []
    print("Retrieving predictions from models")
    for i in range(len(ensemble)):
        model = ensemble[i]
        y_pred_test = model.predict(Xtest)
        
        ensemble_results.append(y_pred_test)
        print("Retrieved predictions from model", i+1)
    
    #Using all predictions, we use probability averaging for all predictions to get an ensemble prediction
    ens_test = pd.Series((np.rint((sum(ensemble_results)/len(ensemble_results))).astype(int)).reshape(-1))
    ens_prob = sum(ensemble_results)/len(ensemble_results)
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
    results = result_eval(ypred=ens_test, ytrue = ytest, yprob=ens_prob)
    results['elapsed_time_data'] = elapsed_time_data
    results['elapsed_time_model'] = elapsed_time_model
    results['n_models'] = n_models
    
    print(results)
    
    return results, ens_prob, df_test


#%% 4. We create a function that extracts probabilities which are used for descriptive attribution
def prob_attri_seq(test_set, y_prob):
    #We extract all campaigns for each touchpoint sequence
    df_eval = test_set.groupby('id')['campaign'].unique().reset_index()
    df_eval = df_eval.merge(y_prob.reset_index(), on = 'id', how = 'left')
    
    campaigns = testset['campaign'].drop_duplicates()
    
    #We calculate the average probability of conversion for each campaign
    average_probability_per_campaign = df_eval.explode('campaign').groupby('campaign')['y_pred_proba'].mean().reset_index()
    
    #Next, we calculate for each campaign the average probability of conversion 
    # for all touch points not containing that campaign
    avg_prob_without_campaign = []
    for campaign in tqdm(campaigns):
        av_prob = df_eval[df_eval['campaign'].apply(lambda x: campaign not in x)]['y_pred_proba'].mean()
        avg_prob_without_campaign.append({'campaign': campaign, 'avg_probability_without_campaign': av_prob})
    
    avg_prob_without_campaign = pd.DataFrame(avg_prob_without_campaign).sort_values(by='campaign').reset_index(drop=True)
    
    #We put it all together and extract attribution values by substracting the mean probability for each campaign
    #with the mean probability without that campaign
    df_attr_seq = {
        'campaign': avg_prob_without_campaign['campaign'],
        'avg campaign prob': average_probability_per_campaign['y_pred_proba'],
        'avg_probability_without_campaign': avg_prob_without_campaign['avg_probability_without_campaign'],
        'attribution values':  average_probability_per_campaign['y_pred_proba'] - avg_prob_without_campaign['avg_probability_without_campaign']
        }
    
    df_attr_seq = pd.DataFrame(df_attr_seq)
    
    df_attr_seq = df_attr_seq.sort_values(by='attribution values', ascending = False)
    
    return df_attr_seq

#As attribution values slightly differ each time, we use a five-fold validation process where we use a random
# test train split each time.

fivefold_attr = []
for i in range(0,5):
    res, yprob, df_test = LSTM_ensemble(data= df_padded, rs=i, n_models=10, ycol=['conversion','id'],
                                        xcol=['id','cost', 'campaign', 'timestamp','time_since_last_click','cat1', 
                                              'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'],
                                        n_epoch=4)
    testset = df_test[['campaign','id']]

    del res
    
    fivefold_attr.append(prob_attri_seq(test_set = testset, 
                                        y_prob = yprob)[['campaign', 'attribution values']])

fivefold_attr2 = pd.concat(fivefold_attr, ignore_index=True, axis = 0)
attribution_LSTM_Ens = fivefold_attr2.groupby('campaign')['attribution values'].mean().reset_index().sort_values(by='attribution values', ascending = False)

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LSTM Ens.tsv"
attribution_LSTM_Ens.to_csv(file_path, sep='\t', index=False)



