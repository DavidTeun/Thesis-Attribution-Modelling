#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:09:16 2023

@author: davidteunissen

This set of Python code creates a functioning cascade ensemble for the LSTM with
descriptive attribution results. We take the following steps:

The code is set up in the following format:
    1. We split the data into a train, test and validation set
    2. We create a function that creates balanced training sets, based on the BalanceCascade idea
    3. We put all of it together and create the final cascade function
    4. We create a function that extracts probabilities which are used for descriptive attribution
    
"""

#%%Importing packages and raw data
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split


#Performance metric packages
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#LSTM packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

#Reading data
df_padded = pd.read_csv('/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/padded_data_norm.tsv',sep='\t')


#%% 1. We split the data into a train, test and validation set
def testtrainval(df, randomstate, y_col, x_col, max_models):
    
    #We split the data in a test, train and validation sets using ids, stratifying on conversion
    #We get a 60/20/20 train/val/test split
    df_ids = df[['id', 'conversion']].drop_duplicates()
    
    trainval, test = train_test_split(
        df_ids, test_size=0.2, random_state = randomstate, shuffle=True,
        stratify = df_ids[['conversion']])
    
    train, val = train_test_split(
        trainval, test_size=0.25, random_state = randomstate, shuffle=True,
        stratify = trainval[['conversion']])
    
    #Next, we split the train data sets into distinct subsampled sets. 
    #We floor the number of conversion such that we do not get an instance where we want to sample more data 
    #than available
    trainsets_conv_ids = []
    #n_conv_sample = math.floor((train['conversion'].sum()/max_models))
    
    conv_train_ids = train[train['conversion']==1]['id'].reset_index(drop=True)
    nonconv_train_ids = train[train['conversion']==0]['id'].reset_index(drop=True)
    for i in range(max_models):
        
        #We extract the sequences that we want in this trainset
        conv_sampled = conv_train_ids.sample(n=len(conv_train_ids), replace=True, random_state=i).reset_index(drop=True)
    
        
        #We concat these sets and append these to the ids list
        trainsets_conv_ids.append(conv_sampled)
        
        #Finally, we remove these ids from our list of ids such that we create non-overlapping data sets
        #conv_train_ids = conv_train_ids[~conv_train_ids.isin(conv_sampled)]
    
    #Train
    Train = df[df['id'].isin(train['id'])]
    
    #Val
    Val = df[df['id'].isin(val['id'])]
    
    #Test 
    Test = df[df['id'].isin(test['id'])]
    
    #Printing some statistics to check
    print("Fraction of data in train set:", (len(train)/len(df_ids)))
    print("Fraction of data in validation set:", len(val)/len(df_ids))
    print("Fraction of data in test set:", len(test)/len(df_ids))
    
    print("Fraction of conversions in train set:", Train[Train['conversion']==1]['id'].nunique()/Train['id'].nunique())
    print("Fraction of conversions in validation set:", Val[Val['conversion']==1]['id'].nunique()/Val['id'].nunique())
    print("Fraction of conversions in test set:", Test[Test['conversion']==1]['id'].nunique()/Test['id'].nunique())
    
    #We use a function to transform our validation and test sets to a 3D shape for inputting in LSTM RNN
    #Next, we create a function that allows us to transform the data into a 3D format for LSTM
    def batches(X, seq_len, x_col_LSTM):
        
        #Reshape the data in the form of (ID, Sequence, Variables)
        batch_X = X[x_col_LSTM].to_numpy().reshape(X['id'].nunique(),seq_len, len(x_col_LSTM))

        return batch_X
    
    #We transform the test data to our desired format
    x_col_lstm = x_col.copy()
    x_col_lstm.remove('id')
    
    X_test = batches(Test, seq_len = 20, x_col_LSTM = x_col_lstm)
    y_test = Test[y_col].drop_duplicates().reset_index(drop=True)['conversion']
    
    X_val = batches(Val, seq_len = 20, x_col_LSTM = x_col_lstm)
    y_val = Val[y_col].drop_duplicates().reset_index(drop=True)['conversion']
    
    return trainsets_conv_ids, nonconv_train_ids, Train, X_val, y_val, X_test, y_test, Test

#%% 2. We create a function that creates balanced training sets, based on the BalanceCascade idea
def BalanceCascade(converted_ids, nonconverted_ids, trainset, xcol, ycol):
    
    #We subsample the nonconverted sequences to get a balanced dataset
    nonconverted_sample = nonconverted_ids.sample(n=len(converted_ids), replace=False).reset_index(drop=True)
    
    #Next we collect all ids present in the subsampled balanced set
    ids = pd.concat([converted_ids, nonconverted_sample])
    
    #Finally, we extract all data present in the set
    trainset_sampled = trainset[trainset['id'].isin(ids)]
    
    #Furthermore, we update the nonconverted ids set such that we remove all ids currently sampled
    #nonconverted_ids = nonconverted_ids[~nonconverted_ids.isin(nonconverted_sample)]
    
    #Check if dataset is balanced
    n_conv_bc = trainset_sampled[trainset_sampled['conversion']==1]['id'].nunique()
    n_nonconv_bc = trainset_sampled[trainset_sampled['conversion']==0]['id'].nunique()
    
    if(n_conv_bc == n_nonconv_bc):
        print("BalanceCascade dataset is balanced.")
    else: print("BalanceCascade dataset is unbalanced. Check: Ratio of converted to non-converted is:", 
                n_conv_bc/n_nonconv_bc)
    
    #Retrieve X and y dataframes in appropriate 3D format
    def batches(X, seq_len, x_col_LSTM):
        
        #Reshape the data in the form of (ID, Sequence, Variables)
        batch_X = X[x_col_LSTM].to_numpy().reshape(X['id'].nunique(),seq_len, len(x_col_LSTM))

        return batch_X
    
    #We transform the test data to our desired format
    x_col_lstm = xcol.copy()
    x_col_lstm.remove('id')
    
    X_train = batches(trainset_sampled, seq_len = 20, x_col_LSTM = x_col_lstm)
    y_train = trainset_sampled[ycol].drop_duplicates().reset_index(drop=True)['conversion']
    
    return X_train, y_train, nonconverted_ids


#%% 3. We put all of it together and create the final cascade function
def Cascade_LSTM(data, x_cols, y_cols,randoms,
            threshold, measure,
            max_models, fraction_discard,
            n_epoch):
    
    start_time_data = time.time()
    print("Creating and splitting data sets")
    conv_ids, nonconv_ids, fulltrain, Xval, yval, Xtest, ytest, testset = testtrainval(df=data,randomstate=randoms, y_col = y_cols,
                                                      x_col = x_cols, max_models=max_models)
    start_time_model = time.time()
    
    #First we create a list in which to store all classifiers to be trained in the cascade
    cascade = []
    
    #We also create a list in which we store all the train data sets
    trainsets = []
    trainsets.append(fulltrain)
    
    #We extract the first training dataset
    Xtrain1, ytrain1, nonconv_ids = BalanceCascade(converted_ids=conv_ids[0], nonconverted_ids = nonconv_ids, 
                                      trainset = fulltrain, xcol = x_cols, ycol = y_cols)
    
    #We initialize the cascade by training the first classifiers and add this to the cascade   
    
    print("Initializing Cascade - training first model.")
    #We build our model
    n_steps, n_features = np.shape(Xtrain1)[1:3]

    model = Sequential()
    model.add(Masking(mask_value=-1))
    #model.add(LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, input_shape=(n_steps, n_features)))
    model.add(LSTM(64, recurrent_dropout=0.1, input_shape=(n_steps, n_features)))
    model.add(Dense(1, activation='sigmoid')) 

    model.compile(keras.optimizers.legacy.Adam(learning_rate=0.07), loss='binary_crossentropy', metrics=['AUC']) 

    model.fit(Xtrain1, ytrain1, batch_size=1024, epochs=n_epoch, verbose=1, shuffle=True) 
    
    cascade.append(model)
    
    #We create a list in which to store all predictions of Prob(conv_i=1)
    prediction = []
    
    print("Retrieving prediction results from first model.")
    prediction.append(model.predict(Xval))    
    
    #Next we determine the prediction based on the predicted probabilities
    cas_pred = pd.Series((np.rint((sum(prediction)/len(prediction))).astype(int)).reshape(-1))
    cas_prob = sum(prediction)/len(prediction)
    #Evaluating results
    results_models = []
    
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
    
    #Evaluating result of first model
    res_threshold1 = result_eval(cas_pred, ytrue=yval, yprob=cas_prob)
    results_models.append(res_threshold1)
    
    #Based on the evaluation of this model, we add additional models. First we check whether the current single model functions
    #good enough
    
    if (results_models[0][measure] >= threshold)[0]:
        print("Cascade complete within 1 classifier, check if threshold is set too low.")
    else: #The current initial model is not accurate enough, hence we initialize a for-loop creating a cascade
        #First we introduce two functions, one reshapes our data in our desired format and one that 
        #allows us to update the data based on correctly identified whole sequences
        
        def batches(X, seq_len, x_col_LSTM):
            
            #Reshape the data in the form of (ID, Sequence, Variables)
            batch_X = X[x_col_LSTM].to_numpy().reshape(X['id'].nunique(),seq_len, len(x_col_LSTM))

            return batch_X
        
        def DataEval(trainset, model, x_columns, y_columns):
            print("Evaluating current performance on trainset")
            
            #First we extract the relevant columns from the whole training set to be able to evalute current model 
            #performance
            X_eval = trainset[x_columns].sort_values(by='id')
            y_eval = trainset[y_columns].sort_values(by='id')
            ids = trainset['id'].sort_values().drop_duplicates().reset_index(drop=True)
            
            x_pred_col = x_columns.copy()
            x_pred_col.remove('id')
            
            X_eval = batches(X_eval, seq_len=20, x_col_LSTM=x_pred_col)
            y_eval = y_eval.drop_duplicates().reset_index(drop=True)
            
            #Finally, we group the ids and the predicted probabilities
            pred_eval = {
                'id': ids,
                'pred': model.predict(X_eval).reshape(-1)
                }
            pred_eval = pd.DataFrame(pred_eval)
           
            return pred_eval, y_eval
        
        
        def Data_update(train_eval, y_eval, frac_discard, rs):

            #First we extract only predicted probabilities            
            y_prob_eval = train_eval.drop('id', axis=1)
            
            #Next we average and round these probabilities for a prediction
            y_pred_eval = y_prob_eval.sum(axis=1)
            y_pred_eval = np.rint((y_pred_eval/y_prob_eval.shape[1])).astype(int)
            
            #Collecting all relevant data to be able to filter on correctly predicted 0 sequences
            y_check = {'id': y_eval['id'],
                       'y_true': y_eval['conversion'],
                       'y_pred': y_pred_eval,
                       'correct_seq': y_eval['conversion'] == y_pred_eval
                }
            
            y_check = pd.DataFrame(y_check).reset_index(drop=True)

            #Split on zero's only
            y_check = y_check[y_check['y_true']==0]
            
            #Filter only correctly identified sequences
            y_check = y_check[y_check['correct_seq']==True]
            
            y_discard = y_check['id'].sample(n=(int(round(len(y_check)*frac_discard))), 
                                             replace=False, random_state=rs).reset_index(drop=True)
            
            print("Amount of discarded sequences:", len(y_discard))
            
            #Finally, we can create a new data set, where we drop all correctly classified 0 sequences
            train_update = train_eval[~train_eval['id'].isin(y_discard)]
            
            return train_update
    
        trainset_eval, y_true_eval = DataEval(trainset= trainsets[0], model = cascade[0], 
                                              x_columns = x_cols, y_columns = y_cols)
        
        for i in range(1, max_models):
            #First, we update the data set
            update_ids = Data_update(train_eval = trainset_eval, y_eval = y_true_eval, 
                                     frac_discard = fraction_discard , rs=i)
            
            trainsets.append(trainsets[i-1][trainsets[i-1]['id'].isin(update_ids['id'])])
            
            #Also, we update our nonconv_ids list
            nonconv_ids_newtrain = trainsets[i][trainsets[i]['conversion']==0]['id'].drop_duplicates()
            nonconv_ids = nonconv_ids[nonconv_ids.isin(nonconv_ids_newtrain)]
            
            #Next, we extract a balanced training set using balancecascade
            X_train_casc, y_train_casc, nonconv_ids = BalanceCascade(converted_ids=conv_ids[i], 
                                                                     nonconverted_ids = nonconv_ids, 
                                                                     trainset = trainsets[i], 
                                                                     xcol = x_cols, ycol = y_cols)
            
            #We subsequently train a new model and append this to the cascade
            n_steps, n_features = np.shape(X_train_casc)[1:3]
            
            model = Sequential()
            model.add(Masking(mask_value=-1))
            model.add(LSTM(64, recurrent_dropout=0.1, input_shape=(n_steps, n_features)))
            model.add(Dense(1, activation='sigmoid')) 
            
            model.compile(keras.optimizers.legacy.Adam(learning_rate=0.07), loss='binary_crossentropy', metrics=['AUC']) 
            
            model.fit(X_train_casc, y_train_casc, batch_size=1024, epochs=n_epoch, verbose=1, shuffle=True) 
            
            cascade.append(model)
            
            #Adding a new prediction
            prediction.append(cascade[i].predict(Xval))    
            
            #Next we update the prediction based on the predicted probabilities
            cas_pred = pd.Series((np.rint((sum(prediction)/len(prediction))).astype(int)).reshape(-1))
            cas_prob = sum(prediction)/len(prediction)
            #Using our new prediction, we append our performance results to the results list
            results_models.append(result_eval(cas_pred, ytrue=yval, yprob = cas_prob))
            
            #Finally, based on this performance, we check whether we have reached a suffcient level of 
            #certainty.
            if (results_models[i][measure] >= threshold)[0]:
                print("Training of cascade has reached sufficient confidence threshold with", i, "models.")
                break
            else: 
                if (i <= (max_models)): 
                    print("Cascade has finished training and evaluating iteration", i, " - Confidence level was not reached and training subsequent model.")
                    trainset_eval_new, y_true_eval = DataEval(trainset= trainsets[i], model = cascade[i], 
                                                              x_columns = x_cols, y_columns = y_cols)
                    trainset_eval_new = trainset_eval_new.rename(columns={'pred':f"pred_{i}"}) 
                    
                    trainset_eval = pd.merge(trainset_eval_new, trainset_eval, on = 'id', how = 'left')
                else: print("Threshold was not reached, evaluating total performance.")
        
        #After we have either reached the max number of models or our desired threshold, 
        #we evaluate the casade on our test set.
        
        predictions_test = []
        for i in range(len(cascade)):
            print("Retrieving final test predictions from model:", i+1)
            predictions_test.append(cascade[i].predict(Xtest))
        
        y_pred_test = pd.Series((np.rint((sum(predictions_test)/len(predictions_test))).astype(int)).reshape(-1))
        y_pred_prob = sum(predictions_test)/len(predictions_test)
        end_time = time.time()
    
        elapsed_time_data = end_time - start_time_data
        elapsed_time_model = end_time - start_time_model
        print("Total function runtime is:", elapsed_time_data)
        print("Ensemble function runtime is:", elapsed_time_model)
        
        results_final = result_eval(ypred=y_pred_test, ytrue=ytest, yprob=y_pred_prob)
        results_final['elapsed_time_data'] = elapsed_time_data
        results_final['elapsed_time_model'] = elapsed_time_model
        results_final['p_discard'] = fraction_discard
        results_final['n_models'] = max_models
        
        print("Final performance of cascade on test set is:")
        print(results_final)
        
        return {"Final model": cascade, 
                "Prediction results": y_pred_test, 
                "Training sets": trainsets, 
                "Final test results": results_final, 
                "Intermittent results": results_models,
                "yprob": y_pred_prob,
                "testset": testset}


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
    for campaign in campaigns:
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
    res = Cascade_LSTM(data=df_padded, randoms=i,
                            x_cols = ['id','cost', 'campaign', 'timestamp','time_since_last_click','cat1', 'cat2', 
                                      'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'], 
                            y_cols = ['id','conversion'], threshold = 1, measure = 'recall', max_models=4,
                            fraction_discard=0.2, n_epoch = 4)

    testset = res['testset']
    yprob = res['yprob']
    del res
    
    fivefold_attr.append(prob_attri_seq(test_set = testset, 
                                        y_prob = yprob)[['campaign', 'attribution values']])

fivefold_attr2 = pd.concat(fivefold_attr, ignore_index=True, axis = 0)
attribution_LSTM_Cas = fivefold_attr2.groupby('campaign')['attribution values'].mean().reset_index().sort_values(by='attribution values', ascending = False)

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LSTM Cas.tsv"
attribution_LSTM_Cas.to_csv(file_path, sep='\t', index=False)


