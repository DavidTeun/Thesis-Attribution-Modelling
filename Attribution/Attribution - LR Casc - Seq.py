#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:08:56 2023

@author: davidteunissen
This set of Python code creates a functioning cascade ensemble for the logistic regression with
descriptive attribution results

The code is set up in the following format:
    1. We split the data into a train, test and validation set
    2. We create a function that creates balanced training sets, based on the BalanceCascade idea
    3. We put all of it together and create the final cascade function
    4. We create a function that extracts probabilities which are used for descriptive attribution
    
ADDITIONALLY: This is an amended version which incorporates a methodology to convert observation-wise 
predictions to predictions for whole sequences for the logistic regression
"""

#%% 0. Packages are imported and pre-processed data is loaded
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
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
    trainsets_conv_ids = []
    
    conv_train_ids = train[train['conversion']==1]['id'].reset_index(drop=True)
    nonconv_train_ids = train[train['conversion']==0]['id'].reset_index(drop=True)
    for i in range(max_models):
        
        #We extract the sequences that we want in this trainset
        conv_sampled = conv_train_ids.sample(n=len(conv_train_ids), replace=True, random_state=i).reset_index(drop=True)
    
        
        #We concat these sets and append these to the ids list
        trainsets_conv_ids.append(conv_sampled)
    
    
    #Next we extract all relevant variables for the train, validation and test set and split into 
    #predictors and outcome variables
    
    #Train
    Train = df[df['id'].isin(train['id'])].sort_values(by=['id','timestamp'], ascending=True)
    
    #Val
    Val = df[df['id'].isin(val['id'])].sort_values(by=['id','timestamp'], ascending=True)
    
    #Test 
    Test = df[df['id'].isin(test['id'])].sort_values(by=['id','timestamp'], ascending=True)
    
    #Printing some statistics to check
    print("Fraction of data in train set:", (len(train)/len(df_ids)))
    print("Fraction of data in validation set:", len(val)/len(df_ids))
    print("Fraction of data in test set:", len(test)/len(df_ids))
  
    #Retrieve X and y columns for validation and test sets
    X_val = Val[x_col]
    y_val = Val[y_col]
    X_test = Test[x_col]
    y_test = Test[y_col]
    
    return trainsets_conv_ids, nonconv_train_ids, Train, X_val, y_val, X_test, y_test


#%% 2. We create a function that creates balanced training sets, based on the BalanceCascade idea
def BalanceCascade(converted_ids, nonconverted_ids, trainset, xcol, ycol):
    
    #We subsample the nonconverted sequences to get a balanced dataset
    nonconverted_sample = nonconverted_ids.sample(n=len(converted_ids), replace=False).reset_index(drop=True)
    
    #Next we collect all ids present in the subsampled balanced set
    ids = pd.concat([converted_ids, nonconverted_sample])
    
    #Finally, we extract all data present in the set
    trainset_sampled = trainset[trainset['id'].isin(ids)]

    #Check if dataset is balanced
    n_conv_bc = trainset_sampled[trainset_sampled['conversion']==1]['id'].nunique()
    n_nonconv_bc = trainset_sampled[trainset_sampled['conversion']==0]['id'].nunique()
    
    if(n_conv_bc == n_nonconv_bc):
        print("BalanceCascade dataset is balanced.")
    else: print("BalanceCascade dataset is unbalanced. Check: Ratio of converted to non-converted is:", 
                n_conv_bc/n_nonconv_bc)
    
    #Retrieve X and y dataframes
    X_train = trainset_sampled[xcol].reset_index(drop=True)
    y_train = trainset_sampled[ycol].reset_index(drop=True)
    
    return X_train, y_train, nonconverted_ids


#%% 3. We put all of it together and create the final cascade function
def LR_Cascade(data, x_cols, y_cols, randoms,
            threshold, measure,
            max_models,
            agg_metric_ens, agg_metric_seq,
            p_discard):
    
    start_time_data = time.time()
    print("Creating and splitting data sets")
    conv_ids, nonconv_ids, fulltrain, Xval, yval, Xtest, ytest = testtrainval(df=data,randomstate=randoms, y_col = y_cols,
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
    def LR(X_train, y_train):
        
        #Fit model
        model = LogisticRegression(C=100, solver='saga')
        model.fit(X_train, y_train)
        
        return model
    
    lr_init = LR(X_train=Xtrain1, y_train=ytrain1['conversion'])
    cascade.append(lr_init)
    
    #We create a list in which to store all predictions of Prob(conv_i=1)
    prediction = []
    prediction.append(lr_init.predict_proba(Xval)[:,1]) 
    
    #Using all predictions, we use probability averaging for all predictions to get an ensemble prediction
    def pred_aggr(agg_metric_ens_, agg_metric_seq_, pred_probs, y_true):
        
        def prob_aggr(pred_res, metric):
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
            'id': y_true['id'],
            'y_true': y_true['conversion'],
            'y_pred_proba': prob_aggr(pred_probs, agg_metric_ens_)
            }
        
        y_pred_eval = pd.DataFrame(y_pred_eval)
        
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
        
        y_prob_final, y_pred_final = seq_aggr(y_pred_eval, agg_metric_seq_)
        
        return y_prob_final, y_pred_final
    
    y_prob_val, y_val_pred = pred_aggr(agg_metric_ens_= agg_metric_ens, agg_metric_seq_=agg_metric_seq, 
                                       pred_probs = prediction, y_true = yval) 
    
    y_val_eval = yval.drop_duplicates().sort_values(by='id')['conversion']
    
    
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
    res_threshold1 = result_eval(ypred=y_val_pred, ytrue=y_val_eval, 
                                 yprob=y_prob_val)
    results_models.append(res_threshold1)
    
    #Based on the evaluation of this model, we add additional models. First we check whether the current single model functions
    #good enough
    
    if (results_models[0][measure] >= threshold)[0]:
        print("Cascade complete within 1 classifier, check if threshold is set too low.")
    else: #The current initial model is not accurate enough, hence we initialize a for-loop creating a cascade
        
    #First we introduce a function that allows us to update the data based on correclty identified whole 
        #sequences
        
        def DataUpdate(trainset, casc, x_columns, frac_discard):
            #First we extract the relevant columns from the whole training set to be able to evalute current model 
            #performance
            X_eval = trainset[['id','cost', 'campaign', 'timestamp','time_since_last_click','cat1', 
                               'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']].sort_values(by='id')
            y_eval = trainset[['conversion','id']].sort_values(by='id')
            
            
            #Next we extract all predictions
            pred_eval = []
            for i in range(len(casc)):
                model = casc[i]
                pred_eval.append(model.predict_proba(X_eval[x_columns])[:,1])
            
            #Retrieving all predictions to get a final cascade prediction
            pred_eval = np.rint((sum(pred_eval)/len(pred_eval))).astype(int)
            
            #Collecting all relevant data to be able to filter on correctly predicted 0 sequences
            y_check = {'id': y_eval['id'],
                       'y_true': y_eval['conversion'],
                       'y_pred': pred_eval
                }
            
            y_check = pd.DataFrame(y_check).reset_index(drop=True)

            #Split on zero's only
            y_check = y_check[y_check['y_true']==0]

            #Check if whole sequence is correclty identified
            y_check_ids = pd.DataFrame(y_check['id'].drop_duplicates().reset_index(drop=True))
            y_check_ids['correct_seq'] = (y_check.groupby('id')['y_pred'].sum()==0).reset_index(drop=True)

            #Filter only correctly identified sequences
            y_correct = y_check_ids[y_check_ids['correct_seq']==True]
            
            y_discard = y_correct['id'].sample(n=(int(round(len(y_correct)*frac_discard))), 
                                             replace=False, random_state=i).reset_index(drop=True)
            
            print("Amount of discarded sequences:", len(y_discard))
            
            #Finally, we can create a new data set, where we drop all correctly classified 0 sequences
            train_update = trainset[~trainset['id'].isin(y_discard)]
            
            return train_update
        
        for i in range(1, max_models):
            #First, we update the data set
            trainsets.append(DataUpdate(trainset = trainsets[i-1], casc = cascade, x_columns = x_cols, 
                                        frac_discard=p_discard))
            
            #Also, we update our nonconv_ids list
            nonconv_ids_newtrain = trainsets[i][trainsets[i]['conversion']==0]['id'].drop_duplicates()
            nonconv_ids = nonconv_ids[nonconv_ids.isin(nonconv_ids_newtrain)]
            
            #Next, we extract a balanced training set using balancecascade
            X_train_casc, y_train_casc, nonconv_ids = BalanceCascade(converted_ids=conv_ids[i], 
                                                                     nonconverted_ids = nonconv_ids, 
                                                                     trainset = trainsets[i], 
                                                                     xcol = x_cols, ycol = y_cols)
            
            #We subsequently train a new model and append this to the cascade
            lr_casc = LR(X_train=X_train_casc, y_train=y_train_casc['conversion'])
            cascade.append(lr_casc)
            
            #Adding a new prediction
            prediction.append(cascade[i].predict_proba(Xval)[:,1])
            
            #Aggregating predictions
            y_val_prob, y_val_pred = pred_aggr(agg_metric_ens_= agg_metric_ens, agg_metric_seq_=agg_metric_seq, 
                                               pred_probs = prediction, y_true = yval) 
            
            #Using our new prediction, we append our performance results to the results list
            results_models.append(result_eval(ypred=y_val_pred, ytrue=y_val_eval, 
                                              yprob = y_val_prob))
            
            #Finally, based on this performance, we check whether we have reached a suffcient level of 
            #certainty.
            if (results_models[i][measure] >= threshold)[0]:
                print("Training of cascade has reached sufficient confidence threshold with", i+1, "models.")
                break
            else: print("Cascade has finished training and evaluating iteration", i+1, "Confidence level was not reached and training subsequent model.")
        
        #After we have either reached the max number of models or our desired threshold, 
        #we evaluate the casade on our test set.
        
        predictions_test = []
        for i in range(len(cascade)):
            predictions_test.append(cascade[i].predict_proba(Xtest)[:,1])
        
        #Aggregating predictions
        y_test_prob, y_test_pred = pred_aggr(agg_metric_ens_= agg_metric_ens, agg_metric_seq_=agg_metric_seq, 
                                             pred_probs = predictions_test, y_true = ytest) 
        
        y_test_eval = ytest.drop_duplicates().sort_values(by='id')['conversion'].reset_index(drop=True)
        
        end_time = time.time()
    
        elapsed_time_data = end_time - start_time_data
        elapsed_time_model = end_time - start_time_model
        print("Total function runtime is:", elapsed_time_data)
        print("Ensemble function runtime is:", elapsed_time_model)
        
        results_final = result_eval(ypred=y_test_pred, ytrue=y_test_eval, 
                                    yprob=y_test_prob)
        results_final['elapsed_time_data'] = elapsed_time_data
        results_final['elapsed_time_model'] = elapsed_time_model
        results_final['p_discard'] = p_discard
        results_final['n_models'] = max_models
        results_final['agg_metric_ens'] = agg_metric_ens
        results_final['agg_metric_seq'] = agg_metric_seq
        
        
        print("Final performance of cascade on test set is:")
        print(results_final)
        
        return {"Final model": cascade, 
                "Prediction results": y_test_pred, 
                "Training sets": trainsets, 
                "Final test results": results_final, 
                "Intermittent results": results_models,
                "ytest": ytest,
                "Xtest": Xtest,
                "yprob": y_test_prob}
            

#%%
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


fivefold_attr = []
for i in range(0,5):
    res = LR_Cascade(data = df_processed,
                         x_cols = ['cost', 'campaign', 'timestamp','time_since_last_click','cat1', 'cat2', 
                                   'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'], 
                         y_cols = ['id','conversion'], threshold = 0.9, measure = 'AUC', max_models=4, 
                         agg_metric_ens = 'average', agg_metric_seq = 'average', p_discard = 0.2, randoms=i)

    Xtest = res['Xtest']
    ytest = res['ytest']
    yprob = res['yprob']
    testset = pd.concat([Xtest, ytest], axis=1)
    
    fivefold_attr.append(prob_attri_seq(test_set = testset, 
                                        y_prob = yprob)[['campaign', 'attribution values']])

fivefold_attr2 = pd.concat(fivefold_attr, ignore_index=True, axis = 0)
attribution_LR_Cas_Seq = fivefold_attr2.groupby('campaign')['attribution values'].mean().reset_index().sort_values(by='attribution values', ascending = False)

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LR Cas seq.tsv"
attribution_LR_Cas_Seq.to_csv(file_path, sep='\t', index=False)



