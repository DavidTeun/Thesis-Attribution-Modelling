#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:22:48 2024

@author: davidteunissen

This notebook contains the code used to preprocess the Criteo data set for the LSTM 
methods. We take the following steps:
1. We import the raw criteo data set, and perform some pre-processing:
    a. We filter the data based on sequence length according to Ren
    b. We normalise the data
    c. We pad the data to ensure all sequences are of the same length

"""

#%%Importing packages and raw data
import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

Reading data
criteo_data=pd.read_csv('/Users/davidteunissen/Desktop/Msc Thesis/data/criteo_attribution_dataset/criteo_attribution_dataset.tsv',sep='\t')

#%% 1. a) Preprocessing the data set based on Ren
def preprocess_nosub(df, bottom, top):
    ## Set Random seed for replication of results
    np.random.seed(33)
    
    # Create a unique sequence ID for each sequence to be able to distinguish 
    # between conversions and non-conversions of a single user id
    df['sid'] = df['uid'].astype(str) + '.' + df['conversion_id'].astype(str)
    
    
    #For each sequence ID, compute sequence length
    seq_length = df.groupby('sid')['sid'].count().reset_index(name = "seq_length")
    
    #Check if all sequence IDs have a sequence length
    if seq_length['sid'].isin(df['sid']).all():
        print("Check: All sequences have a corresponding sequence length.")
    
    
    #Filter data based on sequence length 
    seq_length_filtered = seq_length[seq_length['seq_length'].isin(list(range(bottom,top+1)))]
    seq_length_filtered = seq_length[seq_length['seq_length'].isin(list(range(bottom,top+1)))]
    
    print("Number of sequences before filter:", len(seq_length))
    print("Sequences are filtered on sequence length betweem values", bottom, "and", top)
    print("Number of sequences after filter:", len(seq_length_filtered))
    
    #Obtain conversions & sequence length
    conv = seq_length_filtered.merge(df[['sid','conversion']], on='sid', how = 'left').drop_duplicates()
    conv = conv.reset_index(drop=True)
    
    #Check if merge has been performed correctly
    if conv['sid'].isin(df['sid']).all():
        print("Check: Merge of sequence length and conversions was succesfull.")
    
    #Split conversions and non-conversions on sequence ID
    conv_list = conv[conv['conversion'] == 1]['sid'].reset_index(drop=True)
    nonconv_list = conv[conv['conversion'] == 0]['sid'].reset_index(drop=True)
        
    #Full data set of sequence IDs
    full_data_id = pd.concat([conv_list, nonconv_list]).reset_index(drop=True)
    
    #Print some data statistics
    print("Total conversions in the data set:", len(conv_list), ", Percentage:", len(conv_list)/len(conv)*100)
    print("Total non-conversions in the data set:", len(nonconv_list), ", Percentage:", len(nonconv_list)/len(conv)*100)
    print("Percentage conversions in data set:", len(conv_list)/len(full_data_id)*100)
    
    
    #Retrieve full data set
    df_final = df[df['sid'].isin(full_data_id)]
    
    df_final = df_final[['sid', 'conversion', 'timestamp', 
                 'time_since_last_click', 'campaign', 
                 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 
                 'cat6', 'cat7', 'cat8', 'cat9', 'cost']]
    
    #Check if filter was done correctly
    if df_final['sid'].isin(full_data_id).all():
        print("Check: Filter of data on subsampled IDs was succesfull.")
    
    #Create a new data ID based on the series index
    new_ids = full_data_id.reset_index()
    new_ids.rename(columns={'index': 'id'}, inplace=True)
    
    #Insert new IDs in df and drop previous sequence ID
    df_final = df_final.merge(new_ids, on ='sid', how='left')
    
    #Check if Merge was done correctly
    if df_final['id'].isin(new_ids['id']).all():
        print("Check: Merge of data with new IDs was succesfull.")
    
    # Perform checks
    # Check if there are any duplicates    
    if not df_final.duplicated().any():
        print("Check: Data set does not contain duplicates.")
    else:
        print("Check: Data set contains duplicates, check returned data.")    
    
    return df_final

#%% 1. b) We normalise the data
def min_max_columns(df, cols):
    df_ext = df.copy()
    
    min_max_scaler = MinMaxScaler()
    for column in cols:
        x = df_ext[column].values.reshape(-1, 1) 
        df_ext[column] = min_max_scaler.fit_transform(x)
    
    return df_ext

df_normalised = min_max_columns(df_processed, cols = ['timestamp', 'time_since_last_click', 'campaign',
                                                      'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 
                                                      'cat8', 'cat9','cost'])

#%% 1. c) We pad the data such that all sequences are of length 20
def padding(df, pad_cols):
    #Create a sequence length variabel for each sequence
    seq_length = df.groupby('sid')['sid'].count().reset_index(name = "seq_length")
    
    #Determine the range of sequence lengths in the dataframe
    bottom = seq_length['seq_length'].min()
    top = seq_length['seq_length'].max()
    
    df = df.merge(seq_length, on='sid', how = 'left')    
    
    for seqlen in range(bottom, top):
        #Retrieve IDs corresponding to this sequence length
        ids = df[df['seq_length']==seqlen][['sid','conversion','id']].reset_index(drop=True).drop_duplicates()
        
        #Create a padding dataframe for each corresponding sequence length to get a total of 20 touch points
        ids = pd.concat([ids]*(top-seqlen), ignore_index=True)
        
        #Create the padding data frame
        padding = pd.DataFrame(np.zeros([len(ids), len(pad_cols)]),
                             columns = pad_cols)
        #We use NaN values for now to ensure we can sort dataframe according to touch point sequences
        padding[:]= np.nan
        padding['seq_length'] = seqlen
        
        #Adding dataframes together
        padding = pd.concat([ids, padding], axis = 1)
        
        df = pd.concat([df, padding])
        
        df_check = df[df['seq_length']==seqlen]
    
        seq_length_check = df_check.groupby('sid')['sid'].count().reset_index(name = "seq_length")
    
        if(seq_length_check['seq_length'].unique() == top):
            print("Padding was successfull for sequence length", seqlen)
        else:
            print("Padding was unsuccessfull for sequence length", seqlen)
    
    df.drop('seq_length', axis=1, inplace=True)
        
    seq_length_check = df.groupby('sid')['sid'].count().reset_index(name = "seq_length")

    if seq_length_check['seq_length'].unique() == top:
        print("Padding was successfull for all sequence lengths.")
    else: print("Padding was unsuccesfull.")
    
    df = df.sort_values(by=['id','timestamp'], ascending=[True,True]).reset_index(drop=True)
    
    #We replace all NaN values by -1 as they are no longer necessary for sorting
    df = df.fillna(-1)
    
    return df

df_padded = padding(df_normalised, pad_cols = ['timestamp', 'campaign', 'time_since_last_click',
                                              'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 
                                              'cat7', 'cat8', 'cat9', 'cost'])

#%% Saving data
file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/padded_data_norm.tsv"
df_padded.to_csv(file_path, sep='\t', index=False)
