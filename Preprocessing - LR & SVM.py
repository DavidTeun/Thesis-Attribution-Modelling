#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:20:23 2024

@author: davidteunissen

This notebook contains the code used to preprocess the Criteo data set for the Logistic Regression and SVM 
methods

"""
#%% 1. Raw Criteo data is loaded and packages are imported
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.chdir('/Users/davidteunissen/Desktop/Msc Thesis/data/criteo_attribution_dataset/')
criteo_data=pd.read_csv('criteo_attribution_dataset.tsv',sep='\t')

#%% 2. Data is processed according to Ren steps without subsampling
def preprocess_nosub(df, bottom, top, norm_cols):
    
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
    df_final = df[df['sid'].isin(conv['sid'])]
    
    df_final = df_final[['sid', 'conversion','timestamp', 
                 'time_since_last_click', 'campaign', 
                 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 
                 'cat6', 'cat7', 'cat8', 'cat9', 'cost']]
    
    #Create a new data ID based on the series index
    new_ids = conv['sid'].reset_index()
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
    
    print("Normalising data...")
    
    def normaliser(df, columns):
        df_norm = df.copy()
        
        #Initialize MinMaxScaler
        scaler = MinMaxScaler()    
        
        #Noramlize specified columns
        df_norm[columns] = scaler.fit_transform(df_norm[columns])  
        
        return df_norm
    
    df_final_norm = normaliser(df_final, norm_cols)
    
    df_final_norm = df_final_norm.sort_values(by=['id','timestamp']).reset_index(drop=True)
    
    return df_final

#%% Running the function and saving the data set
df_processed_norm_clicked = preprocess_nosub(criteo_data, 3, 20, ['timestamp', 'time_since_last_click', 'campaign',
                                                     'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 
                                                     'cat7', 'cat8', 'cat9','cost'])

file_path = "/Users/davidteunissen/Desktop/Msc Thesis/Data/Processed_data/processed_nosub_norm.tsv"
df_processed_norm_clicked.to_csv(file_path, sep='\t', index=False)


