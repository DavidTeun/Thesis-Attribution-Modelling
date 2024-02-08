#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:37:32 2023

@author: davidteunissen
"""

#%% Importing dataset and packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

#We load in the criteo data
os.chdir('/Users/davidteunissen/Desktop/Msc Thesis/Data/criteo_attribution_dataset')
data_raw=pd.read_csv('criteo_attribution_dataset.tsv',sep='\t')
#%% Raw data set exploration
data_raw.columns
data_raw.info()

data_info = data_raw.describe()

#Number of Touch points
len(data_raw)

#Number of Campaigns: 675
data_raw['campaign'].nunique()

#Sequences : 6.1 million, non-transformed
# Create a unique sequence ID for each sequence to be able to distinguish 
# between conversions and non-conversions of a single uid
data_raw['sid'] = data_raw['uid'].astype(str) + '.' + data_raw['conversion_id'].astype(str)
data_raw['sid'].nunique()/10**6
data_raw['uid'].nunique()/10**6


#Unique Conversions 438730
conv = data_raw[['conversion','sid']]
conv = conv[conv['conversion']==1]
conv['sid'].nunique()

#Raw Conversion percentage: 6.734855938126456
conv['sid'].nunique() / data_raw['sid'].nunique() * 100

len(conv.drop_duplicates())
# Variable Counting
v_count = data_raw[['uid', 'sid', 'campaign','cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6',
                    'cat7', 'cat8','cat9']].nunique()
v_count

# Convert timestamp column to days
data_raw['day'] = np.floor(data_raw.timestamp / 86400.).astype(int)
data_raw['conversion_day'] = np.floor(data_raw.conversion_timestamp / 86400.).astype(int)

# Create variable indicating days gap between conversion and current day
data_raw['gap_click_sale'] = -1
data_raw.loc[data_raw.conversion == 1, 'gap_click_sale'] = data_raw.conversion_day - data_raw.day

#%% Raw data set plots according to sequence length

#For each sequence ID, compute sequence length and add to raw data
seq_length = data_raw.groupby('sid')['sid'].count().reset_index(name = "seq_length")

seq_length.groupby(['seq_length']).count()

data_seq = data_raw.merge(seq_length, on = 'sid', how = 'left')


data_seq.columns
#Plot a histogram counting the number of conversions for each sequence length
conv_hist = data_seq[['conversion','seq_length']]
conv_hist = conv_hist[conv_hist['conversion'] == 1]
conv_hist = conv_hist[conv_hist['seq_length'] <= 30]


plt.hist(conv_hist['seq_length'], bins = 30, edgecolor = 'black', color = 'green')
plt.title('Conversions per sequence length', loc='center')
plt.xlabel('Sequence Length')
plt.ylabel('Number of Conversions')
plt.show()


#Plot a histogram counting the number of nonconversions for each sequence length
nonconv_hist = data_seq[['conversion','seq_length']]
nonconv_hist = nonconv_hist[nonconv_hist['conversion'] == 0]
nonconv_hist = nonconv_hist[nonconv_hist['seq_length'] <= 30]


plt.hist(nonconv_hist['seq_length'], bins = 30, edgecolor = 'black', color = 'red')
plt.title('Non-conversions per sequence length', loc='center')
plt.xlabel('Sequence Length')
plt.ylabel('Number of Non-Conversions')
plt.show()


# Stacked
plt.hist([conv_hist['seq_length'], nonconv_hist['seq_length']], bins = 30, stacked=True, 
         label = ['Converted sequences', 'Non-converted sequences'], edgecolor = 'black', color = ['green','red'])
plt.title('(Non-)conversions per sequence length', loc='center')
plt.xlabel('Sequence length')
plt.ylabel('Number of (non-)conversions')
plt.legend()
plt.show()

#Make a plot showing the relative percentage of total conversions compared to sequence length

#Stricly converted sequences
c11 = data_seq[['uid','sid']][data_seq['conversion']==1]

#Count sequence length for each converted sequence
c22 = c11.groupby('sid')['sid'].count().reset_index(name="seq_length")

#Count conversions for each sequence length
c33 = c22.groupby(['seq_length'])['seq_length'].count().reset_index(name="frequency")

freq = c33.frequency/c33.frequency.sum()
x = c33.seq_length

plt.plot(x[x<=80], freq[x<=80], label='Converted Sequences', color = 'red')
plt.yscale('log')
plt.xlim(0, 100)
plt.title('Fraction of conversions per sequence length', loc='center')
plt.xlabel('Sequence length')
plt.ylabel('Fraction of Converted Sequences')
plt.show()

#Make a plot showing the relative percentage of total non-conversions compared to sequence length

#Strictly non-converted sequences 
c1 = data_seq[['uid','sid']][data_seq['conversion']==0]

#Count sequence length for each non-converted sequence
c2 = c1.groupby('sid')['sid'].count().reset_index(name="seq_length")

#Count non-conversions for each sequence length
c3 = c2.groupby(['seq_length'])['seq_length'].count().reset_index(name="frequency")

freq_nonconv = c3.frequency/c3.frequency.sum()
x_nonconv = c3.seq_length

plt.plot(x_nonconv, freq_nonconv, label='Non-Converted Sequences', color = 'red')
plt.yscale('log')
plt.xlim(0, 100)
plt.title('Fraction of non-conversions per sequence length', loc='center')
plt.xlabel('Sequence length')
plt.ylabel('Fraction of Non-Converted Sequences')
plt.show()


## Stack the two plots
plt.plot(x_nonconv[x_nonconv<=80], freq_nonconv[x_nonconv<=80], label='Non-Converted Sequences', color = 'red')
plt.plot(x[x<=80], freq[x<=80], label='Converted Sequences', color = 'green')
plt.yscale('log')
plt.xlim(0, 80)
plt.title('Fraction of (non-)conversions per sequence length', loc='center')
plt.xlabel('Sequence length')
plt.ylabel('Fraction of (non-)converted sequences')
plt.legend()
plt.show()

