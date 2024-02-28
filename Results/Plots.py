#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:34:09 2024

@author: davidteunissen

This piece of code is used to generate the figures used in the thesis
"""
#%% Importing packages and results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Logistic Regression
LR_ensemble = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LR/LR_ens_res_5_30_bagged.tsv",
                          sep='\t')
LR_ensemble_seq = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LR/LR_ens_seq_res_5_30_bagged.tsv",
                          sep='\t')

LR_cascade = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LR/LR_cas_res_p_disc_n_mod_bagged.tsv",
                          sep='\t')
LR_cascade_seq = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LR/LR_cas_seq_p_disc_n_mod_bagged.tsv",
                          sep='\t')

#SVM
SVM_ensemble = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/SVM/SVM_ens_res_5_30_bagged.tsv",
                          sep='\t')
SVM_ensemble_seq = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/SVM/SVM_ens_seq_res_5_30_bagged.tsv",
                          sep='\t')

SVM_cascade = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/SVM/SVM_cas_res_p_disc_n_mod_bagged.tsv",
                          sep='\t')
SVM_cascade_seq = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/SVM/SVM_cas_seq_p_disc_max_mod_bagged.tsv",
                          sep='\t')

#LSTM
LSTM_ensemble = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LSTM/LSTM_ens_res_5_30_bagged.tsv",
                          sep='\t')
LSTM_cascade = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Bagged/LSTM/LSTM_cas_res_p_disc_max_mod_bagged.tsv",
                          sep='\t')

#%% Ensemble Plots - individual models

## Logistic Regression
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['AUC'], label = 'AUC')
ax.plot(LR_ensemble['n_models'], LR_ensemble['precision'], label = 'Precision')
ax.plot(LR_ensemble['n_models'], LR_ensemble['recall'], label = 'recall')
ax.plot(LR_ensemble['n_models'], LR_ensemble['accuracy'], label = 'accuracy')
ax.plot(LR_ensemble['n_models'], LR_ensemble['F-measure'], label = 'F-measure')
ax.plot(LR_ensemble['n_models'], LR_ensemble['log-loss'], label = 'log-loss')
ax.plot(LR_ensemble['n_models'], LR_ensemble['brier score'], label = 'brier score')
ax.set_xlabel('Number of models')
ax.set_ylabel('Metric')
ax.set_title('Logistic Regression Ensemble')
ax.legend(loc='lower right')
plt.show()

## Logistic Regression - Sequence Aggregation
fig, ax = plt.subplots()
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['AUC'], label = 'AUC')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['precision'], label = 'Precision')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['recall'], label = 'recall')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['accuracy'], label = 'accuracy')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['F-measure'], label = 'F-measure')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['log-loss'], label = 'log-loss')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['brier score'], label = 'brier score')
ax.set_xlabel('Number of models')
ax.set_ylabel('Metric')
ax.set_title('Logistic Regression Ensemble - Sequence aggregation')
ax.legend(loc='lower right')
plt.show()

## SVM
fig, ax = plt.subplots()
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['AUC'], label = 'AUC')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['precision'], label = 'Precision')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['recall'], label = 'recall')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['accuracy'], label = 'accuracy')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['F-measure'], label = 'F-measure')
ax.set_xlabel('Number of models')
ax.set_ylabel('Metric')
ax.set_title('SVM Ensemble')
ax.legend(loc='lower right')
plt.show()

## SVM - Sequence Aggregation
fig, ax = plt.subplots()
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['AUC'], label = 'AUC')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['precision'], label = 'Precision')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['recall'], label = 'recall')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['accuracy'], label = 'accuracy')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['F-measure'], label = 'F-measure')
ax.set_xlabel('Number of models')
ax.set_ylabel('Metric')
ax.set_title('SVM Ensemble - Sequence aggregation')
ax.legend(loc='lower right')
plt.show()

## LSTM
fig, ax = plt.subplots()
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['AUC'], label = 'AUC')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['precision'], label = 'Precision')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['recall'], label = 'recall')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['accuracy'], label = 'accuracy')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['F-measure'], label = 'F-measure')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['log-loss'], label = 'log-loss')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['brier score'], label = 'brier score')
ax.set_xlabel('Number of models')
ax.set_ylabel('Metric')
ax.set_title('LSTM Ensemble')
ax.legend(loc='lower right')
plt.show()

#%% Ensemble plots - Measures

#AUC
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['AUC'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['AUC'], label = 'Logistic Regression - Seq.', 
        color = 'orange')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['AUC'], label = 'SVM')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['AUC'], label = 'SVM - Seq.')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['AUC'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('AUC')
ax.set_title('AUC - Logistic Regression, SVM, and LSTM')
ax.legend(loc='upper right')
plt.show()

#Log-loss
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['log-loss'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['log-loss'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['log-loss'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Log-Loss')
ax.set_title('Log-Loss - Logistic Regression and LSTM')
ax.legend(loc='lower right')
plt.show()

#Brier score
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['brier score'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['brier score'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['brier score'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Brier Score')
ax.set_title('Brier Score - Logistic Regression and LSTM')
ax.legend(loc='lower right')
plt.show()

#Precision
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['precision'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['precision'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['precision'], label = 'SVM')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['precision'], label = 'SVM - Seq.')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['precision'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Precision')
ax.set_title('Precision - Logistic Regression, SVM, and LSTM')
ax.legend(loc='upper right')
plt.show()

#F1 - Score
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['F-measure'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['F-measure'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['F-measure'], label = 'SVM')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['F-measure'], label = 'SVM - Seq.')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['F-measure'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score - Logistic Regression, SVM, and LSTM')
ax.legend(loc='lower right')
plt.show()

#Recall
fig, ax = plt.subplots()
ax.plot(LR_ensemble['n_models'], LR_ensemble['recall'], label = 'Logistic Regression', color = 'red')
ax.plot(LR_ensemble_seq['n_models'], LR_ensemble_seq['recall'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(SVM_ensemble['n_models'], SVM_ensemble['recall'], label = 'SVM')
ax.plot(SVM_ensemble_seq['n_models'], SVM_ensemble_seq['recall'], label = 'SVM - Seq.')
ax.plot(LSTM_ensemble['n_models'], LSTM_ensemble['recall'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Recall')
ax.set_title('Recall - Logistic Regression, SVM, and LSTM')
ax.legend(loc='lower left')
plt.show()

#%% Cascade - measure plots
lr_casc = LR_cascade[LR_cascade['p_discard']==0.2]
lr_casc_seq = LR_cascade_seq[LR_cascade_seq['p_discard']==0.2]
svm_casc = SVM_cascade[SVM_cascade['p_discard']==0.2]
svm_casc_seq = SVM_cascade_seq[SVM_cascade_seq['p_discard']==0.2]
lstm_casc = LSTM_cascade[LSTM_cascade['p_discard']==0.2]

#AUC
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['AUC'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['AUC'], label = 'Logistic Regression - Seq.', 
        color = 'orange')
ax.plot(svm_casc['n_models'], svm_casc['AUC'], label = 'SVM')
ax.plot(svm_casc_seq['n_models'], svm_casc_seq['AUC'], label = 'SVM - Seq.')
ax.plot(lstm_casc['n_models'], lstm_casc['AUC'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('AUC')
ax.set_title('AUC - Logistic Regression, SVM, and LSTM')
ax.legend(loc='upper right')
plt.show()

#Log-loss
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['log-loss'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['log-loss'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(lstm_casc['n_models'], lstm_casc['log-loss'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Log-Loss')
ax.set_title('Log-Loss - Logistic Regression and LSTM')
ax.legend(loc='lower right')
plt.show()

#Brier score
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['brier score'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['brier score'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(lstm_casc['n_models'], lstm_casc['brier score'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Brier Score')
ax.set_title('Brier Score - Logistic Regression and LSTM')
ax.legend(loc='lower right')
plt.show()

#Precision
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['precision'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['precision'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(svm_casc['n_models'], svm_casc['precision'], label = 'SVM')
ax.plot(svm_casc_seq['n_models'], svm_casc_seq['precision'], label = 'SVM - Seq.')
ax.plot(lstm_casc['n_models'], lstm_casc['precision'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Precision')
ax.set_title('Precision - Logistic Regression, SVM, and LSTM')
ax.legend(loc='upper right')
plt.show()

#F1 - Score
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['F-measure'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['F-measure'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(svm_casc['n_models'], svm_casc['F-measure'], label = 'SVM')
ax.plot(svm_casc_seq['n_models'], svm_casc_seq['F-measure'], label = 'SVM - Seq.')
ax.plot(lstm_casc['n_models'], lstm_casc['F-measure'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score - Logistic Regression, SVM, and LSTM')
ax.legend(loc='lower right')
plt.show()

#Recall
fig, ax = plt.subplots()
ax.plot(lr_casc['n_models'], lr_casc['recall'], label = 'Logistic Regression', color = 'red')
ax.plot(lr_casc_seq['n_models'], lr_casc_seq['recall'], label = 'Logistic Regression - Seq.', color = 'orange')
ax.plot(svm_casc['n_models'], svm_casc['recall'], label = 'SVM')
ax.plot(svm_casc_seq['n_models'], svm_casc_seq['recall'], label = 'SVM - Seq.')
ax.plot(lstm_casc['n_models'], lstm_casc['recall'], label = 'LSTM', color = 'green')
ax.set_xlabel('Number of models')
ax.set_ylabel('Recall')
ax.set_title('Recall - Logistic Regression, SVM, and LSTM')
ax.legend(loc='lower left')
plt.show()


#%% Cascade plot - trade-off
#LR
for frac_disc in [0.1,0.5,1]:
    lr_plot = LR_cascade[LR_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)

#LR - seq.
for frac_disc in [0.1,0.5,1]:
    lr_plot = LR_cascade_seq[LR_cascade_seq['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)

#SVM
for frac_disc in [0.1,0.5,1]:
    lr_plot = SVM_cascade[SVM_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)
    
#SVM - seq.
for frac_disc in [0.1,0.5,1]:
    lr_plot = SVM_cascade_seq[SVM_cascade_seq['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)

#LSTM
for frac_disc in [0.1,0.5,1]:
    lr_plot = LSTM_cascade[LSTM_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)

#%% Cascade plots - LR
for max_mod in list(set(LR_cascade['n_models'])):
    lr_plot = LR_cascade[LR_cascade['n_models']==max_mod]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['p_discard'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['p_discard'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['p_discard'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['p_discard'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['p_discard'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['p_discard'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['p_discard'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Fraction of discarded sequences')
    ax.set_ylabel('Metric')
    ax.set_title(f"Logistic Regression Cascade - Number of models:{max_mod}")
    ax.legend(loc='lower right')
    plt.show()

for frac_disc in list(set(LR_cascade['p_discard'])):
    lr_plot = LR_cascade[LR_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.set_title(f"Logistic Regression Cascade - frac discard:{frac_disc}")
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    print(frac_disc)
    
for max_mod in list(set(LR_cascade_seq['n_models'])):
    lr_plot = LR_cascade_seq[LR_cascade_seq['n_models']==max_mod]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['p_discard'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['p_discard'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['p_discard'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['p_discard'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['p_discard'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['p_discard'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['p_discard'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Fraction of discarded sequences')
    ax.set_ylabel('Metric')
    ax.set_title(f"Logistic Regression Cascade - Sequence Aggregation- Number of models:{max_mod}")
    ax.legend(loc='lower right')
    plt.show()

for frac_disc in list(set(LR_cascade_seq['p_discard'])):
    lr_plot = LR_cascade_seq[LR_cascade_seq['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.set_title(f"Logistic Regression Cascade - Seq. - Fraction of discarded sequences:{round(frac_disc,1)}")
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()

#%% Cascade plots - SVM
for max_mod in list(set(SVM_cascade['n_models'])):
    lr_plot = SVM_cascade[SVM_cascade['n_models']==max_mod]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['p_discard'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['p_discard'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['p_discard'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['p_discard'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['p_discard'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Fraction of discarded sequences')
    ax.set_ylabel('Metric')
    ax.set_title(f"SVM Cascade - Number of models:{max_mod}")
    ax.legend(loc='lower right')
    plt.show()

for frac_disc in list(set(SVM_cascade['p_discard'])):
    lr_plot = SVM_cascade[SVM_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.set_title(f"SVM Cascade - Fraction of discarded sequences:{round(frac_disc,1)}")
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    
for max_mod in list(set(SVM_cascade_seq['n_models'])):
    lr_plot = SVM_cascade_seq[SVM_cascade_seq['n_models']==max_mod]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['p_discard'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['p_discard'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['p_discard'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['p_discard'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['p_discard'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Fraction of discarded sequences')
    ax.set_ylabel('Metric')
    ax.set_title(f"SVM Cascade - Sequence Aggregation - Number of models:{max_mod}")
    ax.legend(loc='lower right')
    plt.show()

for frac_disc in list(set(SVM_cascade_seq['p_discard'])):
    lr_plot = SVM_cascade_seq[SVM_cascade_seq['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.set_title(f"SVM Cascade - Sequence Aggregation - Fraction of discarded sequences:{round(frac_disc,1)}")
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    
#%% Cascade plots - LSTM
for max_mod in list(set(LSTM_cascade['n_models'])):
    lr_plot = LSTM_cascade[LSTM_cascade['n_models']==max_mod]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['p_discard'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['p_discard'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['p_discard'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['p_discard'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['p_discard'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['p_discard'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['p_discard'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Fraction of discarded sequences')
    ax.set_ylabel('Metric')
    ax.set_title(f"LSTM Cascade - Number of models:{max_mod}")
    ax.legend(loc='lower right')
    plt.show()

for frac_disc in list(set(LSTM_cascade['p_discard'])):
    lr_plot = LSTM_cascade[LSTM_cascade['p_discard']==frac_disc]
    fig, ax = plt.subplots()
    ax.plot(lr_plot['n_models'], lr_plot['AUC'], label = 'AUC')
    ax.plot(lr_plot['n_models'], lr_plot['precision'], label = 'Precision')
    ax.plot(lr_plot['n_models'], lr_plot['recall'], label = 'recall')
    ax.plot(lr_plot['n_models'], lr_plot['accuracy'], label = 'accuracy')
    ax.plot(lr_plot['n_models'], lr_plot['F-measure'], label = 'F-measure')
    ax.plot(lr_plot['n_models'], lr_plot['log-loss'], label = 'log-loss')
    ax.plot(lr_plot['n_models'], lr_plot['brier score'], label = 'brier score')
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Metric')
    ax.set_title(f"LSTM Cascade - Fraction of discarded sequences:{round(frac_disc,1)}")
    ax.legend(loc='lower right')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    
    
#%% Next we create tables - Ensemble Table
Ensemble_table = [LR_ensemble.iloc[0].rename("LR Ensemble"),
                  LR_ensemble_seq.iloc[0].rename("LR Ensemble - Seq."),
                  SVM_ensemble.iloc[0].rename("SVM Ensemble"),
                  SVM_ensemble_seq.iloc[0].rename("SVM Ensemble - Seq."),
                  LSTM_ensemble.iloc[1].rename("LSTM Ensemble")]
Ensemble_table = pd.concat(Ensemble_table, axis=1).T
columns = ['log-loss', 'recall', 'precision', 'F-measure', 'AUC','elapsed_time_model', 'n_models',
           'brier score']
Ensemble_table[columns] = Ensemble_table[columns].astype(float)
Ensemble_table[columns] = Ensemble_table[columns].round(4)
columns_table = ['log-loss', 'AUC', 'brier score', 'precision',
                 'elapsed_time_model']
ens_table = Ensemble_table[columns_table].to_latex()
print(ens_table)


#%% Next we create tables - Cascade Table
Cascade_table = [LR_cascade.iloc[13].rename("LR Cascade"),
                  LR_cascade_seq.iloc[13].rename("LR Cascade - Seq."),
                  SVM_cascade.iloc[13].rename("SVM Cascade"),
                  SVM_cascade_seq.iloc[13].rename("SVM Cascade - Seq."),
                  LSTM_cascade.iloc[13].rename("LSTM Cascade")]
Cascade_table = pd.concat(Cascade_table, axis=1).T
columns = ['log-loss', 'recall', 'precision', 'accuracy', 'F-measure', 'AUC',
           'elapsed_time_data', 'elapsed_time_model', 'p_discard']
Cascade_table[columns] = Cascade_table[columns].astype(float)
Cascade_table[columns] = Cascade_table[columns].round(4)

columns_table = ['log-loss', 'AUC', 'brier score', 'precision', 'F-measure',
                 'elapsed_time_model']
cascade_table = Cascade_table[columns_table].to_latex()
print(cascade_table)


#%% Both tables
columns_table = ['log-loss', 'AUC','elapsed_time_model']
Comparison_table = pd.concat([Ensemble_table[columns_table],Cascade_table[columns_table]])
Comparison_table = Comparison_table.to_latex()
print(Comparison_table)

#%% We create the attribution table
LR_ensemble_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LR Ens.tsv",
                          sep='\t')
LR_ensemble_seq_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LR Ens seq.tsv",
                          sep='\t')

LR_cas_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LR Cas.tsv",
                          sep='\t')
LR_cas_seq_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LR Cas seq.tsv",
                          sep='\t')

LSTM_ensemble_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LSTM Ens.tsv",
                          sep='\t')
LSTM_cas_attr = pd.read_csv("/Users/davidteunissen/Desktop/Msc Thesis/Thesis code/Results/Attribution/LSTM Cas.tsv",
                          sep='\t')


attr_table = {"LR Ensemble": pd.concat([LR_ensemble_attr.head(10)['campaign'], LR_ensemble_attr.tail(10)['campaign']]).reset_index(drop=True),
              "LR Ensemble - Seq.":  pd.concat([LR_ensemble_seq_attr.head(10)['campaign'], LR_ensemble_seq_attr.tail(10)['campaign']]).reset_index(drop=True),
              "LR Cascade": pd.concat([LR_cas_attr.head(10)['campaign'], LR_cas_attr.tail(10)['campaign']]).reset_index(drop=True),
              "LR Cascade - Seq": pd.concat([LR_cas_seq_attr.head(10)['campaign'], LR_cas_seq_attr.tail(10)['campaign']]).reset_index(drop=True),
              "LSTM Ensemble": pd.concat([LSTM_ensemble_attr.head(10)['campaign'], LSTM_ensemble_attr.tail(10)['campaign']]).reset_index(drop=True),
              "LSTM Cascade":pd.concat([LSTM_cas_attr.head(10)['campaign'], LSTM_cas_attr.tail(10)['campaign']]).reset_index(drop=True),
    }

df_attribution = pd.DataFrame(attr_table)
df_attribution_latex = df_attribution.to_latex()
print(df_attribution_latex)




