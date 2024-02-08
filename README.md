# Thesis: MTA Ensembles
This repository contains the code used for the research paper Increased Performance in a Shorter Timeframe: An Ensemble Approach in Multi-Touch Conversion Attribution. The scripts are divided into several categories: Data Description, Data Pre-Processing, Models, and Results. Below, we provide a description of the code used and how the results may be replicated.

## Repository Structure
The repository contains the following folders.
- `Data exploration.py` contains the code used for the descriptive statistics of the raw Criteo data set.
-  `Data Pre-processing` contains the code used for the pre-processing steps described in the paper. There are two files, `Preprocessing - LR & SVM.py` and `Preprocessing - LSTM`. The distinction is that the LSTM file makes use of padding, whereas the LR and SVM file does not.
-  `Models` contains the code for the ensembles and cascades that are proposed in the the research paper. For the logistic regression and support vector machine, there are variants found using sequence aggregation. These variants are indicated with a seq. suffix.
-  `Results` contains the code used to generate the plots and tables used in the paper.

