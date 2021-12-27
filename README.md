# BCI - Challenge 2021
This repository contains the 1st place solution to the BCI-Challenge 2021



## Running



1. Install Requirements
```
pip install -r requirements.txt
conda install -c conda-forge shap
```
1. First Download the Datasets from the competition in the Data/Data desafío BCI Challenge folder
2. Run Code/01.EDA.ipynb for running the exploratory data analysis
3. Run Code/02.GenerateDataset.ipynb in order to generate the kfold validation datasets
    + folds_behavior_clients_model.parquet.gzip
    + folds_new_clients_model.parquet.gzip
4. Generate Feature Engineering (Featureset 1&2) 
```
# Featureset 1:
python Code/featureset_1/generate_feature_v1.py
```
```
# Featureset 2:
python Code/featureset_2/generate_feature_v2.py
```
5. Run models from the 3 componenents:
    + Model in the Past:
        + Code/featureset_2/04_1 Model 1 - New Clients Model-Boosting-Target-log.ipynb
        + Code/featureset_2/04_1 Model 1 - New Clients Model-Boosting-Target-sqrt.ipynb
    + Model Behavior:
        + Code/featureset_1/04_2 Model 2 -Behavior Model.ipynb
    + Model New Clientes
        + Code/featureset_2/04_3 Model 3 - Future Model-Boosting-Target-sqrt.ipynb
6. Run Final Ensemble:
    + Code/03.Final_Ensembling.ipynb
