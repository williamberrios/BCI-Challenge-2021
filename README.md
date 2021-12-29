# BCI - Challenge 2021
This repository contains the 1st place solution to the BCI-Challenge 2021



## Structure 


```
├── Code
│   ├── 01.EDA.ipynb
│   ├── 02.GenerateDataset.ipynb
│   ├── 03.Final_Ensembling.ipynb
│   ├── featureset_1
│   │   ├── 04_2 Model 2 -Behavior Model.ipynb
│   │   └── generate_feature_v1.py
│   └── featureset_2
│       ├── 04_1 Model 1 - New Clients Model-Boosting-Target-log.ipynb
│       ├── 04_1 Model 1 - New Clients Model-Boosting-Target-sqrt.ipynb
│       ├── 04_3 Model 3 - Future Model-Boosting-Target-sqrt.ipynb
│       └── generate_feature_v2.py
├── Data
│   ├── Data desafío BCI Challenge
│   │   ├── Diccionario BCI Challenge 2021 V6.xlsx
│   │   ├── ejemplo_submission.csv
│   │   ├── README.md
│   │   ├── test_data.csv
│   │   └── train_data.csv
│   └── DatasetsGenerated
│       ├── features_v1
│       │   ├── features.parquet
│       │   └── target_features.parquet
│       ├── features_v2
│       │   ├── features_list
│       │   │   ├── features_v2_level0_client_and_category_vars.txt
│       │   │   ├── features_v2_level0.txt
│       │   │   ├── features_v2_level1_num_by_cat_mes.txt
│       │   │   ├── features_v2_level1_num_by_cat.txt
│       │   │   └── features_v2_level1_var_aggs.txt
│       │   ├── features_v2_level0_client_and_category_vars.parquet
│       │   ├── features_v2_level0.parquet
│       │   ├── features_v2_level1_num_by_cat_mes.parquet
│       │   ├── features_v2_level1_num_by_cat.parquet
│       │   └── features_v2_level1_var_aggs.parquet
│       ├── folds_behavior_clients_model.parquet.gzip
│       └── folds_new_clients_model.parquet.gzip
├── SavedModels
│   ├── model_behavior
│   │   └── lightgbm_v0_behavior_mes_202004
│   │       ├── fold_0
│   │       │   └── model.pkl
│   │       ├── fold_1
│   │       │   └── model.pkl
│   │       ├── fold_2
│   │       │   └── model.pkl
│   │       ├── fold_3
│   │       │   └── model.pkl
│   │       ├── fold_4
│   │       │   └── model.pkl
│   │       ├── oof.csv
│   │       └── part2_test.csv
│   ├── model_future
│   │   ├── catboost_featureset2_model_future_target_sqrt_after_202004
│   │   │   ├── fold_0
│   │   │   │   └── model.pkl
│   │   │   ├── fold_1
│   │   │   │   └── model.pkl
│   │   │   ├── fold_2
│   │   │   │   └── model.pkl
│   │   │   ├── fold_3
│   │   │   │   └── model.pkl
│   │   │   ├── fold_4
│   │   │   │   └── model.pkl
│   │   │   ├── oof.csv
│   │   │   └── part3_test.csv
│   │   └── ligthgbm_featureset2_model_future_target_sqrt_after_202004
│   │       ├── fold_0
│   │       │   └── model.pkl
│   │       ├── fold_1
│   │       │   └── model.pkl
│   │       ├── fold_2
│   │       │   └── model.pkl
│   │       ├── fold_3
│   │       │   └── model.pkl
│   │       ├── fold_4
│   │       │   └── model.pkl
│   │       ├── oof.csv
│   │       └── part3_test.csv
│   └── model_past
│       ├── catboost_featureset2_model_past_target_sqrt
│       │   ├── fold_0
│       │   │   └── model.pkl
│       │   ├── fold_1
│       │   │   └── model.pkl
│       │   ├── fold_2
│       │   │   └── model.pkl
│       │   ├── fold_3
│       │   │   └── model.pkl
│       │   ├── fold_4
│       │   │   └── model.pkl
│       │   ├── oof.csv
│       │   └── part1_test.csv
│       ├── ligthgbm_featureset2_model_past_target_log
│       │   ├── fold_0
│       │   │   └── model.pkl
│       │   ├── fold_1
│       │   │   └── model.pkl
│       │   ├── fold_2
│       │   │   └── model.pkl
│       │   ├── fold_3
│       │   │   └── model.pkl
│       │   ├── fold_4
│       │   │   └── model.pkl
│       │   ├── oof.csv
│       │   └── part1_test.csv
│       └── ligthgbm_featureset2_model_past_target_sqrt
│           ├── fold_0
│           │   └── model.pkl
│           ├── fold_1
│           │   └── model.pkl
│           ├── fold_2
│           │   └── model.pkl
│           ├── fold_3
│           │   └── model.pkl
│           ├── fold_4
│           │   └── model.pkl
│           ├── oof.csv
│           └── part1_test.csv
├── src
│   ├── modeling
│   │   ├── __pycache__
│   │   │   └── regression.cpython-38.pyc
│   │   └── regression.py
│   └── utils
│       ├── encoding.py
│       ├── feature_engineer.py
│       ├── __pycache__
│       │   ├── encoding.cpython-38.pyc
│       │   ├── feature_engineer.cpython-38.pyc
│       │   └── utils.cpython-38.pyc
│       └── utils.py
├── resources
│   ├── Diccionario_Datos.ipynb
│   ├── Diccionario de Variables.csv
│   ├── Documentación_BCI_2021 - William Berrios.pdf
│   └── Presentacion BCI - Entregable.pdf
├── README.md
├── requirements.txt
└── Submission
    └── final_ensemble_submission.csv
```


## Running


+ Install Requirements
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
5. Run models from the 3 components:
    + Model in the Past:
        + Code/featureset_2/04_1 Model 1 - New Clients Model-Boosting-Target-log.ipynb
        + Code/featureset_2/04_1 Model 1 - New Clients Model-Boosting-Target-sqrt.ipynb
    + Model Behavior:
        + Code/featureset_1/04_2 Model 2 -Behavior Model.ipynb
    + Model New Clientes
        + Code/featureset_2/04_3 Model 3 - Future Model-Boosting-Target-sqrt.ipynb
6. Run Final Ensemble:
    + Code/03.Final_Ensembling.ipynb


## Annexes:

The presentation,report and dictionary of variables can be found at [resources](https://github.com/williamberrios/BCI-Challenge-2021/tree/master/resources)


## Author:
+ [William Berrios](https://williamberrios.github.io/)

