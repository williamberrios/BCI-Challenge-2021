# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import torch
import random


def seed_everything(seed=42):
    '''
    
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fillna_columns_var(train_df,test_df,fillna = None):
    if fillna is not None:
        print('Filling NA')
        for col in train_df.columns:
            if 'VAR' in col:
                train_df[col] =  train_df[col].fillna(0)
                test_df[col]  =  test_df[col].fillna(0)
    return train_df,test_df


# +
# Get Training & Test Datasets
def get_train_data(dataset_name,DATA_PATH = '../Data'):
     return pd.read_parquet(os.path.join(DATA_PATH,'DatasetsGenerated',dataset_name))
    
def get_test_data(segment,DATA_PATH = '../Data'):
    if segment == 'groupkfold_client':
        test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
        test_df  = test_df[test_df['mes']<=202104].reset_index(drop = True)
        return test_df
    
    elif segment == 'all_future':
        # We might use the previous target and indicator of new client
        test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
        test_df  = test_df[test_df['mes']>=202105].reset_index(drop = True)
        return test_df
    
    elif segment == 'behavior_future':
        # We only return clients in test that have information in train
        train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))[['id','mes','target_mes']]
        test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
        test_df  = test_df[(test_df['mes']>=202105)&(test_df['id'].isin(train_df.id.unique()))].reset_index(drop = True)
        del train_df
        return test_df
    elif segment == 'not_behavoir_future':
        train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))[['id','mes','target_mes']]
        test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
        test_df  = test_df[(test_df['mes']>=202105)&(~test_df['id'].isin(train_df.id.unique()))].reset_index(drop = True)
        del train_df
        return test_df
    else:
        raise Exception('This segment does not exist')

def get_reference_datasets(DATA_PATH ='../Data'):
    train = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))[['id','mes','target_mes']]
    test  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))[['id','mes']]
    return train,test
def post_processing(test_df,DATA_PATH = '../Data'):
    '''
    Postprocessing for previos values of target_mes equal 0 in train data:
    '''
    train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))[['id','mes','target_mes']]
    tmp   = train_df.groupby(['id']).agg({'target_mes':'mean'}).reset_index()
    ids_0 = list((set(tmp[tmp['target_mes']==0].id)&set(test_df.id.unique())))
    test_df.loc[test_df['id'].isin(ids_0),'preds'] = 0
    return test_df
# -


