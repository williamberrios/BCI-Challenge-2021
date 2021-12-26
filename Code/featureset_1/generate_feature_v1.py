# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import datetime
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.feature_engineer import get_historic_variables,apply_agg,multiple_apply_agg_by_cat

DATA_PATH = '../../Data'
FILL_NA  = 0


""
def add_months(sourcedate, months,output_type = 'int'):
    '''
    Inputs:
    1. sourcedate : Source date in int or string
    2. months     : Number of months in 
    
    Output:
    . new source date

    '''
    sourcedate = datetime.datetime.strptime(str(sourcedate), '%Y%m')
    month      = sourcedate.month - 1 + months
    year       = sourcedate.year + month // 12
    month      = month % 12 + 1
    output     = convert_type(datetime.date(year, month,1).strftime("%Y%m"),output_type)
    return output


def convert_type(x,type_var):
    if type_var == 'int':
        return int(x)
    if type_var == 'float':
        return float(x)
    if type_var == 'str':
        return str(x) 
    
def check_for_duplicate_cols(df):
    a = []
    for col in df.columns:
        if (col.endswith("_x"))|(col.endswith("_y")):
            print(f'Duplicate col for: {col}')
            a.append(col)
    return len(a)


# +
def agg_count_0(x):
    return (np.array(x)==0).sum()


def linear_trend_slope(x):
    x = [i for i in x if str(i) != 'nan']
    if len(x)==0:
        return None
    from scipy.stats import linregress
    linReg = linregress(range(len(x)), x)

    return linReg.slope

def count_nans(df):
    df['count_nan'] = df.isnull().sum(axis=1)
    return df


# +
def apply_split_tip_segmento(row):
    # Columns PROD1,PROD2,PROD3,PROD4
    if   row['tipo_seg'] == 'MULTIPROD':
        return 1,1,1,1
    elif row['tipo_seg'] == 'NO PROD1':
        return 0,1,1,1
    elif row['tipo_seg'] == 'NO PROD2':
        return 1,0,1,1
    elif row['tipo_seg'] == 'NO PROD3':
        return 1,1,0,1
    elif row['tipo_seg'] == 'NO PROD4':
        return 1,1,1,0
    elif row['tipo_seg'] == 'PROD1':
        return 1,0,0,0
    elif row['tipo_seg'] == 'PROD2':
        return 0,1,0,0
    elif row['tipo_seg'] == 'PROD3':
        return 0,0,1,0
    elif row['tipo_seg'] == 'PROD4':
        return 0,0,0,1
    elif row['tipo_seg'] == 'PROD1/PROD2':
        return 1,1,0,0
    elif row['tipo_seg'] == 'PROD1/PROD3':
        return 1,0,1,0
    elif row['tipo_seg'] == 'PROD1/PROD4':
        return 1,0,0,1
    elif row['tipo_seg'] == 'PROD2/PROD4':
        return 0,1,0,1
    elif row['tipo_seg'] == 'PROD3/PROD2':
        return 0,1,1,0
    elif row['tipo_seg'] == 'PROD3/PROD4':
        return 0,0,1,1
    elif row['tipo_seg'] == 'SINPROD':
        return 0,0,0,0
    else:
        raise Exception('This tip segment does not exist')
        

def preprocess_tipsegmento(df):
    df[[f'tipo_seg_prod_{i}' for i in range(0,4)]] = df.apply(lambda x:apply_split_tip_segmento(x),
                                                                              axis = 1,
                                                                              result_type = 'expand')
    return df




# +
def preprocess_fillna_var(df,fill_value):
    cols_var = [i for i in df.columns if 'VAR' in i]
    if fill_value is not None:
        print(f'**** Filling NA VAR {fill_value} *******')
        for col in cols_var:
            df[col] = df[col].fillna(fill_value) 
    return df

def flag_covid(df):
    print('****** Adding Flag Covid Variable *******')
    # Time crisis covid
    df['FLG_COVID'] = 0
    df.loc[df['mes'].isin([202004,202005,202006,202007,202008]),'FLG_COVID'] = 1
    return df



# +
def ratio_sum_trx(df,cols = []):
    print('****** Adding Ratio(sum/trx) columns *******')
    for col in cols:
        try:
            df[f'ratio_{col}_sum_trx'] = df[f'{col}_sum']/df[f'{col}_trx']
            df[f'ratio_{col}_sum_trx'] = df[f'ratio_{col}_sum_trx'].replace([np.inf, -np.inf],np.nan)
            df[f'ratio_{col}_sum_trx'].fillna(0,inplace = True)
        except:
            print(f'Not worked for {col}')
    return df

def ratio_sum_prom(df,cols = []):
    print('****** Adding Ratio(sum/prom) columns *******')
    for col in cols:
        try:
            df[f'ratio_{col}_sum_prom'] = df[f'{col}_sum']/df[f'{col}_prom']
            df[f'ratio_{col}_sum_prom'] = df[f'ratio_{col}_sum_prom'].replace([np.inf, -np.inf],np.nan)
            df[f'ratio_{col}_sum_prom'].fillna(0,inplace = True)
        except:
            print(f'Not worked for {col}')
    return df


# +
def agg_numeric_features(dataframe,
                         lag_time_list = [3,6,9,12,18,34],
                         agg_dict      = {},
                         descripcion   = '',
                         fillna_value  = None):
    '''
    Generación de Variables con respecto al cliente ('id')
    '''
    print(f'****** Adding agg_numeric_features for: {agg_dict.keys()} *******')
    train  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv')).drop(columns = ['target_mes'])
    test   = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    tablon = pd.concat([train,test],axis = 0).reset_index(drop = True)
    total_shape  = dataframe.shape[0] 
    
    if fillna_value is not None:
        tablon = tablon.fillna(fillna_value)
        
    group_vars    = ['id']
    min_window    = 0
    rang_periodos = [add_months(201812,i) for i in range(0,34)]
    col_periodo = 'mes'
    agg  = get_historic_variables(tablon,group_vars,lag_time_list,rang_periodos,agg_dict,col_periodo,descripcion,min_window).rename(columns = {'codmes_ref':'mes'})
    dataframe = dataframe.merge(agg,on = ['id','mes'],how = 'left').reset_index(drop = True)
    del agg,train,test,tablon
    assert dataframe.shape[0] == total_shape
    return dataframe      


def flatten(t):
    return [item for sublist in t for item in sublist]

def generate_agg_cat_numeric(dataframe,
                             var_groups      = [],
                             var_agg         = [],
                             aggregations    = [],
                             drop_mean_group = True):
    print(f'****** Adding generate_agg_cat_numeric for: {var_agg} by {var_groups} *******')
    total_shape = dataframe.shape[0]
    use_cols = list(set(flatten(var_groups) + var_agg + ['id','mes']))
    df = dataframe[use_cols].reset_index(drop = True).copy()
    cols = []
    for var in var_groups:
        if len(var)>1:
            print('Combine Feat: ',var)
            df['_'.join(var)] = df[var].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            cols.append('_'.join(var))
        else:
            print('Feat: ',var)
            cols.append(var[0])
            
    for agg in aggregations:
        df = multiple_apply_agg_by_cat(df,cols,var_agg,agg,drop_mean_group = drop_mean_group)
        
    for var in var_groups:
        if len(var)>1:
            print('Drop:', '_'.join(var))
            df.drop(columns = ['_'.join(var)],inplace = True) 
    print('Drop: ',list(set(use_cols)-set(['id','mes'])))
    df.drop(columns = list(set(use_cols)-set(['id','mes'])),inplace = True)
    
    dataframe = dataframe.merge(df,on = ['id','mes'],how = 'left').reset_index(drop = True) 
    del df,cols
    assert dataframe.shape[0] == total_shape
    return dataframe

def combine_tipo_seg_ban(df):
    df['tipo_seg_and_tipo_ban'] = df['tipo_seg'].astype(str) + '_' + df['tipo_ban'].astype(str)
    df['tipo_ban_group'] = df['tipo_ban'].replace({'BAN1':'GROUP_BAN_2','BAN2':'GROUP_BAN_1','BAN3':'GROUP_BAN_1','BAN4':'GROUP_BAN_2'})
    return df

def generate_variacion_agg(df,
                           aggs = ['mean','max','min'],
                           variables = ['VAR1_sum','VAR2_sum'],
                           tpast = [34]):
    for var in variables:
        for agg in aggs:
                for past in tpast:
                    if past > 0:
                        try:
                            df[f'variacion_{var}_{0}M_sobre_{past}M'] = 100*(df[var]-df[f'{agg}_{var}_{past}M'])/df[f'{agg}_{var}_{past}M']
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([np.inf], 99999, inplace=True)
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([-np.inf], -99999, inplace=True) 
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([-np.nan], 0, inplace=True) 
                        except:
                            print(f'Failed for var:{var}, agg:{agg}, init:{init}, past:{past}')
                            pass
            
    return df


# +
def number_incrementos(x):
    y = x[1:]
    return np.sum(np.array(y)>np.array(x[0:-1]))

def ratio_incrementos(x):
    y = x[1:]
    return np.sum(np.array(y)>np.array(x[0:-1]))/len(x)

def number_decrementos(x):
    y = x[1:]
    return np.sum(np.array(y)<np.array(x[0:-1]))

def ratio_decrementos(x):
    y = x[1:]
    return np.sum(np.array(y)<np.array(x[0:-1]))/len(x)

##################################
# ### Aggregations that need time order
#
def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

def get_last_time_equal_0(x):
    return len(x)-np.max((np.array(x)==0)*np.arange(len(x)))
    

def generate_variables_target(dataframe,min_window = 5,lag_time_list = [12,18,100],agg_dict = {'target_mes':['mean','std','min','max','skew',pd.DataFrame.kurt,agg_count_0]}):
    print('******** Generate custom aggregations for past target ************')
    '''
    Generación de las variables de target para el modelo de behavior
    '''
    assert min_window >=1
    assert min(lag_time_list)>=5
    total_shape = dataframe.shape[0]
    
    
    train  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    tablon = pd.concat([train,test],axis = 0).reset_index(drop = True)[['id','mes','target_mes']]
    group_vars    = ['id']
    rang_periodos = [add_months(201812,i) for i in range(0,34)]
    col_periodo   = 'mes'
    desc          = f'_variables_target_window{min_window}'
    agg           = get_historic_variables(tablon,group_vars,lag_time_list,rang_periodos,agg_dict,col_periodo,desc,min_window).rename(columns = {'codmes_ref':'mes'})
    dataframe     = dataframe.merge(agg,how = 'left',on = ['id','mes'])
    del tablon,train,test,agg
    assert dataframe.shape[0] == total_shape
    return dataframe

def generate_variables_target_trends(dataframe,min_window = 5,lag_time_list = [12,18,100]):
    '''
    Generación de las variables de target para el modelo de behavior
    '''
    assert min_window >=1
    assert min(lag_time_list)>=5
    total_shape = dataframe.shape[0]
    train  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    tablon = pd.concat([train,test],axis = 0).reset_index(drop = True)[['id','mes','target_mes']]
    group_vars    = ['id']
    rang_periodos = [add_months(201812,i) for i in range(0,34)]
    # Aggregations of Trends
    agg_dict      = {'target_mes':[linear_trend_slope,last_location_of_minimum,first_location_of_minimum,get_last_time_equal_0]}
    col_periodo   = 'mes'
    desc          = f'_variables_target_window{min_window}'
    agg           = get_historic_variables(tablon,group_vars,lag_time_list,rang_periodos,agg_dict,col_periodo,desc,min_window,sort_by = ['id','mes']).rename(columns = {'codmes_ref':'mes'})
    dataframe     = dataframe.merge(agg,how = 'left',on = ['id','mes'])
    del tablon,agg,train,test
    assert dataframe.shape[0] == total_shape
    return dataframe


""
### Version 1
def pipeline_preprocessing(df):
    # Variables use:
    ratio_cols = ['VAR1','VAR3','VAR25','VAR4','VAR24','VAR23']
    # Before filling values, count nan:
    df = count_nans(df)
    # Filling NA VAR features
    df = preprocess_fillna_var(df,FILL_NA)
    # Preprocess tip_segmento
    df = preprocess_tipsegmento(df)
    # Combine tip_segmento & tip_ban:
    df = combine_tipo_seg_ban(df)
    # Generate ratio of Vars:
    df = ratio_sum_trx(df,ratio_cols)
    df = ratio_sum_prom(df,ratio_cols)    
    # Generate Flag Covid Variables:
    df = flag_covid(df)
    
    # Generate Aggregation of numeric features in time
    df = agg_numeric_features(df,
                              lag_time_list = [3,6,9,12,18,34],
                              agg_dict = {'VAR1_sum' : ['mean','min','max'],
                                          'VAR4_sum' : ['mean','min','max'],
                                          'VAR23_sum': ['mean','min','max'],
                                          'VAR5_sum' : ['mean','min','max'],
                                          'VAR3_sum' : ['mean','min','max'],
                                          'VAR25_sum': ['mean','min','max'],
                                          'VAR24_sum': ['mean','min','max'],
                                          'VAR16_sum': ['mean','min','max'],
                                          'VAR12_sum': ['mean','min','max'],
                                         },
                              descripcion = '',
                             fillna_value = 0) 
    
    return df

def pipeline_target_preprocessing(df):
    df = generate_variables_target(df,
                                   min_window    = 5,
                                   lag_time_list = [12,18,100],
                                   agg_dict      = {'target_mes':['mean','std','min','max','skew',
                                                                  pd.DataFrame.kurt,agg_count_0]})
    
    df= generate_variables_target_trends(df,
                                          min_window = 5,
                                          lag_time_list = [12,18])
    
    return df



###############################################################################
# # Version 1:

"""
## Features No Target
"""



def features_not_target():
    train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv')).drop(columns = ['target_mes'])
    test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    # Total Dataset
    total    = pd.concat([train_df,test_df[train_df.columns]]).reset_index(drop = True)
    id_df    = total[['id','mes']].copy()
    initial_cols = total.drop(columns = ['id','mes']).columns
    # GENERAL VARIABLES
    FILL_NA = 0
    total = pipeline_preprocessing(total)
    id_df = id_df.merge(total,on = ['mes','id'],how = 'left').reset_index(drop = True)
    assert check_for_duplicate_cols(id_df) == 0
    # Merge with only ids
    id_df.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v1/features.parquet'),compression = 'gzip',index = False)

###############################################################################
# ## Features Target

def features_target():
    train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    test_df['target_mes'] = np.nan
    # Total Dataset
    total    = pd.concat([train_df,test_df[train_df.columns]]).reset_index(drop = True)
    init_cols = total.drop(columns = ['id','mes']).columns
    id_df    = total[['id','mes']].copy()
    agg_target = pipeline_target_preprocessing(total).drop(columns = init_cols)
    id_df    = id_df.merge(agg_target,on = ['id','mes'],how = 'left')
    assert check_for_duplicate_cols(id_df) == 0
    id_df.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v1/target_features.parquet'),compression = 'gzip',index = False)


""
if __name__ == '__main__':
    features_not_target()
    features_target()
