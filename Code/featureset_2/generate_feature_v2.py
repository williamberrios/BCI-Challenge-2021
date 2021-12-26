# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np
import cudf
import dask_cudf
import os
import sys
from tqdm import tqdm
import gc
import itertools
import datetime
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.feature_engineer import get_historic_variables,apply_agg
DATA_PATH   = '../../Data'
SAVE_FOLDER = '../../Data/DatasetsGenerated/features_v2'
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

def agg_count_diff_0(x):
    return len(x)-(np.array(x)==0).sum()

def count_above_mean(x):
    '''
    Count time the variables is above the mean
    '''
    m = np.mean(x)
    return np.where(x > m)[0].size

def ratio_above_mean(x):
    '''
    Ratio of time the variables is above the mean/size
    '''
    m = np.mean(x)
    return np.where(x > m)[0].size/x.size
    
def count_below_mean(x):
    '''
    Count time the variables is below the mean
    '''
    m = np.mean(x)
    return np.where(x < m)[0].size

def ratio_below_mean(x):
    '''
    Ratio of time the variables is below the mean/size
    '''
    m = np.mean(x)
    return np.where(x < m)[0].size/x.size

def linear_trend_slope(x):
    x = [i for i in x if str(i) != 'nan']
    if len(x)==0:
        return None
    from scipy.stats import linregress
    linReg = linregress(range(len(x)), x)

    return linReg.slope

def linear_trend_r2(x):
    x = [i for i in x if str(i) != 'nan']
    if len(x)==0:
        return None
    from scipy.stats import linregress
    linReg = linregress(range(len(x)), x)
    return linReg.rvalue**2

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
    for col in tqdm(cols):
        try:
            df[f'ratio_{col}_sum_trx'] = df[f'{col}_sum']/df[f'{col}_trx']
            df[f'ratio_{col}_sum_trx'] = df[f'ratio_{col}_sum_trx'].replace([np.inf, -np.inf],np.nan)
            df[f'ratio_{col}_sum_trx'].fillna(0,inplace = True)
        except:
            print(f'Not worked for {col}')
    return df

def ratio_sum_prom(df,cols = []):
    print('****** Adding Ratio(sum/prom) columns *******')
    for col in tqdm(cols):
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
                         fillna_value  = None,
                         sort_by = None):
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
    agg  = get_historic_variables(tablon,group_vars,lag_time_list,rang_periodos,agg_dict,col_periodo,descripcion,min_window,sort_by = sort_by).rename(columns = {'codmes_ref':'mes'})
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

def generate_variacion_agg(dataframe,
                           aggs = ['mean','max','min'],
                           variables = ['VAR1_sum','VAR2_sum'],
                           tpast = [34]):
    
    # Get neccesarry columns:
    vars_list = []
    for var in tqdm(variables):
        for agg in aggs:
                for past in tpast:
                    vars_list.append(f'{agg}_{var}_{past}M')
    
    df = dataframe[['id','mes']+vars_list + variables].copy()
    for var in tqdm(variables):
        for agg in aggs:
                for past in tpast:
                    if past > 0:
                        try:
                            df[f'variacion_{var}_{0}M_sobre_{past}M'] = 100*(df[var]-df[f'{agg}_{var}_{past}M'])/df[f'{agg}_{var}_{past}M']
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([np.inf], 99999, inplace=True)
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([-np.inf], -99999, inplace=True) 
                            df[f'variacion_{var}_{0}M_sobre_{past}M'].replace([-np.nan], 0, inplace=True) 
                        except:
                            print(f'Failed for var:{var}, agg:{agg}, past:{past}')
                            pass
            
    return df.drop(columns = vars_list + variables)
""
def generate_interactions_num(df):
    print(" ********** Generate Interactions ************")
    time.sleep(0.5)
    for i in tqdm(range(1,31)): 
        for j in range(1,31):
            if i==j:
                pass
            else:
                try:
                    df[f'ratio_VAR{i}_sum_VAR{j}_sum'] = df[f'VAR{i}_sum']/(df[f'VAR{j}_sum']+1)
                except:
                    pass
                try:
                    df[f'ratio_VAR{i}_prom_VAR{j}_prom'] = df[f'VAR{i}_prom']/(df[f'VAR{j}_prom']+1)
                except:
                    pass
                try:
                    df[f'ratio_VAR{i}_trx_VAR{j}_trx'] = df[f'VAR{i}_trx']/(df[f'VAR{j}_trx']+1)
                except:
                    pass
                try:
                    df[f'ratio_VAR{i}_sum_VAR{j}_trx'] = df[f'VAR{i}_sum']/(df[f'VAR{j}_trx']+1)
                except:
                    pass
                
    return df


""
def agg_numeric_by_cat(df,
                       num_cols,
                       cat_cols = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli'],
                       aggs     = ['mean','max','std']):
    
    
    id_df = df[['id','mes']+num_cols+cat_cols].copy()
    ori_shape = len(id_df)
    print(f"Aggregation on cat cols: {cat_cols}")
    agg_dict = {}
    for num_col in num_cols:
        agg_dict[num_col] = aggs
    for cat_col in tqdm(cat_cols):
        agg = apply_agg(id_df,var_group=[cat_col],agg_dict= agg_dict,descripcion=f'_groupby_{cat_col}',on_merge = False)
        id_df = id_df.merge(agg.fillna(0),on = [cat_col],how = 'left')
        del agg
        gc.collect()
        for agg in aggs:
            for num_col in num_cols:
                if agg == 'mean':
                    id_df [f'diff_{agg}_{num_col}_groupby_{cat_col}'] = id_df[num_col] - id_df [f'{agg}_{num_col}_groupby_{cat_col}']
                    id_df [f'ratio_{agg}_{num_col}_groupby_{cat_col}'] = id_df[num_col]/id_df [f'{agg}_{num_col}_groupby_{cat_col}']                                                                               
                else:
                    id_df [f'ratio_{agg}_{num_col}_groupby_{cat_col}'] = id_df[num_col]/id_df [f'{agg}_{num_col}_groupby_{cat_col}']                                                                               
    

    assert len(id_df) == ori_shape
    return id_df.drop(columns = num_cols + cat_cols)

def agg_numeric_by_cat_and_period(df,
                                  num_cols,
                                  cat_cols = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli'],
                                  aggs     = ['mean','max','std']):
    
    
    id_df = df[['id','mes']+num_cols+cat_cols].copy()
    ori_shape = len(id_df)
    print(f"Aggregation on cat cols and mes: {cat_cols}")
    
    agg_dict = {}
    for num_col in num_cols:
        agg_dict[num_col] = aggs
    for cat_col in tqdm(cat_cols):
        agg = apply_agg(id_df,var_group=[cat_col,'mes'],agg_dict= agg_dict,descripcion=f'_groupby_{cat_col}_and_mes',on_merge = False)
        id_df = id_df.merge(agg.fillna(0),on = [cat_col,'mes'],how = 'left')
        del agg
        gc.collect()
        for agg in aggs:
            for num_col in num_cols:
                if agg == 'mean':
                    id_df [f'diff_{agg}_{num_col}_groupby_{cat_col}_and_mes'] = id_df[num_col] - id_df [f'{agg}_{num_col}_groupby_{cat_col}_and_mes']
                    id_df [f'ratio_{agg}_{num_col}_groupby_{cat_col}_and_mes'] = id_df[num_col]/id_df [f'{agg}_{num_col}_groupby_{cat_col}_and_mes']                                                                               
                else:
                    id_df [f'ratio_{agg}_{num_col}_groupby_{cat_col}_and_mes'] = id_df[num_col]/id_df [f'{agg}_{num_col}_groupby_{cat_col}_and_mes']                                                                               
    
    assert len(id_df) == ori_shape
    return id_df.drop(columns = num_cols + cat_cols)                        
                                                                                          
def get_combinations_categoric(x :list,min_len,top_len):
    out = []
    for L in range(0, len(x)+1):
        for subset in itertools.combinations(x, L):
            if (len(subset)>min_len)&(len(subset)<top_len):
                out.append(list(subset))
    return out
def combine_cat_features(df,cat_cols,min_len,top_len):
    time.sleep(0.5)
    print(f"******** FE: Combine Features {cat_cols}********")
    cols_combination = get_combinations_categoric(cat_cols,min_len,top_len)    
    for cols in tqdm(cols_combination):
        for idx,col in enumerate(cols):
            if idx == 0:
                suma    = df[col].astype(str)
                col_name = col
            else:
                suma += '_' + df[col].astype(str)
                col_name += '_'+col
        df[col_name] = suma
    return df


""
### Version 1
def pipeline_preprocessing_level0(df):
    origin_cols = [i for i in df.columns.to_list() if 'VAR' in i]
    ratio_cols = [f'VAR{i}' for i in range(1,31)]
    df = count_nans(df)
    # Variables use:
    origin_cols = [i for i in df.columns.to_list() if 'VAR' in i]
    ratio_cols = [f'VAR{i}' for i in range(1,31)]
    # Before filling values, count nan:
    df = count_nans(df)
    # Filling NA VAR features
    df = preprocess_fillna_var(df,FILL_NA)
    # Preprocess tip_segmento
    print("******* Processing Tip Segmento Decompressing *********")
    df = preprocess_tipsegmento(df)
    # Generate Interactions of features:
    df = generate_interactions_num(df)
    # Generate ratio of Vars:
    df = ratio_sum_trx(df,ratio_cols)
    df = ratio_sum_prom(df,ratio_cols)  
    cat_cols = df.drop(columns = ['id']).select_dtypes('object').columns.to_list() 
    df = combine_cat_features(df,cat_cols,1,5)  
    # Generate Flag Covid Variables:
    df = flag_covid(df)
    # Generate Aggregation of numeric features in time
    agg_dict = {}
    for col in origin_cols:
        agg_dict[col] = ['mean','min','max','std']
    df = agg_numeric_features(df,
                              lag_time_list = [2,3,6,9,12,18,34],
                              agg_dict = agg_dict,
                              descripcion = '',
                              fillna_value = 0,
                              sort_by = None).fillna(0)    
    return df

def pipeline_target_preprocessing(df):
    df = generate_variables_target(df,
                                   min_window    = 5,
                                   lag_time_list = [6,9,12,18,100],
                                   agg_dict      = {'target_mes':['mean','std','min','max','skew',
                                                                  pd.DataFrame.kurt,agg_count_0,
                                                                  count_above_mean]}).fillna(0)
    
    df= generate_variables_target_trends(df,
                                         min_window = 5,
                                         lag_time_list = [6,9,12,18,100])
    
    return df


"""
## 1. Level 0 Features:
"""



def features_level0():
    train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv')).drop(columns = ['target_mes'])
    test_df  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    # Total Dataset
    total    = pd.concat([train_df,test_df[train_df.columns]]).reset_index(drop = True)
    id_df    = total[['id','mes']].copy()
    initial_cols = total.drop(columns = ['id','mes']).columns
    # GENERAL VARIABLES
    total = pipeline_preprocessing_level0(total)
    id_df = id_df.merge(total,on = ['mes','id'],how = 'left').reset_index(drop = True)
    assert check_for_duplicate_cols(id_df) == 0
    # Merge with only ids
    id_df.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level0.parquet'),compression = 'gzip',index = False)


###############################################################################
# ## 2. Level 1 Features agg num by cat

def features_agg_num_by_cat_and_mes_level1():
    _  = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','test_data.csv'))
    origin_cols = [i for i in _.columns.to_list() if 'VAR' in i]
    del _
    
    cat_agg_cols_num = ['tipo_ban', 'tipo_seg', 'categoria', 'tipo_com', 'tipo_cat', 'tipo_cli', 
                         'tipo_ban_tipo_seg', 'tipo_seg_tipo_cli','tipo_ban_tipo_cli']
    
    df = pd.read_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level0.parquet'),
                         columns = ['id','mes'] + origin_cols + cat_agg_cols_num)
    #Generate agg by categoric cols
    
    
    agg_cat = agg_numeric_by_cat(df,
                       origin_cols,
                       cat_cols = cat_agg_cols_num,
                       aggs     = ['mean','max','std'])
    
    agg_cat.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level1_num_by_cat.parquet'))
    del agg_cat
    
    agg_cat_mes = agg_numeric_by_cat_and_period(df,
                                                origin_cols,
                                                cat_cols = cat_agg_cols_num,
                                                aggs     = ['mean','max','std'])

    agg_cat_mes.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level1_num_by_cat_mes.parquet'))
    del agg_cat_mes



###############################################################################
# ## 2. Level 1 Features Variacion of agg

def features_variacion_agg_level1():
    df = pd.read_parquet('../../Data/DatasetsGenerated/features_v2/features_v2_level0.parquet')
    agg_var = generate_variacion_agg(df,
                                aggs = ['mean','max','min','std'],
                                variables = ['VAR11_sum','VAR12_sum','VAR13_sum','VAR15_sum','VAR17_sum','VAR1_prom','VAR1_sum',
                                             'VAR1_trx','VAR23_sum','VAR23_trx','VAR24_sum','VAR24_trx','VAR25_sum','VAR28_trx',
                                             'VAR2_prom','VAR2_sum','VAR2_trx','VAR4_sum','VAR5_sum','VAR5_trx','VAR7_sum']  ,
                                tpast = [2,3,6,9,12]).fillna(0)
    agg_var.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level1_var_aggs.parquet'))
    del agg_var


###############################################################################
# ## 3. Level 0 Variables Clientes

def get_cliente_and_count_categories_variables():
    train_df = pd.read_csv(os.path.join(DATA_PATH,'Data desafío BCI Challenge','train_data.csv')).drop(columns = ['target_mes'])
    test_df  = pd.read_csv(os.path.join('../../Data/Data desafío BCI Challenge','test_data.csv'))
    # Total Dataset
    df    = pd.concat([train_df,test_df[train_df.columns]]).reset_index(drop = True)
    cat_features=  df.select_dtypes('object').drop(columns = ['id']).columns.to_list()
    df = combine_cat_features(df,cat_features,1,5)

    agg_dict = {'month_diff':['min','nunique']}
    for cat in ['categoria','tipo_ban_categoria','categoria_tipo_cat']:
        agg_dict [cat] = ['nunique']

    group_vars = ['id']
    lag_time_list  = [6,12,18,34]
    rang_periodos  = [add_months(201812,i) for i in range(0,34)]
    col_periodo    = 'mes'
    min_window     = 0 

    agg_var = get_historic_variables(df,group_vars,lag_time_list,rang_periodos,agg_dict,col_periodo,desc= '',min_window = 0,sort_by = None).fillna(0).rename(columns = {'codmes_ref':'mes'})
    agg_var.to_parquet(os.path.join(DATA_PATH,'DatasetsGenerated/features_v2/features_v2_level0_client_and_category_vars.parquet'))



""
if __name__ == '__main__':
    features_level0()
    features_agg_num_by_cat_and_mes_level1()
    features_variacion_agg_level1()
    get_cliente_and_count_categories_variables()
