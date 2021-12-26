# + endofcell="--"
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

def diff_date_int(df,col1,col2):
    '''
    Function to take diff in months of 2 columns
    date Format: 'YYYYMM'
    Input
    1. df   : Input dataframe
    2. col1 : Column with time reference (YYMM as int)
    3. col2 : Column time that changes for agregation (YYMM as int)
    Output:
    Return a new column with diff in months
    '''
    df['month_diff'] =  (df[col1]//100 -df[col2]//100)*12 + (df[col1]%100 - df[col2]%100)


def apply_agg(df,var_group=[],agg_dict=[],descripcion='',on_merge = False):
    '''
    Function for calculating agregations and simple statistics
    Input
    1. df          : Input dataframe
    2. var_group   : Variables for groupby
    3. agg_dict    : Dictionary for agregations
    4. descripcion : Descripcion de las variables (By what variables im doing the group by)
    5. on_merge    : True -> return the agg merge with initial dataframe, False -> return only agg
    Output:
    . Dataframe with agregations and new names
    Reference:
    Some agregations : ['max','min','mean,'sum','std','nunique','size','count'] 
    '''
    agg         = df.groupby(var_group).agg(agg_dict).reset_index()
    agg.columns = [x[1]+'_'+x[0]+descripcion if x[0] not in var_group else x[0] for x in agg.columns]  
    if on_merge:
        return df.merge(agg,on=var_group,how = 'left')
    else:
        return agg


# # +
def apply_mean_by_cat(dataframe,var_,var_cols =[],drop_mean_group = True):
    '''
    Part of a function used to calculate mean_diff and mean by a categorical column over a numeric column
    Input:
    1. df              : Dataframe as input
    2. var_            : Variable that is used for the cat group by
    3. var_cols        : Numeric columns for doing the aggregations
    4. drop_mean_group : Flag that indicates if the mean of the numeric column would be kept
    '''
    df = dataframe.copy()
    agg_dict = {}
    for i in var_cols:
        agg_dict[i] = ['mean']
    agg         = df.groupby([var_]).agg(agg_dict).reset_index()
    agg.columns = ['mean_'+x[0] + '_by_' + var_ if x[0] not in [var_] else x[0] for x in agg.columns] 
    df_ = df.merge(agg,on =[var_],how = 'left').copy()
    del df
    gc.collect()
    for i in var_cols:
        print(f"****taking diff mean of {i}")
        df_[f"diff_mean_{i}_by_{var_}"] = df_[f"{i}"] - df_[f"mean_{i}_by_{var_}"]
    del agg
    if drop_mean_group:
        var_drop = [f"mean_{i}_by_{var_}" for i in var_cols]
        return df_.drop(columns = var_drop)
    else: 
        return df_

def apply_agg_by_cat(dataframe,var_,var_cols =[],agg_one ='mean' ,drop_mean_group = True):
    df = dataframe.copy()
    agg_dict = {}
    for i in var_cols:
        agg_dict[i] = [agg_one]
    agg         = df.groupby([var_]).agg(agg_dict).reset_index()
    agg.columns = [f"{agg_one}_"+x[0] + '_by_' + var_ if x[0] not in [var_] else x[0] for x in agg.columns] 
    df_ = df.merge(agg,on =[var_],how = 'left').copy()
    del df
    gc.collect()
    for i in var_cols:
        print(f"***taking diff {agg_one} of {i}***")
        df_[f"diff_{agg_one}_{i}_by_{var_}"] = df_[f"{i}"] - df_[f"{agg_one}_{i}_by_{var_}"]
    del agg
    if drop_mean_group:
        var_drop = [f"{agg_one}_{i}_by_{var_}" for i in var_cols]
        return df_.drop(columns = var_drop)
    else: 
        return df_
    
def multiple_apply_mean_by_cat(df,var_for_group = [],var_cols =[],drop_mean_group = True):
    '''
    Function for getting the agg diff mean by cat columns
    '''
    df_ = df.copy()
    for var_ in var_for_group:
        print(f"--Var for grouping: {var_}--")
        df_ = apply_mean_by_cat(df_,var_,var_cols = var_cols,drop_mean_group = drop_mean_group)
    return df_

def multiple_apply_agg_by_cat(df,var_for_group = [],var_cols =[],agg_one = 'mean',drop_mean_group = True):
    '''
    Function for getting the agg diff {agg} by cat columns
    '''
    df_ = df.copy()
    for var_ in var_for_group:
        print(f"***Var for grouping: {var_}***")
        df_ = apply_agg_by_cat(df_,var_,var_cols = var_cols,agg_one =agg_one,drop_mean_group = drop_mean_group)
    return df_


# -

def get_historic_variables(df,group_vars,lag_time_list,rang_periodos,agg_dict_origi,col_periodo,desc= '',min_window = 0,sort_by = None):
    '''
    Function to calculate simple statistics in different lags time
    Input
    1. df            : Input dataframe
    2. group_vars    : Variables for groupby
    3. lag_time_list : List for apply time agregations
    4. rang_periodos : Periods for generating agregations
    5. agg_dict      : Dictionary for agregations
    6. col_periodo   : Name of the column that has the time (format -> YYYYMMM as int)
    Output
    Dataframe with agregations in different times in a concatenate way
    '''
    gc.collect()
    agg_dict = agg_dict_origi.copy()
    if 'month_diff' in agg_dict.keys():
        pass
    else:
        agg_dict['month_diff'] = ['min']
    df_union = None
    for periodo in tqdm(rang_periodos):
        #print(f"{'*'*5} {periodo} {'*'*5}")
        for enum,lag_time in enumerate(lag_time_list):
            #print(f"{lag_time} rango")
            df['codmes_ref'] = periodo
            diff_date_int(df,'codmes_ref',col_periodo)  
            df_ = df.query(f"month_diff >={min_window} and month_diff<{lag_time}").copy()
            if sort_by is not None:
                df_ = df_.sort_values(by =sort_by,ascending = True)
            df_ = apply_agg(df_,var_group = group_vars,agg_dict=agg_dict,descripcion=f"_{lag_time}M{desc}").reset_index(drop = True)
            df_ = df_[df_[f"min_month_diff_{lag_time}M{desc}"]==min_window].reset_index(drop = True).drop(columns = f"min_month_diff_{lag_time}M{desc}")
            if enum == 0:
                df_lag = df_.copy()
            else:
                df_lag = pd.concat([df_lag,df_.drop(columns = group_vars)],axis = 1)
            del df_
            gc.collect()
        df_lag['codmes_ref'] = periodo
        df_union = pd.concat([df_union,df_lag])
        del df_lag
        gc.collect()
    df.drop(columns = ['codmes_ref'],inplace = True)
        
  
    return df_union
# --
