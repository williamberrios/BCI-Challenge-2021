# +
import pandas as pd
import numpy as np
from sklearn import preprocessing

class ClassicEncoding():
    
    def __init__(self,columns,params = None,name_encoding = 'LE'):
        """
        name_encoding : OHE,LE
        """
        self.name_encoding = name_encoding
        self.columns   = columns

        if params is not None:
            self.params = params
        else:
            if name_encoding == 'LE':
                self.params  = {'drop_original':True,
                                'missing_new_cat':True}
            elif name_encoding == 'OHE':
                self.params  = {'dummy_na':True,
                                        'drop_first':True,
                                        'drop_original':True}
        
    def one_hot_encoding(self,df,dummies_col=[], dummy_na=True ,drop_first = True, drop_original = True):
        '''
        Function to get the one_hot_encoding of a variable
        Input:
        -df            : Dataframe al cual se le aplica one hot encoding
        -dummies_col   : Variables categoricas
        -drop_cols     : Variables a eliminar
        -drop first    : Flag para indicar si se elimina una variable del one hot encoding
        -drop_original : Flag para indicar si eliminar las columnas originales
        '''
        df_ = df.copy()
        df_cols = df[dummies_col].copy()
        df_cols = pd.get_dummies(data = df_cols, columns = dummies_col, dummy_na = dummy_na, drop_first = drop_first)
        drop_cols_unique = [c for c in df_cols.columns if df_cols[c].nunique() <= 1]
        df_cols.drop(columns = drop_cols_unique, inplace = True)
        df_ = pd.concat([df_,df_cols],axis = 1)
        if drop_original:
            df_.drop(columns = dummies_col, inplace = True)

        return df_

    def label_encoding(self,df,label_cols = [], drop_original = True, missing_new_cat = True):
        '''
        Function to get the encoding Label of a variable
        Input:
        -df         : Dataframe al cual se le aplica one hot encoding
        -label_cols : Variables categoricas
        '''
        from sklearn import preprocessing
        df_     = df.copy()
        df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
        dict_le ={}
        if missing_new_cat:
            print('Mode: Missing as new category')
            for i in df_cols.columns:
                le = preprocessing.LabelEncoder()
                print('Label Encoding: ',i)
                df_cols[i] = df_cols[i].astype('str')
                le.fit(df_cols[i])
                df_cols[i] = le.transform(df_cols[i])
                var_name = i
                dict_le[var_name] = le
        else:
            print('Mode: Missing as -1')
            for i in df_cols.columns:
                df_cols[i] = df_cols[i].fillna('NaN')
                df_cols[i] = df_cols[i].astype('str')
                le = preprocessing.LabelEncoder()
                print('Label Encoding: ',i)
                a = df_cols[i][df_cols[i]!='NaN']
                b = df_cols[i].values
                le.fit(a)
                b[b!='NaN']  = le.transform(a)
                df_cols[i] = b
                df_cols[i] = df_cols[i].replace({'NaN':-1})
                var_name = i
                dict_le[var_name] = le

        df_ = pd.concat([df_ , df_cols], axis = 1) 
        if drop_original:
            df_.drop(columns = label_cols, inplace = True)
        return df_,dict_le

    def apply_label_encoder(self,df,dict_label_encoder,drop_original = True, missing_new_cat = True):
        from sklearn import preprocessing
        df_     = df.copy()
        label_cols = [i[6:] for i in list(dict_label_encoder.keys())]
        df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
        if missing_new_cat:
            print('Mode: Missing as new category')
            for i in df_cols.columns:
                print('Applying Label Encoding: ',i)
                df_cols[i] = df_cols[i].astype('str')
                le = dict_label_encoder[i]
                df_cols[i] = le.transform(df_cols[i])

        else:
            print('Mode: Missing as -1')
            for i in df_cols.columns:
                df_cols[i] = df_cols[i].fillna('NaN')
                df_cols[i] = df_cols[i].astype('str')
                print('Applying Label Encoding: ',i)
                a = df_cols[i][df_cols[i]!='NaN']
                b = df_cols[i].values
                le = dict_label_encoder[i]
                b[b!='NaN']  = le.transform(a)
                df_cols[i] = b
                df_cols[i] = df_cols[i].replace({'NaN':-1})

        df_ = pd.concat([df_ , df_cols], axis = 1) 
        if drop_original:
            df_.drop(columns = label_cols, inplace = True)
        return df_  

    
    def fit_transform(self,df):
        if self.name_encoding == 'OHE':
            return self.one_hot_encoding(df,
                                    self.columns, 
                                    self.params['dummy_na'],
                                    self.params['drop_first'], 
                                    self.params['drop_original'])
            
            
        elif self.name_encoding == 'LE':
            df,self.dict_le  = self.label_encoding(df,
                                          self.columns, 
                                          self.params['drop_original'], 
                                          self.params['missing_new_cat'])
            
            return df
            
        else: 
            print('None encoding constructed')
    
    def transform(self,df):
        if self.name_encoding == 'OHE':
            return self.one_hot_encoding(df,
                                    self.columns, 
                                    self.params['dummy_na'],
                                    self.params['drop_first'], 
                                    self.params['drop_original'])
            
        elif self.name_encoding == 'LE':
            df   = self.apply_label_encoder(df,
                                       self.dict_le,
                                       self.params['drop_original'], 
                                       self.params['missing_new_cat'])
            
            return df
            
        else: 
            print('None encoding constructed')


# -

class TargetEncoding:
    # Ref: https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b
    def __init__(self, fold_column = 'fold', smooth=20):
        self.fold_column = fold_column
        self.smooth      = smooth
        assert self.smooth >= 0
        
    
    def fit_transform(self, df, x_col, y_col,agg = 'mean',drop_original = False):
        # Initial Parameters
        self.agg = agg
        if self.agg !='mean':
            self.smooth = 0
        self.y_col = y_col
        self.x_col = x_col
        train = df.copy() # Copy only the variables needed
        self.out_col = f'TE_{self.agg}_{self.x_col}_{self.y_col}'
        self.mean    = train[self.y_col].mean()
        self.agg_all = train.copy()
        self.agg_all.loc[:,self.out_col] = self.agg_all[self.x_col].map(self.agg_all.groupby(self.x_col).agg({self.y_col:self.agg})[self.y_col])
        self.agg_all.loc[:,'TE_count'] = self.agg_all[self.x_col].map(self.agg_all.groupby(self.x_col).agg({self.y_col:'count'})[self.y_col])
        self.agg_all = self.agg_all[[self.x_col,self.out_col,'TE_count']].drop_duplicates()
        # Calculating TE:
        for fold in np.sort(train[self.fold_column].unique()):
            X_train = train[train[self.fold_column] !=fold]
            X_val   = train[train[self.fold_column] ==fold]
            train_idx,val_idx = X_train.index.values,X_val.index.values
            # calculates metrics:
            train.loc[val_idx,self.out_col] = X_val[self.x_col].map(X_train.groupby(self.x_col).agg({self.y_col:self.agg})[self.y_col])
            train.loc[val_idx,'TE_count']   = X_val[self.x_col].map(X_train.groupby(self.x_col).agg({self.y_col:'count'})[self.y_col])
            
        if self.smooth > 0:
            # https://maxhalford.github.io/blog/target-encoding/
            train[self.out_col] = (train[self.out_col]*train['TE_count'] + self.smooth*self.mean)/(train['TE_count'] + self.smooth)
            self.agg_all[self.out_col] = (self.agg_all[self.out_col]*self.agg_all['TE_count'] + self.smooth*self.mean)/(self.agg_all['TE_count'] + self.smooth)

        train[self.out_col]   = train[self.out_col].fillna(self.mean)
        
        # Drop TE counting:
        #train.drop(columns = ['TE_count'],inplace = True)
        self.agg_all.drop(columns = ['TE_count'],inplace = True)
        if drop_original:
            train.drop(columns = [self.x_col],inplace = True)
        return train
    
    def transform(self,test):
        # Drop TE counting:
        test = test.merge(self.agg_all,on = self.x_col,how='left').reset_index(drop = True)
        test[self.out_col] = test[self.out_col].fillna(self.mean)
        
        return test

'''
class M_Target_Encoder:
    def __init__(self, fold_column = 'fold', smooth=20):
        self.fold_column = fold_column
        self.smooth      = smooth
        
    def fit_transform(self, df, x_col, y_col,drop = False):
        train      = df.copy()
        self.y_col = y_col     
        out_col    = f'TE_{x_col}_{self.y_col}'
        self.mean  = train[y_col].mean()
        
        cols          = [self.fold_column,x_col]
        agg_each_fold = train.groupby(cols).agg({y_col:['count','sum']}).reset_index()
        agg_each_fold.columns = cols + ['count_y','sum_y']
        

        agg_all = agg_each_fold.groupby(x_col).agg({'count_y':'sum','sum_y':'sum'}).reset_index()
        cols = [x_col] if isinstance(x_col,str) else x_col
        agg_all.columns = cols + ['count_y_all','sum_y_all']
        
        agg_each_fold = agg_each_fold.merge(agg_all,on=x_col,how='left')
        agg_each_fold['count_y_all'] = agg_each_fold['count_y_all'] - agg_each_fold['count_y']
        agg_each_fold['sum_y_all'] = agg_each_fold['sum_y_all'] - agg_each_fold['sum_y']
        agg_each_fold[out_col] = (agg_each_fold['sum_y_all']+self.smooth*self.mean)/(agg_each_fold['count_y_all']+self.smooth)
        agg_each_fold = agg_each_fold.drop(['count_y_all','count_y','sum_y_all','sum_y'],axis=1)
        
        agg_all[out_col] = (agg_all['sum_y_all']+self.smooth*self.mean)/(agg_all['count_y_all']+self.smooth)
        agg_all = agg_all.drop(['count_y_all','sum_y_all'],axis=1)
        self.agg_all = agg_all
        cols = [self.fold_column,x_col] if isinstance(x_col,str) else [self.fold_column]+x_col
        train = train.merge(agg_each_fold,on=cols,how='left')
        del agg_each_fold
        train[out_col] = train[out_col].fillna(self.mean)
        if drop:
            train = train.drop(columns = x_col)
        return train
    
    def transform(self, df_test, x_col,drop = False):
        # Adding for test set the global mean
        test = df_test.copy()
        out_col = f'TE_{x_col}_{self.y_col}'

        test = test.merge(self.agg_all,on = x_col,how='left')
        test[out_col] = test[out_col].fillna(self.mean)
        
        if drop:
            test = test.drop(columns = x_col)
        return test
'''
