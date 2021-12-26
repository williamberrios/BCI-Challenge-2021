# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgbm
from catboost import CatBoostRegressor, Pool, cv
import catboost
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
import warnings
from tqdm.notebook import tqdm as tqdm_notebook
import wandb
import os
import pickle
os.environ['WANDB_SILENT']="true"


# +
def reg_metrics(name = 'RMSE',trues = [], preds = []):
    if name == 'RMSE':
        return np.sqrt(mean_squared_error(trues,preds))
    elif name == 'MAE':
        return mean_absolute_error(trues,preds)
    elif name == 'MAPE':
        return mean_absolute_percentage_error(true,preds)
    elif name == 'log_mae':
        return mean_absolute_error(np.exp(trues)-1,np.exp(preds)-1)
    elif name == 'sqrt_mae':
        return mean_absolute_error(trues**2,preds**2)
    elif name == 'power_mae':
        return mean_absolute_error(trues**4,preds**4)
    elif name == 'inv_mae':
        return mean_absolute_error(1/trues-1,1/preds-1)
    elif name == 'inv_sqrt_mae':
        return mean_absolute_error(np.power(trues,-2)-1,np.power(preds,-2)-1)
    else:
        raise Exception('Metric not implemented yet')

def shap_global_importance(shap_values,shap_df):
    shap_importance = np.mean(np.abs(shap_values),axis = 0)
    features        = shap_df.columns.to_list()
    df = pd.DataFrame({'Features':features,'|mean|-shap':shap_importance})
    return df


# -

class Trainer_Boosting:
    def __init__(self,
                 df,
                 model_params,
                 train_params = {'model_name'  : 'lightgbm',
                                 'fold_column'     : 'fold',
                                 'target_column'   : 'target',
                                 'cat_vars'        : [],
                                 'metric'          : 'RMSE',
                                 'feval'           : None,
                                 'early_stopping'  : 200,
                                 'max_boost_round' : 8000,
                                 'run_wandb'       : False,
                                 'project_name'    : None,
                                 'group_name'      : None}
                ):
        self.df = df
        self.model_params = model_params
        self.train_params = train_params
        
        
        
    
    def _initialization_(self):
        self.tr_metric     = []
        self.val_metric    = []
        self.metric        = self.train_params['metric']
        self.fold_column   = self.train_params['fold_column']
        self.target_column = self.train_params['target_column']
        self.cat_vars      = self.train_params['cat_vars']
        self.model_name    = self.train_params['model_name']

        self.importances = pd.DataFrame()
        self.importances['Features'] = self.df.drop(columns  = [self.fold_column,self.target_column]).columns[:]
        self.models = []
        self.oof = np.zeros((len(self.df)))
    
    def train(self):
        # Init necessary list and parameters
        self._initialization_()
        # ======= Training ===========
        start = int(time.time() * 1000)
        for fold in range(len(self.df[self.fold_column].unique())):
            # Get the train and Valid sets
            self.run = self.run_group_wandb(fold)
            df_tr  = self.df[self.df[self.fold_column] != fold].reset_index(drop  = True)
            df_val = self.df[self.df[self.fold_column] == fold].reset_index(drop  = True)
            # Split by independent and dependent variables
            X_train, y_train  = df_tr.drop(columns   = [self.fold_column,self.target_column]), df_tr[self.target_column]
            X_valid, y_valid  = df_val.drop(columns  = [self.fold_column,self.target_column]), df_val[self.target_column]
            self.features_list = X_train.columns.to_list()
            # Index for categorical variable if exist
            if self.cat_vars:
                features = [x for x in X_train.columns]
                self.cat_ind = [features.index(x) for x in self.cat_vars]
            else:
                self.cat_ind = []
            # Print features for training and categorical features
            print(f"Features for training: {self.features_list}")
            print(f" Cat_indx: {self.cat_ind}, Cat_Features: {[self.features_list[i] for i in self.cat_ind ]}")
            
            if  self.model_name == 'lightgbm':
                self.train_lightgbm(X_train,y_train,X_valid,y_valid,self.cat_ind,fold)
                    
            elif self.model_name == 'xgboost':
                self.train_xgboost(X_train,y_train,X_valid,y_valid,fold)
                    
            elif self.model_name == 'catboost':
                self.train_catboost(X_train,y_train,X_valid,y_valid,self.cat_ind,fold)
            else:
                raise Exception('Boosting Model not implemented yet')
            
            # Saving Metrics:
            if self.run:
                self.run.log({f'Train_{self.metric}' : self.tr_metric[-1],
                              f'Valid_{self.metric}' : self.val_metric[-1],
                               'Feature_List'        : self.features_list,
                               'Cat_Variables'       : self.cat_vars})
        # ===== Summary of Results ======= #
        
        end = int(time.time() * 1000) 
        self.results = pd.DataFrame({f'Model_Name'               : [self.model_name],
                                     f'Mean Valid {self.metric}' : [np.mean(self.val_metric)],
                                     f'Std Valid {self.metric}'  : [np.std(self.val_metric)],
                                     f'Mean Train {self.metric}' : [np.mean(self.tr_metric)],
                                     f'Std Train {self.metric}'  : [np.std(self.tr_metric)],
                                     f'OOF {self.metric}'        : [reg_metrics(self.metric,self.df[self.target_column].values,self.oof)],
                                     f'Diff {self.metric}'       : [np.mean(self.tr_metric) - np.mean(self.val_metric)],
                                     f'Time'                     : [str(end - start) + ' s']})
        print(f'================ OOF Results ==================')
        print(f'OOF {self.metric}: {self.results[f"OOF {self.metric}"].values[0]}')  
        # Saving Metrics:
        if self.run:
            self.run.log({f'Mean_Train_{self.metric}' : self.results[f'Mean Train {self.metric}'].values[0],
                          f'Mean_Valid_{self.metric}' : self.results[f'Mean Valid {self.metric}'].values[0],
                          f'OOF_{self.metric}'        : self.results[f'OOF {self.metric}'].values[0]})
            self.run.finish()
            
        # Save Base Scores for lofo importance:
        self.base_oof = self.results[f"OOF {self.metric}"].values[0]
        self.base_val_metric = self.val_metric
        
        
    def predict(self,test,preds_returns = 'average'):
        if self.model_name == 'lightgbm':
            return self.predict_lightgbm(test,preds_returns)
        elif self.model_name == 'catboost':
            return self.predict_catboost(test,preds_returns)
        elif self.model_name == 'xgboost':
            return self.predict_xgboost(test,preds_returns)
        else:
            raise Exception('Model not implemented for prediction')
            
            
    def save_results(self,output_folder = None):
        # Create Directories:
        if self.train_params.get('group_name',None) is not None:
            if output_folder is None:
                output_folder = '../SavedModels'
            output_folder = os.path.join(output_folder,self.train_params['group_name'])
            
            
        os.makedirs(output_folder, exist_ok=True)
        # ==============
        # Save models
        # ==============
        for fold,model in enumerate(self.models):
            os.makedirs(os.path.join(output_folder,f'fold_{fold}'), exist_ok=True)
            pickle.dump(model, open(os.path.join(output_folder,f'fold_{fold}','model.pkl'), 'wb'))
        # ==============================
        # Save OOF ( Same order as oof)
        # ==============================
        np.save(os.path.join(output_folder,'oof.npy'), self.oof) 
        
        
    ### =================== Training Functions =======================##
    
    def train_lightgbm(self,X_train,y_train,X_valid,y_valid,cat_ind,fold):
        # Create lgbm Dataset 
        lgbm_train = lgbm.Dataset(X_train, label = y_train,categorical_feature = cat_ind)
        lgbm_eval  = lgbm.Dataset(X_valid, y_valid, reference = lgbm_train,categorical_feature = cat_ind)

        # Training lgbm model
        print(f'---------- Training fold Nº {fold} ----------')
        lgbm_model = lgbm.train(self.model_params,
                                lgbm_train,
                                num_boost_round       = self.train_params['max_boost_round'],
                                early_stopping_rounds = self.train_params['early_stopping'], 
                                verbose_eval          = 50, 
                                categorical_feature   = cat_ind,
                                valid_sets            = [lgbm_train,lgbm_eval],
                                feval                 = self.train_params.get('feval',None)) 
        
        # Calculating inside importance
        self.importances[f'importance_{fold}']= lgbm_model.feature_importance(importance_type = 'gain')
        
        # Saving Model
        self.models.append(lgbm_model)
        
        # Saving oof predictions
        valid_idx = self.df[self.df[self.fold_column] == fold].index.values
        
        # Evaluating Metrics for train and Validation
        y_train_pred = lgbm_model.predict(X_train,num_iteration=lgbm_model.best_iteration)
        self.tr_metric    += [reg_metrics(self.metric,y_train,y_train_pred)]

        y_valid_pred   = lgbm_model.predict(X_valid,num_iteration=lgbm_model.best_iteration)
        self.val_metric    += [reg_metrics(self.metric,y_valid,y_valid_pred)]
        
        self.oof[valid_idx] = y_valid_pred
        
        print(f'================ Result Fold {fold} ==================')
        print(f"Train {self.metric}: {self.tr_metric[-1]}        Valid {self.metric}: {self.val_metric[-1]}")
        print(f'=======================================================')
    
    def train_catboost(self,X_train,y_train,X_valid,y_valid,cat_ind,fold):
        cat_train  = Pool(data = X_train, label = y_train,cat_features = cat_ind)
        cat_eval   = Pool(data = X_valid, label = y_valid,cat_features = cat_ind)
        self.model_params['verbose'] = 50
        self.model_params['early_stopping_rounds'] = self.train_params['early_stopping']
        self.model_params['num_boost_round']       = self.train_params['max_boost_round']
        
        print(f'---------- Training fold Nº {fold} ----------')
        cat_model = CatBoostRegressor(**self.model_params)
        cat_model.fit(cat_train,
                      use_best_model = True,
                      eval_set       = [cat_eval],
                      verbose        = 50,
                      plot           = False)
        self.importances[f'importance_{fold}']= cat_model.get_feature_importance(cat_eval,type = 'FeatureImportance')
        # Saving Model
        self.models.append(cat_model)
        
        # Saving oof predictions
        valid_idx = self.df[self.df[self.fold_column] == fold].index.values
        
        # Evaluating Metrics for train and Validation
        y_train_pred = cat_model.predict(X_train)
        self.tr_metric   += [reg_metrics(self.metric,y_train,y_train_pred)]

        y_valid_pred   = cat_model.predict(X_valid)
        self.val_metric    += [reg_metrics(self.metric,y_valid,y_valid_pred)]
        self.oof[valid_idx] = y_valid_pred
        print(f'================ Result Fold {fold} ==================')
        print(f"Train {self.metric}: {self.tr_metric[-1]}   Valid {self.metric}: {self.val_metric[-1]}    Best Iter: {cat_model.get_best_iteration()}")
        print(f'=======================================================')

        
    def train_xgboost(self,X_train,y_train,X_valid,y_valid,fold):
        # Create XGB Dataset 
        xgb_train = xgb.DMatrix(X_train, y_train, feature_names = X_train.columns)
        xgb_val   = xgb.DMatrix(X_valid, y_valid, feature_names = X_train.columns)
        # Training xgboost model
        print(f'---------- Training fold Nº {fold} ----------')
        xgb_model = xgb.train(params = self.model_params, 
                              dtrain = xgb_train , 
                              num_boost_round =  self.train_params['max_boost_round'], 
                              evals = [(xgb_train, "Train"), (xgb_val, "Valid")],
                              verbose_eval = 50, 
                              early_stopping_rounds = self.train_params['early_stopping'])
        
        
        importance_dfi   = pd.DataFrame.from_dict(xgb_model.get_score(importance_type="gain"), orient='index',columns=[f'importance_{fold}'])
        importance_dfi   = importance_dfi.rename_axis('Features').reset_index()
        self.importances = self.importances.merge(importance_dfi,on='Features',how='left')
        
        # Saving Model
        self.models.append(xgb_model)
        
        # Saving oof predictions
        valid_idx = self.df[self.df[self.fold_column] == fold].index.values
        
        # Evaluating Metrics for train and Validation
        y_train_pred      = xgb_model.predict(xgb_train, ntree_limit = xgb_model.best_ntree_limit)
        self.tr_metric   += [reg_metrics(self.metric,y_train,y_train_pred)]

        y_valid_pred        = xgb_model.predict(xgb_val, ntree_limit = xgb_model.best_ntree_limit)
        self.val_metric    += [reg_metrics(self.metric,y_valid,y_valid_pred)]
        self.oof[valid_idx] = y_valid_pred
        
        print(f'================ Result Fold {fold} ==================')
        print(f"Train {self.metric}: {self.tr_metric[-1]}        Valid {self.metric}: {self.val_metric[-1]}")
        print(f'=======================================================')
    
    ### =================== Prediction Functions =======================##
    
    def predict_lightgbm(self,test,preds_return = 'average'):
        preds = []
        for lgbm_model in tqdm(self.models):
            preds.append(lgbm_model.predict(test[self.features_list],num_iteration=lgbm_model.best_iteration))
        if preds_return == 'average':
            return np.mean(preds,axis = 0)
        elif preds_return == 'median':
            return np.median(preds,axis = 0)
        elif preds_return == 'raw':
            return preds
        else:
            raise Exception('No method implemented for prediction')
            
    def predict_catboost(self,test,preds_return = 'average'):
        preds = []
        for cat_model in tqdm(self.models):
            preds.append(cat_model.predict(test[self.features_list]))
        if preds_return == 'average':
            return np.mean(preds,axis = 0)
        elif preds_return == 'median':
            return np.median(preds,axis = 0)
        elif preds_return == 'raw':
            return preds
        else:
            raise Exception('No method implemented for prediction')
            
    def predict_xgboost(self,test,preds_return = 'average'):
        preds = []
        for xgb_model in tqdm(self.models):
            preds.append(xgb_model.predict(xgb.DMatrix(test[self.features_list])))
        if preds_return == 'average':
            return np.mean(preds,axis = 0)
        elif preds_return == 'median':
            return np.median(preds,axis = 0)
        elif preds_return == 'raw':
            return preds
        else:
            raise Exception('No method implemented for prediction')
        
    # ================= Feature Importance =============================
    
    def get_importance(self,limit_importance = None,figsize=(8, 8)):
        plt.style.use('fivethirtyeight')
        # Return dataframe with importance,
        cols_imp = [col for col in self.importances.columns.to_list() if col.startswith('importance')]
        importance_df = self.importances.copy()
        importance_df['mean_importance'] = importance_df[cols_imp].mean(axis = 1)
        importance_df['std_importance']  = importance_df[cols_imp].std(axis = 1)
        importance_df['CoV_importance']  = importance_df['std_importance']/importance_df['mean_importance']
        importance_df.sort_values("mean_importance", inplace=True)
        
        importance_df.plot( x="Features", y="mean_importance", xerr="std_importance",
                                              kind='barh', figsize=figsize)
        plt.show()
        return importance_df.sort_values("mean_importance",ascending = False)
    
    
    def _init_shap_importance(self):
        """
        Shap Importance for Tree Models
        """
        list_shap_values  = []
        list_valid_idxs   = []
        list_explainers   = []
        for fold,model in tqdm(enumerate(self.models)):
            explainer = shap.Explainer(model,algorithm = 'auto')
            list_explainers.append(explainer)
            X_valid   = self.df[self.df[self.fold_column] == fold]
            val_idx   = X_valid.index
            X_valid   = X_valid.reset_index(drop = True)[self.features_list]
            shap_values = explainer.shap_values(X_valid)
            #for each iteration we save the test_set index and the shap_values
            list_shap_values.append(shap_values)
            list_valid_idxs.append(val_idx)
        self.shap_values = np.concatenate(list_shap_values,axis = 0)
        valid_idxs  = np.concatenate(list_valid_idxs,axis = 0)
        self.shap_df     = self.df.iloc[valid_idxs,:][self.features_list]
        
    def plot_shap_importance(self,max_display = 100):
        # Get shap values
        self._init_shap_importance()
        imp_shap_df = shap_global_importance(self.shap_values,self.shap_df)
        # Plot shap independent value
        shap.summary_plot(self.shap_values, self.shap_df,max_display = max_display)
        imp_shap_df.sort_values(by = ['|mean|-shap']).plot( x="Features", y='|mean|-shap',kind='barh',color = 'r')
        plt.show()
        return imp_shap_df.sort_values(by = ['|mean|-shap'],ascending = False)
    
    def plot_interaction_shap(self,columns = None,fold = 0 ):
        explainer = shap.Explainer(self.models[fold])
        shap_values = explainer(self.df[self.df[self.fold_column] == fold].reset_index(drop = True)[self.features_list])
        if columns is None:
            columns = self.features_list
            
        for col in columns:
            shap.plots.scatter(shap_values[:,col], color=shap_values)
    
    def plot_watterfall_shap(self,row_idx = 0):
        # Determine the model for the row:
        fold = int(self.df.loc[row_idx,:][self.train_params['fold_column']])
        explainer   = shap.Explainer(self.models[fold])
        shap_values = explainer(self.df.iloc[row_idx,:][self.features_list])
        # visualize the first prediction's explanation
        shap.plots.waterfall(shap_values)
    # =============================================================================
    # =========================== Lofo Importance =================================
    # =============================================================================
  
    def get_lofo_importance(self):
        from lofo import plot_importance
        len_folds = self.df[self.fold_column].nunique()
        cols_lofo = ['feature','importance_mean','importance_std','oof_importance'] +  [f'val_imp_{i}' for i in range(len_folds)]
        self.table_lofo = []
        
        for feature_to_remove in tqdm_notebook(self.features_list):
            val_metric,oof_value = self._get_cv_score(feature_to_remove)
            row = [feature_to_remove,
                   np.mean(np.array(val_metric)-np.array(self.base_val_metric)),
                   np.std(np.array(val_metric)-np.array(self.base_val_metric)),
                   oof_value - self.base_oof ]
            for idx,fold_metric in enumerate(val_metric):
                row.append(fold_metric - self.base_val_metric[idx])
            self.table_lofo.append(row)
        self.table_lofo = pd.DataFrame(self.table_lofo,columns = cols_lofo)
        self.table_lofo.sort_values(by = ['importance_mean'],ascending = False,inplace = True)
        plot_importance(self.table_lofo)
        plt.show()
        return self.table_lofo
        
        
    def _get_cv_score(self,feature_to_remove):
        tr_metric = []
        val_metric = []
        oof = np.zeros((len(self.df)))
        features,cat_ind = self._get_feats(feature_to_remove)
        for fold in range(len(self.df[self.fold_column].unique())):
            # Get the train and Valid sets
            df_tr  = self.df[self.df[self.fold_column] != fold].reset_index(drop  = True)
            df_val = self.df[self.df[self.fold_column] == fold].reset_index(drop  = True)
            # Split by independent and dependent variables
            X_train, y_train  = df_tr[features]  , df_tr[self.target_column]
            X_valid, y_valid  = df_val[features] , df_val[self.target_column]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if  self.model_name == 'lightgbm':
                    val_metric,oof_value = self._lofo_train_lightgbm(X_train,y_train,X_valid,y_valid,cat_ind,fold,tr_metric,val_metric,oof)
        print('Mean metric:',np.mean(val_metric))
        return val_metric,oof_value
    
    def _lofo_train_lightgbm(self,X_train,y_train,X_valid,y_valid,cat_ind,fold,tr_metric,val_metric,oof):
        
        # Create lgbm Dataset 
        lgbm_train = lgbm.Dataset(X_train, label = y_train,categorical_feature = cat_ind)
        lgbm_eval  = lgbm.Dataset(X_valid, y_valid, reference = lgbm_train,categorical_feature = cat_ind)

        # Training lgbm model
        lgbm_model = lgbm.train(self.model_params,
                                lgbm_train,
                                num_boost_round       = self.train_params['max_boost_round'],
                                early_stopping_rounds = self.train_params['early_stopping'], 
                                verbose_eval          = 0, 
                                categorical_feature   = cat_ind,
                                valid_sets            = [lgbm_train,lgbm_eval]) 
        
        # Saving oof predictions
        valid_idx = self.df[self.df[self.fold_column] == fold].index.values
        
        # Evaluating Metrics for train and Validation
        y_train_pred   = lgbm_model.predict(X_train,num_iteration=lgbm_model.best_iteration)
        tr_metric     += [reg_metrics(self.metric,y_train,y_train_pred)]
        y_valid_pred   = lgbm_model.predict(X_valid,num_iteration=lgbm_model.best_iteration)
        val_metric    += [reg_metrics(self.metric,y_valid,y_valid_pred)]
        oof[valid_idx] = y_valid_pred
        return val_metric,reg_metrics(self.metric,self.df[self.target_column].values,oof)

    
    def _get_feats(self,feature_to_remove):
        # Get features for training
        features = self.features_list.copy()
        features.remove(feature_to_remove)
        
        # Update cat index
        if self.cat_ind:
            cat_vars = self.cat_vars.copy()
            if feature_to_remove in self.cat_vars:
                cat_vars.remove(feature_to_remove)
            cat_ind = [features.index(x) for x in cat_vars]
        else:
            cat_ind = []
        #print(f"Features removed: {feature_to_remove}")
        #print(f"New Cat_indx: {cat_ind}, Cat_Features: {[features[i] for i in cat_ind ]}")
        return features,cat_ind
        
        
    #================== Tracking experiments ======================
    
    def run_group_wandb(self,fold):
        if self.train_params.get('run_wandb',None):
            run = wandb.init(project = self.train_params['project_name'],
                             group   = self.train_params['group_name'],
                             save_code = True,
                             reinit    = True)
            
            run.config.update(self.model_params)
            run.config.update(self.train_params)
            run.config.update({"fold": fold}) # Fold running the model
            run.name = f'fold_{fold}'
            run.save()
            return run
        else:
            return None
