# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 20:38:27 2021

@author: cty
"""
import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

import optuna
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
import lightgbm as lgbm
from config import lgb_param
import warnings
warnings.filterwarnings("ignore")

####code may be used
# load=pd.read_csv(f1)
# load['date']=pd.to_datetime(load['date'])
# df=pd.merge(wea,load,on='date')
# df.reset_index(drop=True)
# df.to_csv('system_load.csv')
# df.rename(columns={'rttower_load':'load'}, inplace=True)
# df=df[['date','load','speed_50_XXL','dir_50_XXL']]
# df['date']=pd.to_datetime(df['date'])
# testset=df.loc[(df['date'] >= pd.to_datetime('2022-5-31')) & (df['date'] < pd.to_datetime('2022-8-23'))]
# df=df.loc[:,~df.columns.str.contains('Unnamed')]

class LGB():
    
    def __init__(self,trial,param):
        self.trial=trial
        self.param=param
        
    def build_model(self):
        if self.trial:
            self.param = {
                'boosting_type':'gbdt',
                'class_weight':None, 
                'colsample_bytree':1.0, 
                'device':'cpu',
                'importance_type':'split', 
                'learning_rate':self.trial.suggest_float('learning_rate', 1e-5,1e-1),
                'max_depth':self.trial.suggest_int('max_depth', 2,10,step=1),
                'min_child_samples':91, 
                'min_child_weight':0.001,
                'min_split_gain':0.2, 
                'n_estimators':self.trial.suggest_int('n_estimators', 50,300,step=10),
                'n_jobs':-1, 
                'num_leaves':self.trial.suggest_int('max_depth', 2,50,step=1),
                'objective':None, 
                'random_state':1822, 
                'reg_alpha':self.trial.suggest_float('reg_alpha', 0.1, 1,step=0.1),
                'reg_lambda':self.trial.suggest_float('reg_lambda', 0.1, 1,step=0.1),
                'silent':True, 
                'subsample':self.trial.suggest_float('subsample', 0.1, 1,step=0.1), 
                'subsample_for_bin':200000,
                'subsample_freq':0
            }             

        model = lgbm.LGBMRegressor(**self.param)
        return model      
      
    def train(self, x_train, y_train, x_val, y_val):###df包含label
        print('lgb training...')
        model = self.build_model()
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                    early_stopping_rounds=20,
                  eval_metric='rmse',
                  verbose=10)
        return model  

    def test(self,x_test, y_test, model):###df包含label
        print('lgb testing...')
        pred=model.predict(x_test)
        pred=pd.DataFrame(pred)
        gt=pd.DataFrame(y_test)
        return pred,gt  
        
    def save_model(self,model,save_path):
        joblib.dump(model,save_path)
        print('saving to {}'.format(save_path))
        
    def load_model(self,model_path):
        ##model_path -> str './xx/model.pkl'
        model = joblib.load(model_path) 
        return model
    

if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    np.random.seed(10)
    f='../data/wind/ghgr.csv'
    df=pd.read_csv(f) ##df原始数据
    
    ####Step2:data process
    #method1:直接插值补全，以此对比和dataclean的效果
    del df['temp_50_XXL'],df['press_50_XXL']
    df=df.loc[:,~df.columns.str.contains('Unnamed')]
    df.interpolate(method="linear",axis=0,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    #method2:dataclean
    
    
    ####Step3:有效的特征工程feature
    from utils.feature import date_to_timeFeatures, wd_to_sincos_wd
    df=date_to_timeFeatures(df)
    df=wd_to_sincos_wd(df,['dir_50_XXL'],delete=True)
    
    ####Step4: 划分数据集
    ##要按天打乱，确保train,val,test同分布
    del df['date'] 
    from utils.tools import dataset_split
    trainset,valset,testset=dataset_split(df,n=96,ratio=[0.7,0.2,0.1])
    trainset=trainset.reset_index(drop=True)
    valset=valset.reset_index(drop=True)
    testset=testset.reset_index(drop=True)
    
    # from utils.dataprocess import IMF
    # trainset=IMF(trainset,col='load',length = 96)
    # valset=IMF(valset,col='load',length = 96)
    # testset=IMF(testset,col='load',length = 96)
    y_train=trainset.pop('load')
    x_train=trainset
    y_val=valset.pop('load')
    x_val=valset
    y_test=testset.pop('load')
    x_test=testset  
    
    ###Step5: 自动调参
    def objective(trial):
        lgb=LGB(trial=trial,param=lgb_param)
        trainded_model=lgb.train(x_train, y_train, x_val, y_val)
        pred,gt=lgb.test(x_val,y_val,trainded_model)
        loss=mean_squared_error(pred.values, gt.values)
        return loss
    
    # study=optuna.create_study(direction='maximize')
    study=optuna.create_study(direction='minimize')
    n_trials=50 # try50次
    study.optimize(objective, n_trials=n_trials)
    
    ####Step6: 使用优化超参训练+推断
    new_lgb_param=lgb_param.copy()
    new_lgb_param.update(study.best_params) ##更新超参
    lgb=LGB(trial=False,param=new_lgb_param)    

    lgb=LGB(trial=False,param=new_lgb_param)
    trainded_model=lgb.train(x_train, y_train, x_val, y_val)
    pred_old,gt=lgb.test(x_test,y_test,trainded_model)
    old_loss=mean_squared_error(pred_old.values, gt.values)
    print('old_loss:',old_loss)
    res=pd.concat([pred_old.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
    res.columns=['pred_old','gt']
    from utils.plot import plot_without_date
    plot_without_date(res,'res',cols = ['pred_old','gt'])           
        
        
        
        