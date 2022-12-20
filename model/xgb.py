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
import xgboost as xgbm
from config import xgb_param
import warnings
warnings.filterwarnings("ignore")

####code may be used
# df.rename(columns={'rttower_load':'load'}, inplace=True)
# df=df[['date','load','speed_50_XXL','dir_50_XXL']]
# df['date']=pd.to_datetime(df['date'])
# testset=df.loc[(df['date'] >= pd.to_datetime('2022-5-31')) & (df['date'] < pd.to_datetime('2022-8-23'))]

class XGB():
    
    def __init__(self,trial,param):
        self.trial=trial
        self.param=param
        
    def build_model(self):
        if self.trial:
            self.param = {
                'booster': 'gbtree',
                'n_estimators':self.trial.suggest_int('n_estimators', 50,300,step=10),
                'max_depth':self.trial.suggest_int('depth', 2,10,step=1),
                'learning_rate':self.trial.suggest_float('lr', 1e-5,1e-1),
                'subsample':self.trial.suggest_float('subsample', 0.1, 1,step=0.1),
                'gamma':self.trial.suggest_float('gamma', 0.1, 1,step=0.1),
                'reg_alpha':self.trial.suggest_float('reg_alpha', 0.1, 1,step=0.1),
                'reg_lambda':self.trial.suggest_float('reg_lambda', 0.1, 1,step=0.1),    
                'nthread':8,
                'objective': 'reg:squarederror',
                'colsample_bytree':1,
                'colsample_bylevel':1,
                'colsample_bynode':1,
                'gpu_id':-1,
                'tree_method':'auto',   
            }            
        
        model = xgbm.XGBRegressor(**self.param)  
        # print(model.get_params)
        return model      
      
    def train(self, x_train, y_train, x_val, y_val):###df包含label
        print('xgb training...')
        self.model = self.build_model()
        self.model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                    early_stopping_rounds=20,
                  eval_metric='rmse',
                  verbose=10)
        return self.model  

    def test(self,x_test, y_test):###df包含label
        print('xgb testing...')
        pred=self.model.predict(x_test)
        pred=pred.reshape(-1,1)
        gt=np.array(y_test).reshape(-1,1)
        
        res=abs(pred-gt)/gt
        print('test mape:',float(1-res.mean()))   

        pred=pd.DataFrame(pred)
        gt=pd.DataFrame(gt)        
        res=pd.concat([pred,gt],axis=1)
        res.columns=['pred','gt']
        from my_utils.plot import plot_without_date
        plot_without_date(res[:600],'res',cols = ['pred','gt']) 
        self.res=res

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
    from dataset import Dataset
    dataset=Dataset()
    # df=dataset.load_system_tang(mode='ts',y_shift=96).data
    # df=dataset.load_wind1(mode='ts',y_shift=96).data
    df=dataset.load_system1(mode='ts',y_shift=96).data
    df=df[['date','load','target']]
    df['day-1']=df['load'].shift(96*1)
    df['day-2']=df['load'].shift(96*2)
    df['day-3']=df['load'].shift(96*3)
    df=df.dropna()
    # df=df.loc[96:,:]
    # df=df[['date','ws30','target']]
    ####Step3:有效的特征工程feature
    # from utils.feature import date_to_timeFeatures, wd_to_sincos_wd
    # df=date_to_timeFeatures(df)
    # df=wd_to_sincos_wd(df,['dir_50_XXL'],delete=True)
    
    ####Step4: 划分数据集
    ##要按天打乱，确保train,val,test同分布
    from my_utils.tools import dataset_split
    del df['date']
    trainset,valset,testset=dataset_split(df,n=96,ratio=[0.7,0.2,0.1],mode=2)
    trainset=trainset.reset_index(drop=True)
    valset=valset.reset_index(drop=True)
    testset=testset.reset_index(drop=True)
    
    y_train=trainset.pop('target')
    x_train=trainset
    y_val=valset.pop('target')
    x_val=valset
    y_test=testset.pop('target')
    x_test=testset  
    
    ###Step5: 自动调参
    def objective(trial):
        xgb=XGB(trial=trial,param=xgb_param)
        xgb.train(x_train, y_train, x_val, y_val)
        xgb.test(x_val,y_val)
        loss=mean_squared_error(xgb.res['pred'].values, xgb.res['gt'].values)
        return loss
    
    # study=optuna.create_study(direction='maximize')
    study=optuna.create_study(direction='minimize')
    n_trials=50 # try50次
    study.optimize(objective, n_trials=n_trials)
    
    ####Step6: 使用优化超参训练+推断
    new_xgb_param=xgb_param.copy()
    new_xgb_param.update(study.best_params) ##更新超参
    xgb=XGB(trial=False,param=new_xgb_param)    

    # xgb=xgb(trial=False,param=xgb_param)
    xgb.train(x_train, y_train, x_val, y_val)
    xgb.test(x_test,y_test)
    
    # from PyEMD import EMD 
    # emd = EMD()
    # ts=np.array(pred_old).reshape(-1)
    # emd.emd(ts)
    # imfs, res = emd.get_imfs_and_residue() 
    # pred_old=res
    # for i in range(0,imfs.shape[0]):
    #     pred_old+=imfs[i] ##将imfs[0]高频噪音去掉，得到平滑的曲线

        
        
        