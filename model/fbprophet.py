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
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")

####code may be used
# df.rename(columns={'rttower_load':'load'}, inplace=True)
# df=df[['date','load','speed_50_XXL','dir_50_XXL']]
# df['date']=pd.to_datetime(df['date'])
# testset=df.loc[(df['date'] >= pd.to_datetime('2022-5-31')) & (df['date'] < pd.to_datetime('2022-8-23'))]
    

if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    np.random.seed(10)
    
    from dataset import Dataset
    dataset=Dataset()
    df=dataset.load_system2().data
    
    
    # df=df.iloc[:64704,:]
    # df.columns=['y','ds']
    # trainset=df.iloc[:35040,:]
    # testset=df.iloc[35040:,:]     
    ####Step2:data process
    #method1:直接插值补全，以此对比和dataclean的效果
    # df=df.loc[:,~df.columns.str.contains('Unnamed')]
    df.interpolate(method="linear",axis=0,inplace=True)
    df=df[['date','target']]
    df.columns=['ds','y']  ###prophet的命名要求
    df['ds']=pd.to_datetime(df['ds'])
    # df['ds']=df['ds']-pd.Timedelta(1,unit='d') ##nextdaydate日期要减一天
    df['cap'] = 80000
    trainset=df.iloc[9983:45023,:]
    testset=df.iloc[45023:45023+96*1,:] 
    
    
    # length=int(len(df)*0.8)
    # trainset=df.iloc[:length]
    # testset=df.iloc[length:,:]         
        
    # model = Prophet(growth='logistic')
    model = Prophet()
    model.add_country_holidays(country_name='CN') ##增加节假日
    model.fit(trainset)
    future = model.make_future_dataframe(periods=96*1,freq='15t',include_history=False)
    # future['cap']=80000
    forecast = model.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)        
    
    pred=forecast['yhat']
    gt=testset['y']
    loss=mean_squared_error(pred.values, gt.values)
    print('loss:',loss)
    res=pd.concat([pred.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
    res.columns=['pred','gt']
    from my_utils.plot import plot_without_date
    plot_without_date(res,'res',cols = ['pred','gt'])         
