# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 20:38:27 2021
ARIMA缺点很多：
1 训练慢
2 无法长期预测，超过q值会预测一条直线
3 纯自回归
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
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from config import arima_param
import warnings
warnings.filterwarnings("ignore")

####code may be used
# df.rename(columns={'rttower_load':'load'}, inplace=True)
# df=df[['date','load','speed_50_XXL','dir_50_XXL']]
# df['date']=pd.to_datetime(df['date'])
# testset=df.loc[(df['date'] >= pd.to_datetime('2022-5-31')) & (df['date'] < pd.to_datetime('2022-8-23'))]


class ARIMA():
    
    def __init__(self):
        pass
    
    def build_model(self,df,if_hyp=True):
        if if_hyp==True:
            print('search for best param')
            search = sm.tsa.arma_order_select_ic(trainset['load'], ic=['bic'], 
                                                         trend='n', max_ar=5,max_ma=5)
            print('best (p,q)', search.bic_min_order)    
            global arima_param
            ##arima_param=(p,i,q),这里用最优p，q替换原来的p，q
            arima_param=(search.bic_min_order[0],arima_param[1],search.bic_min_order[1])
            
        print(f'using param:{arima_param}')
        model = sm.tsa.arima.ARIMA(df,order=arima_param)

        return model      
    
    def ADF_check(self,df):# 单位根检验-ADF检验
        print(sm.tsa.stattools.adfuller(df)) ##p值小于0.05，说明数据平稳

    def white_noise_check(self,df):# 白噪声检验
        print(acorr_ljungbox(df, lags = [6, 12],boxpierce=True))##p值小于0.05，说明数据不是白噪 

    def acf(self,df): #画ACF图
        plot_acf(df)
        plt.show()    
    def pacf(self,df):#画PACF图
        plot_pacf(df)
        plt.show()    

if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    
    ####用风电数据集，效果不好
    # np.random.seed(10)
    # f='../data/wind/ghgr.csv'
    # df=pd.read_csv(f) ##df原始数据
    # df=df[['date','load']] ##arima只能是自回归
    # df.interpolate(method="linear",axis=0,inplace=True)
    # df['date']=pd.to_datetime(df['date'])
    # df=df.loc[(df['date'] <= pd.to_datetime('2022-5-31'))]
    # length=int(len(df)*0.8)
    # trainset=df.iloc[:length]
    # testset=df.iloc[length:,:]   

    ##风电数据效果差，使用sin函数做出来的数据测试效果
    df=pd.DataFrame([np.sin(i) for i in np.arange(0,500,np.pi/4)]) 
    df.columns=['load']
    length=int(len(df)*0.8)
    trainset=df.iloc[:length]
    testset=df.iloc[length:,:]
    
    arima=ARIMA()
    model=arima.build_model(trainset['load'],if_hyp=True)
    trained_model=model.fit()
    pred=trained_model.forecast(10) ###10指预测trainset后面10个点，arima预测长度不能超过q值，所以长期预测不好，需要规律性较强的数据，风电很差，甚至一条直线
    gt=testset['load'][:10]
    loss=mean_squared_error(pred.values, gt.values)
    print('loss:',loss)
    res=pd.concat([pred.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
    res.columns=['pred_old','gt']
    from my_utils.plot import plot_without_date
    plot_without_date(res,'res',cols = ['pred_old','gt']) 

        
        
        
