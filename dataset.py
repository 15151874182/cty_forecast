import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

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

class Dataset():
    
    def __init__(self):
        ##项目目录加入环境变量
        self.project_path=os.path.dirname(__file__)
        sys.path.append(self.project_path)
            
    def load_system1(self,mode='normal',y_shift=96):
        self.data=pd.read_csv(os.path.join(self.project_path,'data/system/system1.csv'))
        if mode=='normal': ##普通模式，用协变量tmp预测load，load就是target
            self.data.columns=['date', 'tmp', 'target']
            self.feas=self.data[['date','tmp']] 
            self.target=self.data['target']
        elif mode=='ts':##time_series模式，用load自回归+协变量预测y_shift个点后的load，shift load是target,前面的小部分数据会丢掉
            self.data['target']=self.data['load'].shift(y_shift)
            self.data=self.data.iloc[y_shift:,:]
            self.data=self.data.reset_index(drop=True)
            self.feas=self.data[['date','tmp','load']] 
            self.target=self.data['target']   
        self.freq=1440/int(pd.infer_freq(self.data['date'].head(5))[:-1])
        self.days=len(self.data)/self.freq
        return self

    def load_system2(self,mode='normal',y_shift=96):
        self.data=pd.read_csv(os.path.join(self.project_path,'data/system/system2.csv'))
        if mode=='normal': ##普通模式，用协变量tmp预测load，load就是target
            self.data.columns=['date', 'tmp', 'target']
            self.feas=self.data[['date','tmp']] 
            self.target=self.data['target']
        elif mode=='ts':##time_series模式，用load自回归+协变量预测y_shift个点后的load，shift load是target,前面的小部分数据会丢掉
            self.data['target']=self.data['load'].shift(y_shift)
            self.data=self.data.iloc[y_shift:,:]
            self.data=self.data.reset_index(drop=True)
            self.feas=self.data[['date','tmp','load']] 
            self.target=self.data['target']   
        self.freq=1440/int(pd.infer_freq(self.data['date'].head(5))[:-1])
        self.days=len(self.data)/self.freq
        return self

    def load_system_tang(self,mode='normal',y_shift=96):
        self.data=pd.read_csv(os.path.join(self.project_path,'data/system/system_tang.csv'))
        if mode=='normal': ##普通模式，用协变量tmp预测load，load就是target
            self.data.columns=['target','date']
            self.feas=self.data['date'] 
            self.target=self.data['target']
        elif mode=='ts':##time_series模式，用load自回归+协变量预测y_shift个点后的load，shift load是target,前面的小部分数据会丢掉
            self.data['target']=self.data['load'].shift(y_shift)
            self.data=self.data.iloc[y_shift:,:]
            self.data=self.data.reset_index(drop=True)
            self.feas=self.data[['date','load']] 
            self.target=self.data['target']   
        self.freq=1440/int(pd.infer_freq(self.data['date'].head(5))[:-1])
        self.days=len(self.data)/self.freq
        return self
    
    def load_busbar1_cluster(self): ##这个数据母线聚类用的
        self.data=pd.read_csv(os.path.join(self.project_path,'data/busbar/busbar1.csv'))
        return self
    def load_busbar1_duibi(self): ##wulin的对比
        self.data=pd.read_excel(os.path.join(self.project_path,'data/busbar/对比.xlsx'))
        return self

    def load_wind1(self,mode='normal',y_shift=96):
        self.data=pd.read_csv(os.path.join(self.project_path,'data/wind/wind1.csv'))
        if mode=='normal': ##普通模式，用协变量预测load，load就是target
            self.data.columns=['date', 'ws30', 'wd30', 'ws50', 'wd50', 'ws70', 'wd70', 't_50', 'p_50',
                   'target']
            self.feas=self.data[['date', 'ws30', 'wd30', 'ws50', 'wd50', 'ws70', 'wd70', 't_50', 'p_50']] 
            self.target=self.data['target']
        elif mode=='ts':##time_series模式，用load自回归+协变量预测y_shift个点后的load，shift load是target,前面的小部分数据会丢掉
            self.data['target']=self.data['load'].shift(y_shift)
            self.data=self.data.iloc[y_shift:,:]
            self.data=self.data.reset_index(drop=True)
            self.feas=self.data[['date', 'ws30', 'wd30', 'ws50', 'wd50', 'ws70', 'wd70', 't_50', 'p_50']] 
            self.target=self.data['target'] 
        self.freq=1440/int(pd.infer_freq(self.data['date'].head(5))[:-1])
        self.days=len(self.data)/self.freq
        return self