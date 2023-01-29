# -*- coding: utf-8 -*-
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_pinball_loss
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from prophet import Prophet

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
# group=df.groupby(df['date'].apply(lambda x:x.split()[0]))
    
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LstmPyorch(nn.Module):
    def __init__(self, fea_size, seq_len, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        # self.trial=trial
        # hidden_size=self.trial.suggest_int('hidden_size', 1,20,step=1),
        # num_layer=self.trial.suggest_int('num_layer', 1,10,step=1),        
        self.fea_size = fea_size
        self.seq_len=seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.fea_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h_0 = torch.rand(self.num_directions * self.num_layers, x.shape[0], self.hidden_size).to(device)*x[0].std()+x[0].mean()
        c_0 = torch.rand(self.num_directions * self.num_layers, x.shape[0], self.hidden_size).to(device)*x[0].std()+x[0].mean()
        output, (hidden, cell) = self.lstm(x,(h_0,c_0)) # [16, 96, 3]
        pred = self.linear(output)  # [16, 96, 1]
        return pred    

class LSTM():
    
    def __init__(self):
        pass
        
    def build_model(self,x_train):
        self.model=LstmPyorch(fea_size=x_train.shape[1],
                         seq_len=96,
                         hidden_size=5,
                         num_layers=1,
                         output_size=1, 
                         batch_size=32).to(device)
        return self.model      


    def train(self, x_train, y_train, x_val, y_val):###df包含label
        self.model = self.build_model(x_train)
        # 先转换成torch能识别的dataset
        import torch.utils.data as data
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        x_train = self.scaler_x.fit_transform(x_train)
        y_train = self.scaler_y.fit_transform(y_train[:,None])      
        x_val = self.scaler_x.transform(x_val)
        y_val = self.scaler_y.transform(y_val[:,None])

        x=torch.Tensor(x_train.reshape(-1,self.model.seq_len,x_train.shape[1]))
        y=torch.Tensor(y_train.reshape(-1,self.model.seq_len,1))
        trainset = data.TensorDataset(x, y)
        train_loader = data.DataLoader(dataset=trainset,
                                       batch_size=self.model.batch_size,
                                       shuffle=False, ##x_train已经打乱过了
                                       num_workers=1,
                                       drop_last=True)
        
        x=torch.Tensor(x_val.reshape(-1,self.model.seq_len,x_val.shape[1]))
        y=torch.Tensor(y_val.reshape(-1,self.model.seq_len,1))        
        valset = data.TensorDataset(x, y)
        val_loader = data.DataLoader(dataset=valset,
                                     batch_size=self.model.batch_size,
                                     shuffle=False, ##x_val已经打乱过了
                                     num_workers=1,
                                     drop_last=True)        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3,
                                     weight_decay=0)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
        #                             momentum=0.9, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        
        for epoch in tqdm(range(500)):
            # 训练步骤开始
            self.model.train()
            for x,y in train_loader:
                x=x.to(device)
                y=y.to(device)
                pred = self.model(x)
                loss = loss_fn(pred, y)           
                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res=abs(pred-y)/y
            # print('train mape:',float(1-res.mean()))
            
            # pred=pred.detach().cpu().numpy().reshape(-1,1)
            # y=y.cpu().numpy().reshape(-1,1)
            # pred=pd.DataFrame(pred)
            # y=pd.DataFrame(y.reshape(-1))
            # res=pd.concat([pred,y],axis=1)
            # res.columns=['pred','gt']
            # from my_utils.plot import plot_without_date
            # plot_without_date(res[:600],'res',cols = ['pred','gt'])             

            # scheduler.step()
            
            # if epoch%10==0:
            #     # 验证步骤开始
            #     self.model.eval()
            #     val_mape = []
            #     for x,y in val_loader:
            #         pred = self.model(x)
            #         res=abs(pred-y)/y
            #         val_mape.append(float(1-res.mean()))
            #     print('val mean loss:',np.mean(val_mape))

    def test(self,x_test, y_test, model):   
        self.model.eval()
        x_test = self.scaler_x.transform(x_test)
        import torch.utils.data as data
        x=torch.Tensor(x_test.reshape(-1,self.model.seq_len,x_test.shape[1]))
        y=torch.Tensor(y_test.values.reshape(-1,self.model.seq_len,1))  
        testset = data.TensorDataset(x, y)
        test_loader = data.DataLoader(dataset=testset,
                                       batch_size=testset.tensors[0].shape[0], ##相当于整个batch
                                       shuffle=False, ##x_train已经打乱过了
                                       num_workers=1,
                                       drop_last=False)    
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            pred = self.model(x)
            pred=pred.detach().cpu().numpy().reshape(-1,1)
            pred=self.scaler_y.inverse_transform(pred)
            y=y.cpu().numpy().reshape(-1,1)
            res=abs(pred-y)/y
            print('test mape:',float(1-res.mean()))   
        pred=pd.DataFrame(pred)
        y=pd.DataFrame(y.reshape(-1))
        res=pd.concat([pred,y],axis=1)
        res.columns=['pred','gt']
        from my_utils.plot import plot_without_date
        plot_without_date(res[:600],'res',cols = ['pred','gt']) 
        self.res=res



if __name__=='__main__':
    ######################################prophet part
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    np.random.seed(10)
    
    from dataset import Dataset
    dataset=Dataset()
    df=dataset.load_system2().data
    df2=df.copy()
    from my_utils.dataprocess import points_to_days
    df=points_to_days(df,cols=['tmp','target'])
    
    # df=df.iloc[:64704,:]
    # df.columns=['y','ds']
    # trainset=df.iloc[:35040,:]
    # testset=df.iloc[35040:,:]     
    ####Step2:data process
    #method1:直接插值补全，以此对比和dataclean的效果
    # df=df.loc[:,~df.columns.str.contains('Unnamed')]
    # df.interpolate(method="linear",axis=0,inplace=True)
    df.rename(columns={'date':'ds','mean_target':'y'}, inplace=True)###prophet的命名要求
    df['ds']=pd.to_datetime(df['ds'])
    # df['ds']=df['ds']-pd.Timedelta(1,unit='d') ##nextdaydate日期要减一天
    # df['cap'] = 80000
    trainset_prophet=df.iloc[:-14*2,:] 
    testset_prophet=df.iloc[-14*2:-14,:] ##test baseline是12/4~12/17号
    
    # model = Prophet(growth='logistic')
    model = Prophet()
    model.add_country_holidays(country_name='CN') ##增加节假日
    model.fit(trainset_prophet)
    future = model.make_future_dataframe(periods=14,freq='d',include_history=False)
    # future['cap']=80000
    # forecast = model.predict(future)
    prophet_pred = model.predict(future)
    # prophet_pred.to_csv('../result/prophet_pred.csv',index=False)
    # prophet_pred=pd.read_csv('../result/prophet_pred.csv')
    # prophet_pred=prophet_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    fig1 = model.plot(prophet_pred)
    # fig2 = model.plot_components(prophet_pred)        
    
    pred=prophet_pred['yhat']
    gt=testset_prophet['y']
    loss=1-mean_absolute_percentage_error(gt.values, pred.values)
    print('loss:',loss)
    res=pd.concat([pred.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
    res.columns=['pred','gt']
    from my_utils.plot import plot_without_date
    plot_without_date(res,'res',cols = ['pred','gt'])         

#######################LSTM part
    df2['day-1']=df2['target'].shift(96*1)
    df2['day-2']=df2['target'].shift(96*2)
    df2['day-3']=df2['target'].shift(96*3)
    df2=df2.dropna()
    df2['ds']=df2['date'].apply(lambda x:x.split()[0])
    df2['ds']=pd.to_datetime(df2['ds'])
    trainset_prophet['ds']=pd.to_datetime(trainset_prophet['ds'])
    
    
    prophet_pred=prophet_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    prophet_pred.columns=['ds','mean_target','min_target','max_target']
    testset_prophet['ds']=pd.to_datetime(testset_prophet['ds'])
    prophet_pred['ds']=pd.to_datetime(prophet_pred['ds'])
    xx=testset_prophet[['ds','max_tmp','min_tmp','mean_tmp']]
    xxx=pd.merge(prophet_pred,xx,on='ds',how='left')
    prophet_pred.columns=['ds','mean_target','min_target','max_target']
    xxx.rename(columns={'mean_target':'y'}, inplace=True)
    xxxx=pd.concat([trainset_prophet,xxx],axis=0)
    df2=df2.iloc[:-96*14,:]
    df=pd.merge(df2,xxxx,on='ds',how='left')
    df.rename(columns={'y':'mean_target'}, inplace=True)
    del df['ds']
    # df.to_csv('../result/prophet_lstm_train.csv',index=False)
    # df=pd.read_csv('../result/prophet_lstm_train.csv')

    ##要按天打乱，确保train,val,test同分布
    from my_utils.tools import dataset_split
    del df['date']
    test_ratio=14/(df.shape[0]/96) ##这里由于按比例划分，小数计算不精确，导致用写13，testset才是14天
    trainset,valset,testset=dataset_split(df,n=96*14,ratio=[1-2*test_ratio,test_ratio,test_ratio],mode=2)
    trainset=trainset.reset_index(drop=True)
    valset=valset.reset_index(drop=True)
    testset=testset.reset_index(drop=True)
    
    y_train=trainset.pop('target')
    x_train=trainset
    y_val=valset.pop('target')
    x_val=valset
    y_test=testset.pop('target')
    x_test=testset  
    
    lstm=LSTM()
    lstm.train(x_train, y_train, x_val, y_val)
    lstm.test(x_test,y_test,lstm.model)