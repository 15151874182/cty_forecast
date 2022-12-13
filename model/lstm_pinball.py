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
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PinballLoss(nn.Module):
    """ Pinball loss at all confidence levels.
    """
    def __init__(self):
        super(PinballLoss, self).__init__()

    def forward(self, pred, gt, alpha):
        diff = gt - pred
        sign = diff >= 0
        loss = alpha * sign * diff - (1 - alpha) * (~sign) * diff
        return loss.mean()


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
        h_0 = torch.rand(self.num_directions * self.num_layers, x.shape[0], self.hidden_size)*x[0].std()+x[0].mean().to(device)
        c_0 = torch.rand(self.num_directions * self.num_layers, x.shape[0], self.hidden_size)*x[0].std()+x[0].mean().to(device)
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
        self.model_upper = self.build_model(x_train)
        self.model_lower = self.build_model(x_train)
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
        loss_fn = PinballLoss()
        optimizer_upper = torch.optim.Adam(self.model_upper.parameters(), lr=1e-3,
                                     weight_decay=0)
        optimizer_lower = torch.optim.Adam(self.model_lower.parameters(), lr=1e-3,
                                     weight_decay=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3,
                                     weight_decay=0)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
        #                             momentum=0.9, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        self.model_upper.train()
        self.model_lower.train()
        self.model.train()        
        for epoch in tqdm(range(300)):
            # 训练步骤开始
            for x,y in train_loader:
                x=x.to(device)
                y=y.to(device)
                
                pred_upper = self.model_upper(x)
                loss_90=loss_fn(pred_upper,y,alpha=0.9)    
                optimizer_upper.zero_grad()
                loss_90.backward()
                optimizer_upper.step()
                
                pred_lower = self.model_lower(x)
                loss_10=loss_fn(pred_lower,y,alpha=0.1)    
                optimizer_lower.zero_grad()
                loss_10.backward()
                optimizer_lower.step()                

                pred = self.model(x)
                loss_50=loss_fn(pred,y,alpha=0.5)    
                optimizer.zero_grad()
                loss_50.backward()
                optimizer.step()

            pred_upper=pred_upper.detach().cpu().numpy().reshape(-1,1)
            pred_lower=pred_lower.detach().cpu().numpy().reshape(-1,1)
            pred=pred.detach().cpu().numpy().reshape(-1,1)
            
            y=y.cpu().numpy().reshape(-1,1)
            pred_upper=pd.DataFrame(pred_upper)
            pred_lower=pd.DataFrame(pred_lower)
            pred=pd.DataFrame(pred)
            y=pd.DataFrame(y.reshape(-1))
            res=pd.concat([pred_upper,pred_lower,pred,y],axis=1)
            res.columns=['pred_upper','pred_lower','pred','gt']
            from my_utils.plot import plot_fill_between
            plot_fill_between(res[:600],'res',cols = ['pred_upper','pred_lower','pred','gt']) 

                
            # res=abs(pred-y)/y
            # print('train mape:',float(1-res.mean()))
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
        self.model_upper.eval()
        self.model_lower.eval()
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

            pred_upper = self.model_upper(x)
            pred_upper=pred_upper.detach().cpu().numpy().reshape(-1,1)
            pred_upper=self.scaler_y.inverse_transform(pred_upper)

            pred_lower = self.model_lower(x)
            pred_lower=pred_lower.detach().cpu().numpy().reshape(-1,1)
            pred_lower=self.scaler_y.inverse_transform(pred_lower)
            
            pred = self.model(x)
            pred=pred.detach().cpu().numpy().reshape(-1,1)
            pred=self.scaler_y.inverse_transform(pred)
            
            y=y.cpu().numpy().reshape(-1,1)
            res=abs(pred-y)/y
            print('test mape:',float(1-res.mean()))  
            
        pred_upper=pd.DataFrame(pred_upper)
        pred_lower=pd.DataFrame(pred_lower)
        pred=pd.DataFrame(pred)
        y=pd.DataFrame(y.reshape(-1))
        res=pd.concat([pred_upper,pred_lower,pred,y],axis=1)
        res.columns=['pred_upper','pred_lower','pred','gt']
        from my_utils.plot import plot_fill_between
        plot_fill_between(res[:600],'res',cols = ['pred_upper','pred_lower','pred','gt']) 
        self.res=res
        
if __name__=='__main__':
    ##项目目录加入环境变量
    project_path=os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_path)
    ####Step1:read raw data
    from dataset import Dataset
    dataset=Dataset()
    # df=dataset.load_system_tang(mode='ts',y_shift=96).data
    df=dataset.load_system1(mode='ts',y_shift=96).data
    df=df[['date','load','target']]
    df['day-1']=df['load'].shift(96*1)
    df['day-2']=df['load'].shift(96*2)
    df['day-3']=df['load'].shift(96*3)
    df=df.dropna()
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
    
    lstm=LSTM()
    lstm.train(x_train, y_train, x_val, y_val)
    lstm.test(x_test,y_test,lstm.model)