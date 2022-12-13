import os,sys
import datetime
import math
import scipy
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
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

class DeeparPyorch(nn.Module):
    def __init__(self, fea_size, seq_len, hidden_size, num_layers, batch_size):
        super().__init__()
        # self.trial=trial
        # hidden_size=self.trial.suggest_int('hidden_size', 1,20,step=1),
        # num_layer=self.trial.suggest_int('num_layer', 1,10,step=1),        
        self.fea_size = fea_size
        self.seq_len=seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.fea_size, self.hidden_size, self.num_layers, batch_first=True)
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
                
        self.mu = nn.Linear(self.hidden_size * self.num_layers, 1)
        self.std=nn.Sequential(nn.Linear(self.hidden_size * self.num_layers, 1),
                               nn.Softplus()# softplus to make sure standard deviation is positive
                               )
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def init_cell(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))  
        h=hidden.contiguous() ##h要参与中间计算，但是不能影响hidden,contiguous相当于深拷贝
        h = h.permute(1,0,2).reshape(hidden.shape[1], -1)
        mu = self.mu(h)
        std = self.std(h)  
        return mu, std, hidden, cell
    
class DeepAR():
    
    def __init__(self):
        pass
    
    def log_prob_loss_fn(self,mu,std,gt):
        '''
        Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            mu: (Variable) dimension [batch_size] - estimated mean at time step t
            sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
            labels: (Variable) dimension [batch_size] z_t
        Returns:
            loss: (Variable) average log-likelihood loss across the batch
        '''
        distribution = torch.distributions.normal.Normal(mu, std)
        likelihood = distribution.log_prob(gt)
        return -torch.mean(likelihood)
    
    def build_model(self,x_train):
        self.model=DeeparPyorch(fea_size=x_train.shape[1],
                         seq_len=96,
                         hidden_size=6,
                         num_layers=2,
                         batch_size=4).to(device)
        return self.model      


    def train(self, x_train, y_train, x_val, y_val):###df包含label
        self.model = self.build_model(x_train)
        # 先转换成torch能识别的dataset
        import torch.utils.data as data
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        setattr(self.scaler_y, 'mu_inverse_scale', (y_train.max()-y_train.min())*y_train.sum()/(y_train.sum()-len(y_train)*y_train.min()))
        setattr(self.scaler_y, 'std_inverse_scale',y_train.max()-y_train.min())
                
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3,
                                     weight_decay=0)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
        #                             momentum=0.9, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in tqdm(range(10)):
            # 训练步骤开始
            self.model.train()
            for x,y in train_loader:
                loss = 0
                hidden = self.model.init_hidden(self.model.batch_size)##lstm的hidden和cell初始化
                cell = self.model.init_cell(self.model.batch_size)
                x=x.to(device)
                y=y.to(device)          
                for t in range(x.shape[1]):
                    mu, std, hidden, cell = self.model(x[:,t,:].unsqueeze(1), hidden, cell)                        
                    loss += self.log_prob_loss_fn(mu, std, y[:,t,:])
                
                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                hidden = hidden.detach()
                cell = cell.detach()
                optimizer.step()
            print('loss:',loss)
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
            hidden = self.model.init_hidden(testset.tensors[0].shape[0])##lstm的hidden和cell初始化
            cell = self.model.init_cell(testset.tensors[0].shape[0])
            x=x.to(device)
            y=y.to(device)     
            mus=[]
            stds=[]
            for t in range(x.shape[1]):
                mu, std, hidden, cell = self.model(x[:,t,:].unsqueeze(1), hidden, cell)   
                mus.append(mu)    
                stds.append(std)    
            mus=torch.concat(mus,axis=0)
            mus=mus*self.scaler_y.mu_inverse_scale
            mus=mus.cpu().detach().numpy().reshape(-1,1)
            stds=torch.concat(stds,axis=0)
            stds=stds*self.scaler_y.std_inverse_scale
            stds=stds.cpu().detach().numpy().reshape(-1,1)
            
            y=y.cpu().numpy().reshape(-1,1)
            res=abs(mus-y)/y
            print('test mape:',float(1-res.mean()))   
        
        pred=mus
        pred_upper=mus+2*stds
        pred_lower=mus-2*stds
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
    
    deepAR=DeepAR()
    deepAR.train(x_train, y_train, x_val, y_val)
    deepAR.test(x_test,y_test,deepAR.model)