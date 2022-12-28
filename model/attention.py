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

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class Encoder(nn.Module):
    def __init__(self, fea_dim):
        super().__init__()
        self.fea_dim = fea_dim 
        self.pos_embedding = nn.Embedding(100, fea_dim)
        self.attention = MultiHeadAttentionLayer(fea_dim, n_heads=1, dropout=0.1)
        self.fc = nn.Linear(fea_dim, 1)
        
    def forward(self,x):
        batch_size = x.shape[0]
        src_len = x.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # x=x+self.pos_embedding(pos)
        query=key=value=x
        output,_ = self.attention(query, key, value, mask = None)
        pred = self.fc(output)
        return pred


class Attention():
    def __init__(self,seq_len):
        self.seq_len=seq_len
        
    def build_model(self,x_train):
        self.enc = Encoder(fea_dim=x_train.shape[1]).to(device)

    def train(self, x_train, y_train, x_val, y_val):###df包含label
        self.build_model(x_train)
        self.enc.train()      
        # 先转换成torch能识别的dataset
        import torch.utils.data as data
        
        self.scaler_x = MinMaxScaler(feature_range=(0.1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0.1, 1))
        x_train = self.scaler_x.fit_transform(x_train)
        y_train = self.scaler_y.fit_transform(y_train[:,None])      
        x_val = self.scaler_x.transform(x_val)
        y_val = self.scaler_y.transform(y_val[:,None])

        x=torch.Tensor(x_train.reshape(-1,self.seq_len,x_train.shape[1]))
        y=torch.Tensor(y_train.reshape(-1,self.seq_len,1))
        trainset = data.TensorDataset(x, y)
        train_loader = data.DataLoader(dataset=trainset,
                                       batch_size=x.shape[0],
                                       shuffle=False, ##x_train已经打乱过了
                                       num_workers=1,
                                       drop_last=True)
        
        x=torch.Tensor(x_val.reshape(-1,self.seq_len,x_val.shape[1]))
        y=torch.Tensor(y_val.reshape(-1,self.seq_len,1))        
        valset = data.TensorDataset(x, y)
        val_loader = data.DataLoader(dataset=valset,
                                     batch_size=x.shape[0],
                                     shuffle=False, ##x_val已经打乱过了
                                     num_workers=1,
                                     drop_last=True)        
        loss_fn = nn.MSELoss()
        # import itertools 
        optimizer = torch.optim.Adam(self.enc.parameters(), lr=1e-2,
                                     weight_decay=0)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
        #                             momentum=0.9, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        
        for epoch in tqdm(range(3000)):
            # 训练步骤开始

            for x,y in train_loader:
                preds = []
                x=x.to(device)
                y=y.to(device)

                preds = self.enc(x)
                loss = loss_fn(preds, y)      
                # print(loss)
                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            res=abs(preds-y)/y
            print('train mape:',float(1-res.mean()))
            
            # preds=preds.detach().cpu().numpy().reshape(-1,1)
            # y=y.cpu().numpy().reshape(-1,1)            
            # preds=pd.DataFrame(preds)
            # y=pd.DataFrame(y.reshape(-1))
            # res=pd.concat([preds,y],axis=1)
            # res.columns=['preds','gt']
            # from my_utils.plot import plot_without_date
            # plot_without_date(res[:600],'res',cols = ['preds','gt'])         

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

    def test(self,x_test, y_test):  
        self.enc.eval()
        x_test = self.scaler_x.transform(x_test)
        import torch.utils.data as data
        x=torch.Tensor(x_test.reshape(-1,self.seq_len,x_test.shape[1]))
        y=torch.Tensor(y_test.values.reshape(-1,self.seq_len,1))  
        testset = data.TensorDataset(x, y)
        test_loader = data.DataLoader(dataset=testset,
                                       batch_size=x.shape[0], ##相当于整个batch
                                       shuffle=False, ##x_train已经打乱过了
                                       num_workers=1,
                                       drop_last=False)    
        for x,y in test_loader:

            x=x.to(device)
            y=y.to(device)
            preds = self.enc(x)
            preds=preds.detach().cpu().numpy().reshape(-1,1)
            preds=self.scaler_y.inverse_transform(preds)
            y=y.cpu().numpy().reshape(-1,1)
            res=abs(preds-y)/y
            print('test mape:',float(1-res.mean()))   
        preds=pd.DataFrame(preds)
        y=pd.DataFrame(y.reshape(-1))
        res=pd.concat([preds,y],axis=1)
        res.columns=['preds','gt']
        from my_utils.plot import plot_without_date
        plot_without_date(res[:600],'res',cols = ['preds','gt']) 
        self.res=res
        
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
    
    attention=Attention(seq_len=96)
    attention.train(x_train, y_train, x_val, y_val)
    attention.test(x_test,y_test)