'''
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from model.xgb import XGB
from model.lgb import LGB
from model.lstm import LSTM
from config import lgb_param
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

####code may be used
# df.rename(columns={'rttower_load':'load'}, inplace=True)
# df=df[['date','load','speed_50_XXL','dir_50_XXL']]
# df['date']=pd.to_datetime(df['date'])
# testset=df.loc[(df['date'] >= pd.to_datetime('2022-5-31')) & (df['date'] < pd.to_datetime('2022-8-23'))]

####Step1:read raw data
np.random.seed(10)
f='./data/wind/ghgr.csv'
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

####Step5: 自动调参
# def objective(trial):
#     lgb=LGB(trial=trial,param=lgb_param)
#     trainded_model=lgb.train(x_train, y_train, x_val, y_val)
#     pred,gt=lgb.test(x_val,y_val,trainded_model)
#     loss=evaluation(pred.values, gt.values)
#     return loss

# # study=optuna.create_study(direction='maximize')
# study=optuna.create_study(direction='minimize')
# n_trials=50 # try50次
# study.optimize(objective, n_trials=n_trials)

####Step6: 使用优化超参训练+推断
lgb=LGB(trial=False,param=lgb_param)
trainded_model=lgb.train(x_train, y_train, x_val, y_val)
pred_old,gt=lgb.test(x_test,y_test,trainded_model)
old_loss=mean_squared_error(pred_old.values, gt.values)
print('old_loss:',old_loss)
res=pd.concat([pred_old.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
res.columns=['pred_old','gt']
from utils.plot import plot_without_date
plot_without_date(res,'res',cols = ['pred_old','gt'])

# new_lgb_param=lgb_param.copy()
# new_lgb_param.update(study.best_params) ##更新超参
# lgb=LGB(trial=False,param=new_lgb_param)
# trainded_model=lgb.train(x_train, y_train, x_val, y_val)
# pred_new,gt=lgb.test(x_test,y_test,trainded_model)
# new_loss=evaluation(pred_new.values, gt.values)
# print('new_loss:',new_loss)

# res=pd.concat([pred_old.reset_index(drop=True),pred_new.reset_index(drop=True),gt.reset_index(drop=True)],axis=1)
# res.columns=['pred_old','pred_new','gt']
# from utils.plot import plot_without_date
# plot_without_date(res,'res',cols = ['pred_old','pred_new','gt'])