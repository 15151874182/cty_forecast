'''
将预测出来的pred和gt值拼接，算mape，赋予列名，形成最终的res
'''
import numpy as np
import pandas as pd

def predGt_to_res(test_date_list,pred,gt):
    ###test_date_list时间
    res=pd.concat([test_date_list,gt,pred],axis=1)
    res.columns=['date','gt_max_load','pred_max_load']
    res['mape']=1-abs(res['pred_max_load']-res['gt_max_load'])/res['gt_max_load']
    res['average_mape']=res['mape'].mean()
    return res