import numpy as np

def evaluation(pred,gt,method='mse'):
    if method=='mse':
        res=np.mean((pred-gt)**2)
    if method=='rmse':
        res=np.mean((pred-gt)**2)**(0.5)
    elif method=='mape':
        res=np.mean(1-abs(pred-gt)/gt)
    elif method=='mae':
        res=np.mean(abs(pred-gt))   
    return res