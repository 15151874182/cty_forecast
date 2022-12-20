import numpy as np

def mse_train(gt, pred): ##xgboost,lgb自定义损失函数
    diff = (gt - pred).astype("float")
    grad = np.where(diff>=0, -2*diff, -2*diff)
    hess = np.where(diff>=0, 2.0, 2.0)
    return grad, hess    