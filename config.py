lgb_param = {
    'boosting_type':'gbdt',
    'class_weight':None, 
    'colsample_bytree':1.0, 
    'device':'cpu',
    'importance_type':'split', 
    'learning_rate':0.044,
    'max_depth':10, 
    'min_child_samples':91, 
    'min_child_weight':0.001,
    'min_split_gain':0.2, 
    'n_estimators':140, 
    'n_jobs':-1, 
    'num_leaves':31,
    'objective':None, 
    'random_state':1822, 
    'reg_alpha':0.9, 
    'reg_lambda':0.6,
    'silent':True, 
    'subsample':0.4, 
    'subsample_for_bin':200000,
    'subsample_freq':0
}     

xgb_param = {
    'booster': 'gbtree',
    'n_estimators': 170,
    'max_depth': 2,
    'learning_rate': 0.0621,
    'subsample': 0.8,
    'gamma': 0.3,
    'reg_alpha': 0.9,
    'reg_lambda': 0.2,
    'nthread': 8,
    'objective': 'reg:squarederror',
    # 'objective': self.__fix_mse_obj__,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'gpu_id': -1,
    'tree_method': 'auto',
}

arima_param=(1,1,1)