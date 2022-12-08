'''
SHAP特征分析
'''
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestRegressor
# load JS visualization code to notebook
class SHAP():
    
    def __init__(self,trained_model,df,label_name,explainer_type='tree'):
        self.df2=df.copy()
        self.y = self.df2.pop(label_name)  
        self.x = self.df2
        
        if explainer_type=='tree':         
            self.explainer = shap.TreeExplainer(trained_model)    
        '''TODO:
            deep：用于计算深度学习模型，基于DeepLIFT算法
            gradient：用于深度学习模型，综合了SHAP、集成梯度、和SmoothGrad等思想，形成单一期望值方程
            kernel：模型无关，适用于任何模型
            linear：适用于特征独立不相关的线性模型
            tree：适用于树模型和基于树模型的集成算法
            sampling ：基于特征独立性假设，当你想使用的后台数据集很大时，kenel的一个很好的替代方案
        if explainer_type=='deep':         
            self.explainer = shap.DeepExplainer(...)  
        
        ......
        '''
        self.shap_values = self.explainer.shap_values(self.x) 
        self.shap_object = self.explainer(self.x) 
        
    def summary_plot(self,flag=[0,2]):####flag是想数组切片，想显示哪几行数据
        if flag=='all':
            shap.summary_plot(self.shap_object, self.x)
        else:
            shap.summary_plot(self.shap_object[flag[0]:flag[1]], self.x[flag[0]:flag[1]])
        
    def bar_plot(self):
        ###全局条形图
        shap.plots.bar(self.shap_object,max_display=100)
    
    def bar_plot2(self,n):
        ###局部条形图
        ##将一行 SHAP 值传递给条形图函数会创建一个局部特征重要性图
        shap.plots.bar(self.shap_object[n],max_display=100)

    # def bar_plot3(self,n):   暂时有问题，且没看懂，后期可以再研究
    #     ###队列条形图
    #     ##队列条形图还有另一个比较有意思的绘图，他使用 Explanation 对象的自动群组功能来使用决策树创建一个群组。调用Explanation.cohorts(N)
    #     ##将创建 N 个队列，使用 sklearn DecisionTreeRegressor 最佳地分离实例的 SHAP 值。
    #     v = self.shap_object.cohorts(n).abs.mean(0)
    #     shap.plots.bar(v,max_display=30)
    
    def beeswarm_plot(self):
        ###蜂群图旨在显示数据集中的TOP特征如何影响模型输出的信息密集摘要。
        shap.plots.beeswarm(self.shap_object,order=self.shap_values.abs.max(0))

    def force_plot(self,X_display,n):
        ###SHAP force plot 提供了单一模型预测的可解释性，可用于误差分析，找到对特定实例预测的解释
        shap.force_plot(self.explainer.expected_value,self.shap_values[n,:],
                        X_display.iloc[n,:],matplotlib=True)
        
    def dependence_plot(self,fea1,fea2,X_display):
        shap.dependence_plot(fea1, self.shap_values, self.x, 
                     display_features=X_display,
                     interaction_index=fea2)    
    def scatter_plot(self,fea1,fea2,X_display):
        self.shap_object.display_data = X_display.values
        shap.plots.scatter(self.shap_object[:, fea1], 
                   color=self.shap_object[:,fea2])
        
    def cluster(self):####特征太多没法运行
        clustering = shap.utils.hclust(self.x, self.y) 
        shap.plots.bar(self.shap_object,clustering=clustering,clustering_cutoff=0.5)  
        
    def interaction_values(self):
        ###是将SHAP值推广到更高阶交互的一种方法。树模型实现了快速、精确的两两交互计算，这将为每个预测返回一个矩阵，其中主要影响在对角线上，交互影响在对角线外。这些数值往往揭示了有趣的隐藏关系
        shap_interaction_values = explainer.shap_interaction_values(self.x)
        shap.summary_plot(shap_interaction_values, self.x)    
        
    def decision_plot(self,features_display):
        ###SHAP 决策图显示复杂模型如何得出其预测（即模型如何做出决策）
        expected_value = self.explainer.expected_value
        # 限制20个样本
        features = self.x.iloc[range(20)]
        # 展示第一条样本
        shap_values = self.explainer.shap_values(features)[1]
        
        shap.decision_plot(expected_value, shap_values, 
                           features_display)