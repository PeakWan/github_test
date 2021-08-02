#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020/11/19 by Owen Yang
##hz 修改机器学习验证集和测试集结果一致问题  ML_Classfication


外部可引用函数:

    特征分析选择:
        func::principle_component_analysis:

        func::feature importance:
            regression based:       Lasso, Ridge
            classification based:   xgboost(classifier)

        func::model_score_with_features_add:
            classification based:   xgboost classifier, Logistic regression, SVC, 


    数据建模:
        func::ML_Classification:
            'LogisticRegression',
            'XGBClassifier',
            'RandomForestClassifier',
            'SVC',
            'KNeighborsClassifier',

        func::ML_Regression:
            'LinearRegression',
            'XGBRegressor',
            'RandomForestRegressor',
            'LinearSVR',
            'KNeighborsRegressor',

        func::ML_Clustering:
            'KMeans',
            'Birch',
            'SpectralClustering',
            'AgglomerativeClustering',
            'GaussianMixture',            


    模型对比:
        funct::two_groups_classfication_multimodels:
            'XGBClassifier'
            'LogisticRegression',
            'SVC',
            'MLPClassifier',
            'RandomForestClassifier',
            'AdaBoostClassifier',
            # 'KNeighborsClassifier',
            # 'DecisionTreeClassifier',
            # 'GradientBoostingClassifier',
            # 'BaggingClassifier',
            # 'ExtraTreesClassifier',



内部函数:

    供 func::feature importance 使用:

        func::_lasso_features_importance
        func::_ridge_features_importance
        func::_xgboost_features_importance


自动寻参:
    
    对于监督式学习(分类和回归)：
        GridSearcherCV: 利用交叉验证为选择标准，对所有可能参数构型进行遍历搜索
        RandSearcherCV: 利用交叉验证为选择标准，对所有可能参数构型进行随机搜索     

    对于非监督式学习(聚类)：
        GridSearcherSelf: 利用给定模型选择标准(11/19：轮廓系数)，对所有可能参数构型进行随机搜索  
        RandSearcherSelf: 利用给定模型选择标准(11/19：轮廓系数)，对所有可能参数构型进行随机搜索  

    是否自动寻参作为每个函数的传递参量，默认为否。
    目前支持自动寻参的函数有：
        0. features_importance (11/19)
        1. ML_Classification (11/19)
        2. ML_Regression (11/19)
        3. ML_Clustering (11/19)
        4. two_groups_classfication_multimodels (11/19)
    尚不支持自动寻参的是：
        1. model_score_with_features_add




2020/12/12: 所有本文件中函数
            输入参数统一增加 savePath 用来存储图片
            返回形式统一为：
                    df_dict, str_result, plot_name_list
            分别为：dataframe从名字到内容的字典，字符串输出，图片文件名列表
            
2020/12/31: 
    1. features_importance 入参增加 bool::standardization (数据标准化, default = True)
    2. model_score_with_features_add 入参增加 bool::importance_first (先进行XGBoost重要度排序，default = True)
    3. two_groups_classfication_multimodels 入参删除 4 个 (ylim_min, ylim_max, title1, title2)，后端views.py已修改
    4. 所有图片保存前增填 savePath != None 的检查; savePath = None 作为内部调用使用，不额外保存图片
    

2021/01/13:
    1. 所有函数增加变量筛选(调用 utils_ml.ML_assistant 中的 filtering 函数)


2021/02/09:
    1.  two_class_multimodel 森林图调用X5 forest_plot()


2021/05/05:
    前端修改要求：
    feature_importance 的 Advanced/views.py 中， model_type 与方法的对应编号修改如下:
        1 -- 回归/Lasso
        2 -- 回归/岭回归
        3 -- 回归/XGBoost
        4 -- 回归/随机森林
        5 -- 回归/AdaBoost
        6 -- 分类/Logistic
        7 -- 分类/XGBoost
        8 -- 分类/随机森林
        9 -- 分类/AdaBoost

    Classification 增加一个 dataframe
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB


from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from sklearn.metrics import auc
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mutual_info_score, v_measure_score, normalized_mutual_info_score


from xgboost import plot_importance
import AnalysisFunction.X_5_SmartPlot as x5
from AnalysisFunction.X_1_DataGovernance import data_standardization

from AnalysisFunction.utils_ml import filtering, dic2str, round_dec, save_fig
from AnalysisFunction.utils_ml import classification_metric_evaluate, regression_metric_evaluate
from AnalysisFunction.utils_ml import make_class_metrics_dict, make_regr_metrics_dict, multiclass_metric_evaluate
from AnalysisFunction.utils_ml import ci

from AnalysisFunction.utils_ml import GridSearcherCV, RandSearcherCV, GridSearcherSelf, RandSearcherSelf


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号




#-------------------------------
#-----------重要度排序----------
#-------------------------------
def features_importance(
        df_input,
        group,
        features,
        top_features = 10,
        model_type = 1,
        standardization = True,
        savePath = None,
        searching=True,
        decimal_num = 3,
    ):
    """
        重要度排序

        Input:
            df_input:DataFrame 输入的待处理数据
            group:str 因变量
            features:list 自变量
            top_features:图表中展示的特征数量
            model_type:int 使用的模型编号
                        '_lasso_features_importance':LassoCV(),
                        '_ridge_features_importance':RidgeV(),
                        'xgboost_features_importance':XGBClassifier(),
            standardization:bool (2020/12/31) 是否先对数据进行标准化处理，默认为True,
            searching:bool 是否进行自动寻参，默认为True
                           (2020/11/30: lasso和ridge因为用了cv，属于自带寻参，因此目前只在xgboost时寻参)
            savePath:str 图片存储路径
            
            filter_conditions:tuple(行间关系, 行条件) 
                行间关系 : "或者", "并且", "无"
                行条件 : 

        Return:
            df_dict:  模型performance综述
            str_result: 模型及参数信息
            plot_name_list: 图片名列表
    """
    df_temp = df_input[features+[group]].dropna().reset_index().drop(columns='index')
    if standardization :
        df_temp, _, _, _, _, _ = data_standardization(df_temp,features,method='StandardScaler')

    if model_type > 5 and len(pd.unique(df_temp[group])) > 5:
        return {'error' : '暂不允许类别数目大于5的情况。请检查因变量取值情况。'}

    if model_type == 1:
        df_result,str_result, plot_name_list = _lasso_rfeatures_importance(
            df_temp, features, group, top_features, savePath,
        )
    elif model_type == 2:
        df_result, str_result, plot_name_list = _ridge_rfeatures_importance(
            df_temp, features, group, top_features, savePath,
        )
    elif model_type == 3:
        df_result,str_result, plot_name_list = _xgboost_rfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    elif model_type == 4:
        df_result,str_result, plot_name_list = _randomforest_rfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    elif model_type == 5:
        df_result,str_result, plot_name_list = _adaboost_rfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    elif model_type == 6:
        df_result,str_result, plot_name_list = _logisticL1_cfeatures_importance(
            df_temp, features, group, top_features, savePath,
        )
    elif model_type == 7:
        df_result,str_result, plot_name_list = _xgboost_cfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    elif model_type == 8:
        df_result,str_result, plot_name_list = _randomforest_cfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    elif model_type == 9:
        df_result,str_result, plot_name_list = _adaboost_cfeatures_importance(
            df_temp, features, group, top_features, searching, savePath,
        )
    plt.close()

    if model_type <= 5:
        str_result += '接下来可使用这些相关度较高的变量进行，利用左侧栏‘机器学习回归’进一步的细致化回归建模。'
    else:
        str_result += '接下来可使用这些相关度较高的变量进行，利用左侧栏‘机器学习分类’进一步的细致化分类建模。'
    
    df_result = df_result.applymap(lambda x: round_dec(x, d=decimal_num))
    return {'重要度排序' : df_result}, str_result, plot_name_list



#--------------------------------------
#-----------变量集评分分析-------------
#--------------------------------------
"""
df_input:Dataframe
features:list
group：str
v_round:int 迭代次数
scoring：str
model：model

hyper params: model.get_params()
"""
def model_score_with_features_add(
        df_input, 
        group, 
        features, 
        v_round=20, 
        scoring='roc_auc',
        model_type=1,
        importance_first=True,
        savePath = None,
        searching=False,
        decimal_num = 3,
    ):
    """
        变量集评分分析

        Input:
            df_input:DataFrame 输入的待处理数据
            group:str 因变量
            features:list 自变量名
            v_round:int 交叉验证次数
            scoring:str 目标评价指标
            model:int 使用的模型编号
                        1 --> 'XGBClassifier':XGBClassifier(),
                        2 --> 'LogisticRegression':LogisticRegression(),
                        3 --> 'SVC':SVC(),
            importance_first:bool (2020/12/31) 是否先对features进行重要度排序，默认为True
            searching:bool 是否进行自动寻参，默认为False；
                           (11/30: 目前不支持, 因为每加入一个变量寻参的结果都会不同, 没有一般意义下的最优参数; 
                                   可以随时取消注释行开启, 彼时模型参数输出显示将无意义.
                            )
            savePath:str 图片存储路径

        Return:
            df_dict:  模型performance综述
            str_result: 模型及参数信息
            plot_name_list: 图片名列表
    """
    if len(features) > 15 :
        return {'error' : "不支持变量数大于15的情况。请先用重要度排序进行变量筛选。"}

    dftemp = df_input[features + [group]].dropna()

    if len(pd.unique(dftemp[group])) > 2: 
        if scoring == 'roc_auc': 
            scoring = scoring + '_ovo'
        else:
            scoring = scoring + '_macro'

    str_result = ''
    plot_name_list = []

    if (model_type ==1):
        model = XGBClassifier()
        str_result += '采用极端梯度提升树(XGBOOST)进行变量集评分，模型参数为\n' + dic2str(model.get_params(), model.__class__.__name__)
    elif (model_type == 2):
        model = LogisticRegression()
        str_result += '采用逻辑回归(LogisticRegression)进行变量集评分，模型参数为\n' + dic2str(model.get_params(), model.__class__.__name__)
    elif (model_type == 3):
        model = SVC(probability=True)
        str_result += '采用支持向量机(SVM)进行变量集评分，模型参数为\n' + dic2str(model.get_params(), model.__class__.__name__)
    # searcher = RandSearcherCV('Classification', model)
    
    if importance_first :
        temp_dict, _, plot_name_list = features_importance(
            dftemp,
            group,
            features,
            top_features = len(features),
            model_type = 2,
            standardization = True,
            savePath = savePath,
            searching=True,
        )
        features = list(temp_dict['重要度排序']["Variable"])

    columns_size = len(features)
    y = dftemp[group]
    num_features = []
    features_list = []
    train_mean = []
    train_se = []
    test_mean = []
    test_se = []
    cv = ShuffleSplit(
        train_size = 0.8,
        test_size = 0.2,
        n_splits = v_round,
    )

    for i in range(1,columns_size+1,1):
        num_features.append(i)
        x = dftemp[features[:i]]
        # if searching:
        #     model = searcher(x, y); searcher.report()

        cv_result = cross_validate(
            model, x, y,
            cv = cv,
            scoring = scoring,
            return_train_score=True,
        )
        features_list.append(features[:i])
        test_mean.append(cv_result['test_score'].mean())
        test_se.append(cv_result['test_score'].var())
        train_mean.append(cv_result['train_score'].mean())
        train_se.append(cv_result['train_score'].var())

    df_result=pd.DataFrame()
    df_result['变量数'] = num_features
    df_result['训练集均值'] = train_mean
    df_result['训练集SD'] = train_se
    df_result['测试集均值'] = test_mean
    df_result['测试集SD'] = test_se
    df_result['模型中的变量'] = features_list

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.set_ylim([0.0, 1.0])
    ax.grid(which='major', axis='y', linestyle='-.')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.tick_params(bottom=False,labelbottom=True)

    x = np.arange(1, columns_size+1, 1)
    ax.errorbar(x, train_mean, fmt="d--",ecolor=x5.CB91_Grad_BP[0], color=x5.CB91_Grad_BP[0], yerr=train_se, label='训练集')
    ax.errorbar(x, test_mean, fmt="d--", ecolor=x5.CB91_Grad_BP[1], color=x5.CB91_Grad_BP[1], yerr=test_se, label='测试集')
    ax.set_xlabel('Number of Variables', fontsize=15)
    ax.set_ylabel(scoring, fontsize=15)
    ax.set_title('Model Performance')
    ax.legend(loc="lower right", ncol=3, fontsize=5)

    
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, '变量数vs建模表现', '.jpeg', fig))
    plt.close()

    all_var = df_result.tail(1).iloc[0]['模型中的变量']
    best_ = df_result.sort_values(by='测试集均值', ascending=False).head(1)
    str_result += "\n评分过程将变量集{0}中的变量按顺序加入（此集合中变量顺序默认为先验估计得到的重要性排序），在包含{1}个变量时得到最佳表现。最佳的变量集为：{2}。".format(
        all_var, best_.iloc[0]['变量数'], best_.iloc[0]['模型中的变量']
    )
    
    df_result = df_result.applymap(lambda x: round_dec(x, d=decimal_num))
    return {'变量数vs评分' : df_result}, str_result, plot_name_list




def ML_Classfication(
        df,
        group,
        features,
        decimal_num=3,
        validation_ratio=0.15,
        scoring='roc_auc',
        method='KNeighborsClassifier',
        n_splits=10,
        savePath = None,
        searching=False,
        **kwargs,
    ):
    """
        机器学习分类分析

        Input:
            df_input:DataFrame 输入的待处理数据
            group_name:str 分组名
            validation_ratio:float 测试集比例
            scoring:str 目标评价指标
            method:str 使用的机器学习分类方法/模型
                        'LogisticRegression':LogisticRegression(**kwargs),
                        'XGBClassifier':XGBClassifier(**kwargs),
                        'RandomForestClassifier':RandomForestClassifier(**kwargs),
                        'SVC':SVC(**kwargs),
                        'KNeighborsClassifier':KNeighborsClassifier(**kwargs),
            n_splits:int 交叉验证的子集数目
            searching:bool 是否进行自动寻参，默认为否
            savePath:str 图片存储路径
            **kwargs:dict 使用机器学习分类方法的参数

        Return:
            df_dict: dataframe字典，包含：
                    df_train_result: 模型在训练集上的表现
                    df_test_result:  模型在测试集上的表现
            str_result: 分析结果综述
            plot_name_list: 图片文件名列表
    """
    list_name = [group]
    df = df[features + [group]].dropna()
    
    binary = True
    if len(pd.unique(df[group])) > 2:
        if len(pd.unique(df[group])) > 10:
            return {'error' : '暂不允许类别数目大于10的情况。请检查因变量取值情况。'}
        binary = False
        if scoring == 'roc_auc': 
            scoring = scoring + '_ovo'
        else:
            scoring = scoring + '_macro'

    X = df.drop(group, axis=1)
    Y = df.loc[:, list_name].squeeze(axis=1)
    Xtrain, Xtest, Ytrain, Ytest = TTS(
        X, Y, 
        test_size = validation_ratio, 
        random_state = 0,
    )

    str_result = "采用%s机器学习方法进行分类，分类变量为%s，模型中的变量包括"%(method, group)
    str_result += '、'.join(features)

    if searching:
        searcher = RandSearcherCV('Classification', globals()[method]())
        clf = searcher(Xtrain, Ytrain); searcher.report()
    else:
        if (method == 'SVC'): kwargs['probability'] = True
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    str_result += "\n模型参数为:\n%s"%dic2str(clf.get_params(), clf.__class__.__name__)
    str_result += "\n数据集样本数总计N=%d例，应变量中包含的类别信息为：\n"%(df.shape[0])
    group_labels = df[group].unique()
    group_labels.sort()
    for label in group_labels:
        n = sum(df[group] == label)
        str_result += "\t 类别("+ str(label)+")：N="+str(n)+"例\n"


    plot_name_list = x5.plot_learning_curve(
        clf,
        Xtrain,
        Ytrain,
        cv = 10, 
        scoring = scoring,
        path = savePath,
    )

    if binary:
        fig = plt.figure(figsize=(4, 4), dpi=600)
        # 画对角线
        plt.plot(
            [0, 1], [0, 1], 
            linestyle='--', 
            lw=1, color='r', 
            alpha = 0.8,
        )
        plt.grid(which='major', axis='both', linestyle='-.', alpha = 0.08, color='grey')

    best_auc = 0.0
    tprs_train, tprs_valid = [], []
    mean_fpr = np.linspace(0, 1, 100)
    list_evaluate_dic_train = make_class_metrics_dict()
    list_evaluate_dic_valid = make_class_metrics_dict()
    KF = KFold(n_splits = n_splits)
    for i, (train_index, valid_index) in enumerate(KF.split(Xtrain)):
        # 划分训练集和验证集
        X_train, X_valid = Xtrain.iloc[train_index], Xtrain.iloc[valid_index]
        Y_train, Y_valid = Ytrain.iloc[train_index], Ytrain.iloc[valid_index]

        # 建立模型(模型已经定义)并训练
        model = clone(clf).fit(X_train, Y_train)

        # 利用classification_metric_evaluate函数获取在验证集的预测值
        fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(model, X_train, Y_train, binary)
        fpr_valid, tpr_valid, metric_dic_valid, _ = classification_metric_evaluate(model, X_valid, Y_valid, binary)

        # model selection using validation set
        if metric_dic_valid['AUC'] > best_auc :
            clf = model

        # 计算所有评价指标
        for key in list_evaluate_dic_train.keys():
            list_evaluate_dic_train[key].append(metric_dic_train[key])
            list_evaluate_dic_valid[key].append(metric_dic_valid[key])

        if binary:
            # interp:插值 把结果添加到tprs列表中
            tprs_valid.append(np.interp(mean_fpr, fpr_valid, tpr_valid))
            tprs_valid[-1][0] = 0.0

            # 画图, 只需要plt.plot(fpr,tpr), 变量roc_auc只是记录auc的值, 通过auc()函数计算出来
            plt.plot(
                fpr_valid, tpr_valid, 
                lw=1, alpha=0.4, 
                label='ROC fold %4d (auc=%0.3f)'%(i+1, metric_dic_valid['AUC']),
            )
    
    if binary:
        mean_tpr_valid = np.mean(tprs_valid, axis=0)
        mean_tpr_valid[-1] = 1.0    
        mean_auc = auc(mean_fpr, mean_tpr_valid)  # 计算平均AUC值
        aucs_lower, aucs_upper = ci(list_evaluate_dic_valid['AUC'])
        plt.plot(
            mean_fpr, mean_tpr_valid, 
            color = 'b',
            lw = 2, alpha = 0.8,
            label = r'Mean (validation) ROC (auc=%0.3f)' % (mean_auc), 
            # label = r'Mean ROC (auc=%0.3f 0.95CI(%0.3f-%0.3f)' % (mean_auc, aucs_lower, aucs_upper), 
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC')
        plt.legend(loc='lower right', fontsize=5)
        if savePath is not None:
            plot_name_list.append(save_fig(savePath, 'ROC_curve', '.jpeg', fig))
        plt.close()


    mean_dic_train, stdv_dic_train = {}, {}
    mean_dic_valid, stdv_dic_valid = {}, {}
    for key in list_evaluate_dic_valid.keys():
        mean_dic_train[key] = np.mean(list_evaluate_dic_train[key])
        stdv_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
        mean_dic_valid[key] = np.mean(list_evaluate_dic_valid[key])
        stdv_dic_valid[key] = np.std(list_evaluate_dic_valid[key], axis=0)
    df_train_result = pd.DataFrame([mean_dic_train, stdv_dic_train], index=['Mean', 'SD'])
    df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))
    df_valid_result = pd.DataFrame([mean_dic_valid, stdv_dic_valid], index=['Mean', 'SD'])
    df_valid_result = df_valid_result.applymap(lambda x: round_dec(x, d=decimal_num))

    _, _, _, df_test_result = classification_metric_evaluate(clf, Xtest, Ytest, binary)
    df_test_result  = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))

    df_dict = {
        '训练集结果汇总' : df_train_result,
        '验证集结果汇总' : df_valid_result,
        '测试集结果汇总' : df_test_result,
    }


    str_result += "其中在总体样本中随机抽取测试集N=%d例(%3.2f%%)，剩余样本作为训练集进行%d折交叉验证，并在验证集中得到AUC=%5.4f±%5.4f。\n最终模型在测试集中的AUC=%5.4f，准确度=%5.4f。\n"%(
        df.shape[0] * validation_ratio, 
        validation_ratio * 100, 
        n_splits,
        df_valid_result['AUC'].values[0], 
        df_valid_result['AUC'].values[1], 
        df_test_result['AUC'].values[0],
        df_test_result['准确度'].values[0],
    )
    
    diff = df_valid_result.loc['Mean', 'AUC'] - df_test_result.loc['Mean', 'AUC']
    ratio = diff/df_test_result.loc['Mean', 'AUC']
    if (diff>0 and (ratio > 0.1)):
        str_result += '注意到AUC指标下验证集表现超出测试集{}，约{}%，可能存在过拟合现象。建议更换模型或重新设置参数。'.format(diff, ratio*round_dec(100.0, d=decimal_num))
    else:
        str_result += '鉴于AUC指标下验证集表现未超出测试集或超出比小于10%，可认为拟合成功，{}模型可以用于此数据集的分类建模任务。'.format(method)
    str_result += '\n如果想进一步对比更多分类模型的表现，可使用左侧栏智能分析中的‘分类多模型综合分析’功能。'


    return df_dict, str_result, plot_name_list


def ML_Regression(
        df_input,
        group,
        features,
        decimal_num=3,
        validation_ratio=0.15,
        scoring='r2',
        method='LinearRegression',
        n_splits=10,
        savePath = None,
        searching=False,
        **kwargs,
    ):
    """
        机器学习回归分析

        Input: 
            df_input:DataFrame 输入的待处理数据
            group_name:str 分组名
            validation_ratio:float 测试集比例
            scoring:str 目标评价指标
            method:str 使用的机器学习回归方法/模型
                        'LinearRegression':LinearRegression(**kwargs),
                        'XGBRegressor':XGBRegressor(**kwargs),
                        'RandomForestRegressor':RandomForestRegressor(**kwargs),
                        'LinearSVR':LinearSVR(**kwargs),
                        'KNeighborsRegressor':KNeighborsRegressor(**kwargs),
            n_splits:int 交叉验证的子集数目
            searching:bool 是否进行自动寻参，默认为否
            savePath:str 图片存储路径
            **kwargs:dict 使用机器学习回归方法的参数

        Return:
            df_dict: dataframe字典，包含：
                    df_train_result: 模型在训练集上的表现
                    df_test_result:  模型在测试集上的表现
            str_result: 分析结果综述
            plot_name_list: 图片文件名列表
    """

    list_name = [group]
    df_temp = df_input[features + [group]].dropna()
    
    X = df_temp.drop(group, axis=1)
    Y = df_temp.loc[:, list_name].squeeze(axis=1)
    Xtrain, Xtest, Ytrain, Ytest = TTS(
        X, Y, 
        test_size = validation_ratio, 
        random_state=0,
    )

    if searching:
        searcher = RandSearcherCV('Regression', globals()[method]())
        clf = searcher(Xtrain, Ytrain); searcher.report()
    else:
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    plot_name_list = []

    plot_name_list = x5.plot_learning_curve(
        clf, 
        Xtrain, 
        Ytrain, 
        cv=10, 
        scoring=scoring,
        path = savePath,
    )
        
    list_evaluate_dic = make_regr_metrics_dict()
    KF = KFold(n_splits=n_splits)
    cv_pred = np.asarray([])
    for train_index, test_index in KF.split(Xtrain):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_valid = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        Y_train, Y_valid = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        # 建立模型(模型已经定义)并训练
        estimator = clone(clf).fit(X_train, Y_train)
        cv_pred = np.append(cv_pred, estimator.predict(X_valid))
        # 利用regression_metric_evaluate函数获取在测试集的预测值
        df_result, metric_dic = regression_metric_evaluate(estimator, X_valid, Y_valid)

        for key in list_evaluate_dic.keys():
            list_evaluate_dic[key].append(metric_dic[key])

    for key in list_evaluate_dic.keys():
        metric_dic[key] = np.mean(list_evaluate_dic[key])
        list_evaluate_dic[key] = np.std(list_evaluate_dic[key], axis=0)

    df_train_result = pd.DataFrame([metric_dic, list_evaluate_dic], index=['Mean', 'SD'])
    df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))

    df_test_result, metric_dic = regression_metric_evaluate(clf, Xtest, Ytest)
    df_test_result  = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))


    # 绘制训练集10折交叉验证预测值-真实值曲线
    fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=600)
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=600)
    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=600)

    ax1.scatter(Ytrain, cv_pred, edgecolors='none', marker='*', color=x5.CB91_Blue, s=50.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(which='major', axis='both', linestyle='-.', alpha=0.2)
    ax1.spines['left'].set_position('center')
    ax1.spines['bottom'].set_position('center')
    ax1.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=2)
    ax1.set_xlabel('True', fontsize='x-small', loc='right')
    ax1.set_ylabel('Predicted', fontsize='x-small', loc='top')
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'Range Comparison', '.jpeg', fig1))

    pred = clf.predict(Xtrain)
    # 画出原始值的曲线和模型的预测线
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.tick_params(bottom=False)
    ax2.grid(which='major', axis='y', linestyle='--')
    ax2.plot(np.arange(Xtrain.shape[0]), Ytrain, color='k', label='True')
    ax2.plot(np.arange(Xtrain.shape[0]), pred, color='r', label=method+'Predicted')
    ax2.set_xlabel('Data Samples', fontsize='x-small', loc='center')
    ax2.set_ylabel('Value', fontsize='x-small', loc='top')
    ax2.legend(loc='lower right')
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'True_vs_Predicted', '.jpeg', fig2))

    # 残差图
    ax3.scatter(Ytrain, pred - Ytrain.values[0], marker='X', edgecolors='none', color=x5.CB91_Violet, s=50.0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines['left'].set_position('center')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.grid(which='major', axis='y', linestyle='-.')
    ax3.set_xlabel('True Value', fontsize='x-small', loc='right')
    ax3.set_ylabel('Residual', fontsize='x-small', loc='top')
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'Residual', '.jpeg', fig3))


    str_result = "采用%s机器学习方法进行回归，分类变量为%s，模型中的变量包括" % (method, group)
    str_result += '、'.join(features)
    str_result += "\n模型参数为:\n%s" % dic2str(clf.get_params(), clf.__class__.__name__)
    str_result += "\n数据集样本数总计N=%d例，" % (df_temp.shape[0])
    str_result += "其中在总体样本中随机抽取测试集N=%d例(%3.2f%%)，剩余样本作为训练集进行%d折交叉验证，最终模型在验证集中的R方=%3.2f±%3.2f,在测试集中的R方=%3.2f。" % (
        df_temp.shape[0] * validation_ratio, 
        validation_ratio * 100, 
        n_splits,
        df_train_result['R方'].values[0], 
        df_train_result['R方'].values[1],
        df_test_result['R方'].values[0],
    )
    diff = df_train_result.loc['Mean', 'R方'] - df_test_result.loc['Mean', 'R方']
    ratio = diff/df_test_result.loc['Mean', 'R方']
    if (diff>0 and (ratio > 0.1)):
        str_result += '注意到R方指标下验证集表现超出测试集{}，约{}%，可能存在过拟合现象。建议更换模型或重新设置参数。'.format(diff, ratio*round_dec(100.0, d=decimal_num))
    else:
        str_result += '鉴于R方指标下验证集表现未超出测试集或超出比小于10%，可认为拟合成功，{}模型可以用于此数据集的回归建模任务。'.format(method)
    # str_result += '如果想进一步对比更多回归模型的表现，可使用左侧栏智能分析中的‘回归多模型综合分析’功能。'

    df_dict = {
        '验证集结果汇总' : df_train_result,
        '测试集结果汇总' : df_test_result,
    }
    plt.close()
    return df_dict, str_result, plot_name_list


def ML_Clustering(
        df_input,
        features,
        group=None,
        decimal_num=3,
        method='KMeans',
        searching=False,
        savePath=None,
        **kwargs,
    ):
    """
        机器学习聚类分析

        Input:
            df_input：DataFrame输入的待处理数据
            feature:list 特征
            group_name:str 已有类标签
            method:str 使用的机器学习聚类方法/模型
                        'KMeans':KMeans(**kwargs),
                        'Birch':Birch(**kwargs),
                        'SpectralClustering':SpectralClustering(**kwargs),
                        'AgglomerativeClustering':AgglomerativeClustering(**kwargs),
                        'GaussianMixture':GaussianMixture(**kwargs),
            n_splits:int 交叉验证的子集数目
            searching:bool 是否进行自动寻参，默认为否
            savePath:str 图片存储路径
            **kwargs:dict 使用机器学习聚类方法的参数

        Return:
            
            df_dict: dataframe字典，包含：
                    df_result: 聚类分析模型的评分
                    df_output: 聚类分析的标注结果
            str_result: 分析结果综述
            plot_name_list: 图片文件名列表
    """
    null = np.nan

    X_0 = df_input[features].dropna().copy()
    X = StandardScaler().fit_transform(X_0)
    
    if searching:
        searcher = RandSearcherSelf('Clustering', globals()[method]())
        clf = searcher(X); searcher.report()
    else:
        clf = globals()[method](**kwargs)
        if (method in ['SpectralClustering', 'GaussianMixture',]):
            clf.set_params(**{'n_components' : 2})

    if (method in ['SpectralClustering', 'AgglomerativeClustering',]):
        labels = clf.fit_predict(X)
    else:
        clf.fit(X)
        labels = clf.predict(X)
    metric_dic = {
        '轮廓系数': silhouette_score(X, labels),
        '卡林斯基-哈拉巴斯指数': calinski_harabasz_score(X, labels),
    }

    if group!=None:
        Y = df_input.loc[:, group]
        metric_dic['互信息'] = mutual_info_score(Y, labels)
        metric_dic['V-measure'] = v_measure_score(Y, labels)


    df_result = pd.DataFrame(metric_dic,index=[0])
    df_label  = pd.DataFrame(labels, columns=['类别'])
    df_output = pd.concat([df_input,df_label],axis=1)

    str_result = "采用%s机器学习方法进行聚类，其中参数为:\n%s"%(method, dic2str(clf.get_params(), clf.__class__.__name__))
    group_labels = df_label['类别'].unique()
    group_labels.sort()
    str_result += "\n此数据集中有效样本总数为N={0}。聚类模型将所有样本分为C={1}个数据簇类别。每一类别分别包含样本数量如下：\n".format(X_0.shape[0], len(group_labels))
    for label in group_labels:
        n = sum(df_label['类别'] == label)
        str_result += "\t 类别("+ str(int(label))+")：N="+str(n)+"例\n"
    str_result += '聚类分析中所得的轮廓系数为：%3.2f。'%(metric_dic['轮廓系数'])

    if group is None:
        str_result += '由于没有提供可参照的类别标注，这一聚类标注结果可以作为样本数据基于变量集{0}的初步分类。'.format(features)
    else:
        str_result += '用户已将{0}选定为类别标注的参照结果，与此参照变量相比较，二者的V-测度值（算数归一化互信息）为：{1}'.format(group, metric_dic['V-measure'])
        if (metric_dic['V-measure'] >= 0.5):
            str_result += " >= 0.5，可认为是较合理的聚类结果。{}模型在此数据集聚类任务中可以使用。".format(method)
        else:
            str_result += " < 0.5，聚类结果较参照变量相差较大。需要斟酌使用{}模型提供的这一聚类标注，同时可以尝试其他聚类方法或重新设定参数。".format(method)

    df_result = df_result.applymap(lambda x: round_dec(x, d=decimal_num))

    df_dict = {
        '聚类分析表现汇总' : df_result,
        '聚类分析数据标注结果': df_output,
    }
    plt.close()
    return df_dict, str_result, []


#-------------------------------------------------------------
#----------------------分类多模型综合分析------------------------
#-------------------------------------------------------------
def two_groups_classfication_multimodels(
        df_input,
        group,
        features,
        methods = [],
        decimal_num=3,
        testsize=0.2, 
        boostrap=5,
        searching=False,
        savePath=None,
    ):
    """
        df_input:Dataframe
        features:自变量list
        group：因变量str
        testsize: 测试集比例
        boostrap：重采样次数
        searching:bool 是否进行自动寻参，默认为否
        savePath:str 图片存储路径
    """
    dftemp = df_input[features + [group]].dropna()

    x = dftemp[features]
    y = dftemp[group]

    if len(y.unique()) > 2:
        return {'error' : '暂时只支持二分类。请检查因变量取值情况。'}


    name_dict = {
        'LogisticRegression' : 'logistic',
        'XGBClassifier' : 'XGBoost',
        'RandomForestClassifier' : 'RandomForest',
        'SVC' : 'SVM',
        'MLPClassifier' : 'MLP',
        'GaussianNB': 'GaussianNB',
        'AdaBoostClassifier' : 'AdaBoost',  

        'KNeighborsClassifier' : 'KNN',        
        'DecisionTreeClassifier' : 'DecisionTree',
        'BaggingClassifier' : 'Bagging',
    }
    if len(methods) == 0 :
        methods = [
            'LogisticRegression',
            'XGBClassifier',
            'RandomForestClassifier',
            # 'SVC',
            # 'MLPClassifier',
            # 'AdaBoostClassifier',
            # 'KNeighborsClassifier',
            # 'DecisionTreeClassifier',
            # 'BaggingClassifier',
        ]
    str_result = '已采用多种机器学习模型尝试完成数据样本分类任务，包括：{}。各模型的参数值选取情况如下所示：\n\n'.format(methods)

    plot_name_list = []

    fig, ax = plt.subplots(figsize=(6,6))

    # 画对角线
    ax.plot(
        [0, 1], [0, 1], 
        linestyle='--', 
        lw=1, color='r', 
        alpha = 1.0,
    )
    ax.grid(which='major', axis='both', linestyle='-.', alpha = 0.3, color='grey')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.tick_params(top=False, right=False)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC Curve')

    mean_fpr = np.linspace(0, 1, 100)
    colors = x5.CB91_Grad_BP
    
    df_0 = pd.DataFrame(columns=list(make_class_metrics_dict().keys()), index=[0])
    df_0_test = df_0.copy()

    df_plot = pd.DataFrame(columns = ['method', 'mean', 'std'])

    for i, method in enumerate(methods):
        tprs_train, tprs_test = [], []

        if searching:
            searcher = RandSearcherCV('Classification', globals()[method]())
            selected_model = searcher(x, y)#; searcher.report()
        else:
            selected_model = globals()[method]() if (method != 'SVC') else globals()[method](probability=True)

        list_evaluate_dic_train = make_class_metrics_dict()
        list_evaluate_dic_test = make_class_metrics_dict()

        for index in range(0, boostrap):
            Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=testsize)
            model = clone(selected_model).fit(Xtrain, Ytrain)
            
            # 利用classification_metric_evaluate函数获取在测试集的预测值
            fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(model, Xtrain, Ytrain)
            fpr_test,  tpr_test,  metric_dic_test,  _ = classification_metric_evaluate(model, Xtest,  Ytest)
            
            # interp:插值 把结果添加到tprs列表中
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
            tprs_train[-1][0] = 0.0
            tprs_test[-1][0] = 0.0
            
            # 计算所有评价指标
            for key in list_evaluate_dic_train.keys():
                list_evaluate_dic_train[key].append(metric_dic_train[key])
                list_evaluate_dic_test[key].append(metric_dic_test[key])

        for key in list_evaluate_dic_train.keys():
            metric_dic_train[key] = np.mean(list_evaluate_dic_train[key])
            metric_dic_test[key]  = np.mean(list_evaluate_dic_test[key])
            list_evaluate_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
            list_evaluate_dic_test[key]  = np.std(list_evaluate_dic_test[key], axis=0)
        df_train_result = pd.DataFrame([metric_dic_train, list_evaluate_dic_train], index = ['Mean', 'SD'])
        df_test_result  = pd.DataFrame([metric_dic_test,  list_evaluate_dic_test],  index = ['Mean', 'SD'])
        df_train_result['分类模型']= method
        df_test_result['分类模型'] = method

        df_0 = pd.concat([df_0, df_train_result])
        df_0_test = pd.concat([df_0_test, df_test_result])

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_test  = np.mean(tprs_test, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_tpr_test[-1]  = 1.0
        mean_auc_train = auc(mean_fpr, mean_tpr_train)  # 计算训练集平均AUC值
        mean_auc_test  = auc(mean_fpr, mean_tpr_test)

        df_plot = df_plot.append({
            'method': name_dict[method], 
            'mean' : mean_auc_test, 
            'std' : list_evaluate_dic_test['AUC'],
        }, ignore_index=True)
        ax.plot(mean_fpr, mean_tpr_test, c=colors[i], label=method+'(AUC = %0.2f)' % (mean_auc_test) , lw=1.5, alpha=1)
        str_result += method + ': AUC='+str(mean_auc_train) + ';  模型参数:\n'+dic2str(selected_model.get_params(), method)+'\n'


    ymin = min([y-dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    ymax = max([y+dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    ymin, ymax = ymin - (ymax - ymin)/4.0, ymax + (ymax - ymin)/10.0
    ax.legend(loc="lower right")
    
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'ROC_curve', '.jpeg', fig))
        plot_name_list += x5.forest_plot(
            df_input = df_plot,
            name = 'method', value = 'mean', err = 'std', direct = 'horizontal',
            fig_size=[len(methods)+3, 6],
            ylim = [ymin, ymax],
            title = 'Forest Plot of Each Model AUC Score ',
            path = savePath,
        )
    plt.close()

    df_train_result1= df_0.drop([0])
    df_test_result1 = df_0_test.drop([0])

    classfier = df_train_result1.pop('分类模型')

    df_train_result= df_train_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_train_result.insert(0, '分类模型', classfier)
    
    df_test_result1.pop('分类模型')
    df_test_result=df_test_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_test_result.insert(0,'分类模型',classfier)    

    df_dict = {
        '多模型分类-训练集结果汇总': df_train_result,
        '多模型分类-验证集结果汇总': df_test_result,
    }

    str_result +='\n下示森林图展示了各模型进行'+group+'预测的ROC结果,图中的误差线为ROC均值及SD。\n'\
               +'模型的ROC均值及SD的是通过多次重复采样计算，重复采样次数为'+str(boostrap)+'次,'\
               +'每一次重采样训练的验证集占总体样本的'+str(testsize*100)+'%,训练集占'\
               +str((1-testsize)*100)+'%,'+'模型中的变量包括'\
               +','.join(features)+'。\n'

    best_ = df_train_result.loc[df_train_result.index == 'Mean'].sort_values(by='AUC', ascending=False).head(1)
    name_train = best_.iloc[0]['分类模型']
    str_result += '在目前所有模型中，训练集表现最佳者为{}（依据AUC排序），在各评价标准中其在验证集对应分数分别为：\n'.format(name_train)
    for col in best_.columns[1:]:
        str_result += '\t{}：{}\n'.format(col, best_.iloc[0][col])
    
    best_ = df_test_result.loc[df_test_result.index == 'Mean'].sort_values(by='AUC', ascending=False).head(1)
    name_test = best_.iloc[0]['分类模型']
    str_result += '验证集表现最佳者为{}（依据AUC排序），在各评价标准中其在测试集对应分数分别为：\n'.format(name_test)
    for col in best_.columns[1:]:
        str_result += '\t{}：{}\n'.format(col, best_.iloc[0][col])
    
    if (name_test == name_train) :
        str_result += '二者吻合，可以认为{}是针对此数据集的最佳模型选择。'.format(name_train)
    else:
        str_result += '二者不吻合，{}极可能存在过拟合现象，{}可能稳定性相对较好。具体模型选择可根据下表详细评分信息进行取舍。'.format(name_train, name_test)

    return df_dict, str_result, plot_name_list




#====================================================
""" 
    INTERNAL FUNCTIONS PART
"""

#----------回归重要度排序-----------
def _lasso_rfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    alpharange = np.logspace(-10, -2, 200, base=10)
    
    lasso_ = LassoCV(alphas=alpharange,cv=5).fit(x, y)
    param_dict = lasso_.get_params()
    param_dict['alpha']  = lasso_.alpha_
    str_result = '采用lasso Regressor进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, lasso_.__class__.__name__)
    
    c1={'Variable':x_columns, 'Weight Importance':abs(lasso_.coef_)}
    a1=pd.DataFrame(c1)    
    df_result = a1.sort_values(by="Weight Importance",ascending=False)
    
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])
    str_result += '\n注意：在使用Lasso进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。'

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True), 
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list


def _ridge_rfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    alpharange = np.logspace(-10, -2, 200, base=10)

    Ridge_ = RidgeCV(alphas=alpharange,cv=5).fit(x, y)
    param_dict = Ridge_.get_params()
    param_dict['alpha']  = Ridge_.alpha_
    str_result = '采用Ridge Regressor进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, Ridge_.__class__.__name__)
    
    c1={'Variable':x_columns, 'Weight Importance':abs(Ridge_.coef_)}
    a1=pd.DataFrame(c1)
    df_result = a1.sort_values(by="Weight Importance",ascending=False)
    
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])
    str_result += '\n注意：在使用Ridge进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。'

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list


def _xgboost_rfeatures_importance(
        df_input, 
        x_columns, 
        y_column, 
        top_features, 
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        top_features:图表中展示的特征数量
        model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
        searching: 是否自动寻参，默认为是
        savePath:str 图片存储路径

        hyperparams: XGBClassifier params -- no selection yet
    """
    x=df_input[x_columns]
    y=df_input[y_column]

    if searching:
        searcher = RandSearcherCV('Regression', XGBRegressor())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = XGBRegressor()
    str_result = '采用XGBoost进行变量重要度分析，模型参数为:\n' + dic2str(model.get_params(), model.__class__.__name__)
    model.fit(x, y)
    
    col_refs = {
        'Variable': 'total_gain',
        'Total Gain': 'total_gain', 
        'Total Cover': 'total_cover',
        'Gain': 'gain', 
        'Cover': 'cover', 
        'Weight Importance': 'weight', 
    }
    df = pd.DataFrame(columns=list(col_refs.keys()))
    
    for col_name, importance_type in col_refs.items():
        row_index = 0
        for d, x in model.get_booster().get_score(importance_type=importance_type).items():
            df.loc[row_index, col_name] = x if (col_name!='Variable') else d
            row_index += 1
    df = df.sort_values(by='Total Gain',ascending=False).head(top_features)
    top_list = list(df["Variable"])
        
    str_result += '\n重要度最高的{}个变量（由高到低）分别为：{}。'.format(top_features, str(top_list)[1:-1])
    
    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df.head(top_features).sort_values(by="Total Gain", ascending=True),
            'Variable', 
            'Total Gain',
            'Feature Importance (Total Gain)',
            savePath,
        )

    return df, str_result, plot_name_list


def _randomforest_rfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandSearcherCV('Regression', RandomForestRegressor())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = RandomForestRegressor().fit(x, y)
    param_dict = model.get_params()

    str_result = '采用Random Forrest Regressor进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, model.__class__.__name__)
        
    df_result = pd.DataFrame({
        'Variable':x_columns, 
        'Weight Importance':abs(model.feature_importances_)
    }).sort_values(by="Weight Importance",ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list


def _adaboost_rfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandSearcherCV('Regression', AdaBoostRegressor())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = AdaBoostRegressor().fit(x, y)
    param_dict = model.get_params()

    str_result = '采用AdaBoost Regressor进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, model.__class__.__name__)
        
    df_result = pd.DataFrame({
        'Variable':x_columns, 
        'Weight Importance':abs(model.feature_importances_)
    }).sort_values(by="Weight Importance",ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list



#----------分类重要度排序-----------
def _logisticL1_cfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    crange = np.logspace(-9, 1, 200, base=10)
    
    logicv_ = LogisticRegressionCV(Cs=crange, cv=5, penalty='l1', solver='saga').fit(x, y)
    param_dict = logicv_.get_params()
    param_dict['C'] = logicv_.C_
    str_result = '采用L1正则化的Logistic回归进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, logicv_.__class__.__name__)
    
    df_result = pd.DataFrame({
        'Variable':x_columns, 
        'Weight Importance':abs(logicv_.coef_[0])
    }).sort_values(by="Weight Importance",ascending=False)
    
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])
    str_result += '\n注意：在使用Logistiv+L1进行重要度排序时，由于其指数部分的线性形式，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。'

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list


def _xgboost_cfeatures_importance(
        df_input, 
        x_columns, 
        y_column, 
        top_features, 
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        top_features:图表中展示的特征数量
        model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
        searching: 是否自动寻参，默认为是
        savePath:str 图片存储路径

        hyperparams: XGBClassifier params -- no selection yet
    """
    x=df_input[x_columns]
    y=df_input[y_column]

    if searching:
        searcher = RandSearcherCV('Classification', XGBClassifier())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = XGBClassifier()
    str_result = '采用极端梯度提升树(XGBOOST)进行变量重要度分析，模型参数为:\n' + dic2str(model.get_params(), model.__class__.__name__)
    model.fit(x, y)
    
    col_refs = {
        'Variable': 'total_gain',
        'Total Gain': 'total_gain', 
        'Total Cover': 'total_cover',
        'Gain': 'gain', 
        'Cover': 'cover', 
        'Weight Importance': 'weight', 
    }
    df = pd.DataFrame(columns=list(col_refs.keys()))
    
    for col_name, importance_type in col_refs.items():
        row_index = 0
        for d, x in model.get_booster().get_score(importance_type=importance_type).items():
            df.loc[row_index, col_name] = x if (col_name!='Variable') else d
            row_index += 1
    df = df.sort_values(by='Total Gain',ascending=False).head(top_features)
    top_list = list(df["Variable"])
        
    str_result += '\n重要度最高的{}个变量（由高到低）分别为：{}。'.format(top_features, str(top_list)[1:-1])
    
    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df.head(top_features).sort_values(by="Total Gain", ascending=True),
            'Variable', 
            'Total Gain',
            'Feature Importance (Total Gain)',
            savePath,
        )

    return df, str_result, plot_name_list


def _randomforest_cfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandSearcherCV('Classification', RandomForestClassifier())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = RandomForestClassifier().fit(x, y)
    param_dict = model.get_params()

    str_result = '采用Random Forrest Classifier进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, model.__class__.__name__)
        
    df_result = pd.DataFrame({
        'Variable':x_columns, 
        'Weight Importance':abs(model.feature_importances_)
    }).sort_values(by="Weight Importance",ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list


def _adaboost_cfeatures_importance(
        df_input,
        x_columns,
        y_column,
        top_features,
        searching = True,
        savePath = None,
    ):
    """
        df_input:Dataframe
        x_columns:自变量list
        y_column：因变量str
        num_features:图表中展示的特征数量
        (会剔除空值)
        savePath:str 图片存储路径

        hyperparams: alpha(alpharange), cv, tol
    """
    dftemp=df_input[x_columns+[y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandSearcherCV('classification', AdaBoostClassifier())
        model = searcher(x, y)#; searcher.report()
    else: 
        model = AdaBoostClassifier().fit(x, y)
    param_dict = model.get_params()

    str_result = '采用AdaBoost Classifier进行变量重要度分析，模型参数为:\n' + dic2str(param_dict, model.__class__.__name__)
        
    df_result = pd.DataFrame({
        'Variable':x_columns, 
        'Weight Importance':abs(model.feature_importances_)
    }).sort_values(by="Weight Importance",ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += '\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n'.format(top_features, str(top_list)[1:-1])

    plot_name_list = []
    if savePath is not None:
        plot_name_list = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(by="Weight Importance", ascending=True),
            'Variable', 
            'Weight Importance',
            'Feature Importance (Coefficient)',
            savePath,
        )
    return df_result, str_result, plot_name_list





