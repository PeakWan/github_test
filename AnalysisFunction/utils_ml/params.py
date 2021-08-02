"""
2020/12/14 by Owen Yang

这一文件只用来存储和模型参数相关的信息，包含：
    1. ToShow: 用来现实的模型参数列表
    2. GridDefaultRange: 用来进行网格遍历式搜索的参数空间
    3. RandDefaultRange: 用来进行随机式搜索的参数空间

"""

import numpy as np


############################################################################
#################       for parameter return      ##########################
############################################################################

ToShow = {}

ToShow['LassoCV'] = {
    'max_iter' : '迭代次数',
    'tol' : '收敛度量',
    'cv' : '交叉验证折数',
    'alpha' : 'L1正则化系数',
}

ToShow['RidgeCV'] = {
    'cv' : '交叉验证折数',
    'alpha' : 'L2正则化系数',
}


ToShow['KMeans'] = {
    'n_clusters' : '团簇中心个数', 
    'max_iter' : '迭代次数',
    'tol' : '收敛度量',
}

ToShow['Birch'] = {
    'n_clusters' : '团簇中心个数',
    'threshold' : '可融合范围',
    'branching_factor' : '最大分支数',
}

ToShow['SpectralClustering'] = {
    'n_clusters' : '团簇中心个数',
    'affinity' : '近邻矩阵类型',
}

ToShow['AgglomerativeClustering'] = {
    'n_clusters' : '团簇中心个数', 
    'affinity' : '近邻矩阵类型', 
    'linkage' : '团簇距离度量',
}

ToShow['GaussianMixture'] = {
    'n_components' : '团簇中心个数',
    'max_iter' : '迭代次数',
    'tol' : '收敛度量',
}


ToShow['XGBRegressor'] = {
    'objective' : '优化目标函数',
    'learning_rate' : '学习速率',
    'max_depth' : '最大树深度',
    'min_child_weight' : '最小分叉权重和',
    'reg_lambda' : 'L2正则化系数',
}

ToShow['XGBClassifier'] = {
    'objective' : '优化目标函数',
    'learning_rate' : '学习速率',
    'max_depth' : '最大树深度',
    'min_child_weight' : '最小分叉权重和',
    'reg_lambda' : 'L2正则化系数',
}


ToShow['RandomForestRegressor'] = {
    'n_estimators' : '树数目',
    'max_depth' : '最大树木深度',
    'criterion' : '度量指标',
}

ToShow['RandomForestClassifier'] = {
    'n_estimators' : '树数目',
    'max_depth' : '最大树木深度',
    'criterion' : '度量指标',
    'min_impurity_decrease' : '最小分叉纯度收益',
}


ToShow['SVC'] = {
    'kernel' : '核类型',
    'C' : '正则化因子',
    'tol' : '收敛度量',
}

ToShow['LinearSVR'] = {
    'C' : '正则化因子',
    'tol' : '收敛度量',
    'epsilon' : 'epsilon',
    'max_iter' : '迭代次数',
}


ToShow['KNeighborsClassifier'] = {
    'n_neighbors' : '近邻数目',
    'weights' : '权重类型', 
}

ToShow['KNeighborsRegressor'] = {
    'n_neighbors' : '近邻数目',
    'weights' : '权重类型', 
}


ToShow['LogisticRegression'] = {
    'C' : '正则化因子',
    'tol' : '收敛度量',
    'max_iter' : '迭代次数',
    'penalty' : '正则化类型',
}
ToShow['LogisticRegressionCV'] = {
    'C' : '正则化因子',
    'tol' : '收敛度量',
    'max_iter' : '迭代次数',
    'penalty' : '正则化类型',
}

ToShow['LinearRegression'] = {
    'fit_intercept' : '拟合截距'
}


ToShow['AdaBoostRegressor'] = {
    'learning_rate' : '学习速率',
    'n_estimators' : '单模型数目',
}
ToShow['AdaBoostClassifier'] = {
    'learning_rate' : '学习速率',
    'n_estimators' : '单模型数目',
}

ToShow['MLPClassifier'] = {
    'activation' : '非线性函数',
    'hidden_layer_sizes' : '隐藏层宽度',
    'max_iter' : '迭代次数',
}



############################################################################
#################       for parameter tuning      ##########################
############################################################################

GridDefaultRange = {}


### clustering models
### For these models (as unsupervised learning), cv-based method is not applicable
### Instead, hyperparam search could use silhouette_score and information based criteria.
GridDefaultRange['KMeans'] = {
    'n_clusters' : [2, 4, 6, 8],
    'tol' : [1e-4, 1e-2, 1e-1],
## fixed part
    'algorithm': ['auto',],
    'copy_x': [True,],
    'init': ['k-means++',],
    'max_iter': [300,],
    'n_init': [10,],
    'random_state': [None,],
    'verbose': [0,],    
}

GridDefaultRange['Birch'] = {
    'n_clusters': [2, 4, 6, 8],
    'threshold': [0.2, 0.5, 1.0,],
    'branching_factor': [25, 50, 75],
## fixed part
    'compute_labels': [True,],
    'copy': [True,],
}

GridDefaultRange['SpectralClustering'] = {
    'n_clusters': [2, 4, 8],
    'affinity': ['rbf', 'nearest_neighbors'],
## fixed part
    'assign_labels': ['kmeans',],
    'coef0': [1,],
    'degree': [3,],
    'eigen_solver': [None,],
    'eigen_tol': [0.0,],
    'gamma': [1.0,],
    'kernel_params': [None,],
    'n_components': [2,],
    'n_init': [10,],
    'n_jobs': [None,],
    'n_neighbors': [4,],
    'random_state': [None,],    
}

GridDefaultRange['AgglomerativeClustering'] = [
## for large number of clusters
    {'n_clusters': [2, 4, 8], 'affinity': ['euclidean'], 'linkage': ['ward'], }, 
## for sparse features (l1) or scale-invariant feature (cos)
    {'n_clusters': [2, 4, 8], 'affinity': ['cosine', 'l1'], 'linkage': ['average'], }, 
# ## for efficient computation with large dataset
#   {'n_clusters': [10, 20, 40], 'affinity': ['euclidean', 'l1'], 'linkage': 'single', 'compute_full_tree': True},
]

## model.bic()
GridDefaultRange['GaussianMixture'] = {
    'covariance_type': ['spherical', 'tied', 'diag', 'full'],
    'n_components': [2, 4, 6, 8, 10],
## fixed part
    'init_params': ['kmeans',],
    'max_iter': [100,],
    'means_init': [None,],
    'n_init': [1,],
    'precisions_init': [None,],
    'random_state': [None,],
    'reg_covar': [1e-06,],
    'tol': [0.001,],
    'verbose': [0,],
    'verbose_interval': [10,],
    'warm_start': [False,],
    'weights_init': [None,],
}



### classification/regression models
### For these models, cross-validation based hyperparam search can be implemented
GridDefaultRange['XGBRegressor'] = {
    'reg_lambda': [0.5, 1,],
    'max_depth': [4, 8,],
    'min_child_weight': [2, 4, 6],
    'learning_rate': [0.1, 0.3,],
## fixed part
    'base_score': [0.5,],
    'booster': ['gbtree',],
    'colsample_bylevel': [1,],
    'colsample_bynode': [1,],
    'colsample_bytree': [1,],
    'gamma': [0,],
    'gpu_id': [-1,],
    'importance_type': ['gain',],
    'max_delta_step': [0,],
    'missing': [np.nan,],
    'monotone_constraints': [(),],
    'n_estimators': [100,],
    'n_jobs': [0,],
    'num_parallel_tree': [1,],
    'random_state': [0,],
    'reg_alpha': [0,],
    'scale_pos_weight': [1,],
    'subsample': [1,],
    'tree_method': ['exact',],
    'validate_parameters': [1,],
    'verbosity': [0,],   
}
GridDefaultRange['XGBClassifier'] = {
    'reg_lambda': [0.5, 1,],
    'max_depth': [4, 8,],
    'min_child_weight': [2, 4, 6],
    'learning_rate': [0.1, 0.3,],
## fixed part
    'base_score': [0.5,],
    'booster': ['gbtree',],
    'colsample_bylevel': [1,],
    'colsample_bynode': [1,],
    'colsample_bytree': [1,],
    'gamma': [0,],
    'gpu_id': [-1,],
    'importance_type': ['gain',],
    'max_delta_step': [0,],
    'missing': [np.nan,],
    'monotone_constraints': [(),],
    'n_estimators': [100,],
    'n_jobs': [0,],
    'num_parallel_tree': [1,],
    'random_state': [0,],
    'reg_alpha': [0,],
    'scale_pos_weight': [1,],
    'subsample': [1,],
    'tree_method': ['exact',],
    'validate_parameters': [1,],
    'verbosity': [0,],
}


GridDefaultRange['RandomForestRegressor'] = {
## fixed part
    'bootstrap': [True,],
    'ccp_alpha': [0.0,],
    'criterion': ['mse',],
    'max_depth': [None,],
    'max_features': ['auto',],
    'max_leaf_nodes': [None,],
    'max_samples': [None,],
    'min_impurity_decrease': [0.0,],
    'min_impurity_split': [None,],
    'min_samples_leaf': [1,],
    'min_samples_split': [2,],
    'min_weight_fraction_leaf': [0.0,],
    'n_estimators': [100,],
    'n_jobs': [None,],
    'oob_score': [False,],
    'random_state': [None,],
    'verbose': [0,],
    'warm_start': [False,], 
}
GridDefaultRange['RandomForestClassifier'] = {
## fixed part
    'bootstrap': [True,],
    'ccp_alpha': [0.0,],
    'class_weight': [None,],
    'criterion': ['gini',],
    'max_depth': [None,],
    'max_features': ['auto',],
    'max_leaf_nodes': [None,],
    'max_samples': [None,],
    'min_impurity_decrease': [0.0,],
    'min_impurity_split': [None,],
    'min_samples_leaf': [1,],
    'min_samples_split': [2,],
    'min_weight_fraction_leaf': [0.0,],
    'n_estimators': [100,],
    'n_jobs': [None,],
    'oob_score': [False,],
    'random_state': [None,],
    'verbose': [0,],
    'warm_start': [False,], 
}


GridDefaultRange['SVC'] = {
    'C' : [1.0, 0.1],
    'tol': [1e-3, 1e-1],
## fixed part
    'kernel': ['rbf',],
    'break_ties': [False,],
    'cache_size': [200,],
    'class_weight': [None,],
    'coef0': [0.0,],
    'decision_function_shape': ['ovr',],
    'degree': [3,],
    'gamma': ['scale',],
    'max_iter': [100,],
    'probability': [True,],
    'random_state': [None,],
    'shrinking': [True,],
    'verbose': [False,],    
}
GridDefaultRange['LinearSVR'] = {
    'C' : [1.0, 0.1],
    'tol': [1e-4, 1e-6],
## fixed part
    'dual': [True,],
    'epsilon': [0.0,],
    'fit_intercept': [True,],
    'intercept_scaling': [1.0,],
    'loss': ['epsilon_insensitive',],
    'max_iter': [100,],
    'random_state': [None,],
    'verbose': [0,],    
}


GridDefaultRange['KNeighborsClassifier'] = {
    'n_neighbors' : [2, 4, 6],
    'weights': ['uniform', 'distance'], 
    'leaf_size': [10, 30, 50],
## fixed part
    'algorithm': ['auto',],
    'metric': ['minkowski',],
    'metric_params': [None,],
    'n_jobs': [None,],
    'p': [2,],
}
GridDefaultRange['KNeighborsRegressor'] = {
    'n_neighbors' : [2, 4, 6],
    'weights': ['uniform', 'distance'], 
    'leaf_size': [10, 30, 50],
## fixed part
    'algorithm': ['auto',],
    'metric': ['minkowski',],
    'metric_params': [None,],
    'n_jobs': [None,],
    'p': [2,],
}




GridDefaultRange['LogisticRegression'] = {
    'C' : np.logspace(-4, 1, 6, base=10),
    'tol': [1e-4, 1e-6],
## fixed part
    'class_weight': [None,],
    'dual': [False,],
    'fit_intercept': [True,],
    'intercept_scaling': [1,],
    'l1_ratio': [None,],
    'max_iter': [100,],
    'multi_class': ['auto',],
    'n_jobs': [None,],
    'penalty': ['l2',],
    'random_state': [None,],
    'solver': ['lbfgs',],
    'verbose': [0,],
    'warm_start': [False,], 
}

GridDefaultRange['LinearRegression'] = {
## fixed part
    'copy_X': [True,],
    'fit_intercept': [True,],
    'n_jobs': [None,],
    'normalize': [False,],
}


GridDefaultRange['AdaBoostRegressor'] = {
    'learning_rate': [0.1, 0.3, 1.0,],
## fixed part
    'base_estimator': [None,],
    'n_estimators': [50,],
    'random_state': [None,],
}
GridDefaultRange['AdaBoostClassifier'] = {
    'learning_rate': [0.1, 0.3, 1.0,],
## fixed part
    'algorithm': ['SAMME.R',],
    'base_estimator': [None,],
    'n_estimators': [50,],
    'random_state': [None,],
}


GridDefaultRange['MLPClassifier'] = {
## fixed part
    'activation': ['logistic',],
    'alpha': [0.0001,],
    'batch_size': ['auto',],
    'beta_1': [0.9,],
    'beta_2': [0.999,],
    'early_stopping': [False,],
    'epsilon': [1e-08,],
    'hidden_layer_sizes': [(60, 10),],
    'learning_rate': ['constant',],
    'learning_rate_init': [0.001,],
    'max_fun': [15000,],
    'max_iter': [200,],
    'momentum': [0.9,],
    'n_iter_no_change': [10,],
    'nesterovs_momentum': [False,],
    'power_t': [0.5,],
    'random_state': [None,],
    'shuffle': [True,],
    'solver': ['adam',],
    'tol': [0.0001,],
    'validation_fraction': [0.1,],
    'verbose': [False,],
    'warm_start': [False,], 
}



RandDefaultRange = GridDefaultRange
