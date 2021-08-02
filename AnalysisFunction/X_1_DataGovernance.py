# V 1.0.14
# date 2021-3-20
# author：xuyuan
# V1.0.1 更新说明：剔除相关性（get_var_low_colinear）增加了str_result初始赋值；
# 修改了分组重编码：直接赋值改为copy深拷贝；
# 修改了数据平衡，对于空值填补0
# V1.0.2 更新说明:修改了_round_dec 对inf 和nan 数据进行了判断
# V1.0.3 更新说明:修改了get_var_low_vif 默认仅对定量数据进行处理
# V1.0.4 更新说明:修改了abnormal_deviation_process图片，改为调用X5的方法
# V1.0.5 更新说明:修改了get_var_low_colinear方法
# V1.0.6 更新说明:修改了data_transform方法
# V1.0.7 更新说明:修改了group_recoding方法,增加入参group_range_value
# V1.0.8 更新说明:修改了miss_data_filling方法 新增入参path 新增返回值plot_name_list
# V1.0.9 更新说明:优化了相关性提出，优化了分组重编码
# V1.0.10 更新说明：优化了分组重编码
# v1.0.11 更新说明：优化了分组重编码
# v1.0.12 更新说明：优化了内部函数_dataframe_to_categorical
# v1.0.13 更新说明：优化了哑变量分组、分组重编码
# v1.0.14 更新说明：_round_dec增加参数pvalue
# v1.0.15 更新说明：psm匹配增加参数precious

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import patsy
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import AnalysisFunction.X_5_SmartPlot as x5
import random
import warnings
import math


warnings.filterwarnings('ignore')

imputation_method = {'mean': '均值', 'median': '中位数', 'most_frequent': '众数', 'zero': '零值',
                     'nan': '空值', 'randomforest': '随机森林', 'interpolate': '插值法', 'KNN': 'KNN'}
balance_method = {'SMOTE': 'SMOTE(合成少数类过采样技术)', 'ADASYN': 'ADASYN(自适应过采样)',
                  'RandomOverSampler': 'RandomOverSampler(随机过采样)',
                  'RandomUnderSampler': 'RandomUnderSampler(随机下采样)',
                  'ClusterCentroids': 'ClusterCentroids(用K-Means算法的中心点来进行合成的下采样)'
                  }
preprocessing_method = {'StandardScaler': '标准化(将样本的均值变为0,方差变为1)', 'Normalizer': '正则化(将每个样本缩放到单位范数)',
                        'MinMaxScaler': 'MinMaxScaler(样本会缩放到[0,1])', 'MaxAbsScaler': 'MaxAbsScaler(样本会缩放到[-1,1])',
                        'RobustScaler': 'RobustScaler(通过四分位间距缩放数据)',
                        'sqrt': '平方根', 'ln': '以自然数e为底的对数', 'log2': '以2位底的对数', 'log10': '以10为底的对数'}
dict_method = dict(imputation_method, **balance_method, **preprocessing_method)


# ----------------内部函数-------------------
def _miss_row(data):
    """
    行缺失函数
    input：原始数据
    output：行的缺失个数和缺失率
    """
    row, col = data.shape
    row_miss = []
    row_total = []
    for i in range(row):
        w = data.iloc[i, :].isnull().sum()  # 第i行缺失的总数
        row_total.append(w)
        row_miss.append(w.sum() / col)
    row_miss = pd.Series(row_miss)
    row_total = pd.Series(row_total)
    row_miss.index = data.index  # 要保证row_miss和data的index相同
    row_percent = row_miss.sort_values(axis=0, ascending=False)  # 对其进行排序
    row_total = row_total.sort_values(axis=0, ascending=False)
    return row_total, row_percent


def _miss_col(data):
    """
    列缺失函数
    input:原始数据
    output：列的缺失个数和缺失率
    """
    col_total = data.isnull().sum().sort_values(ascending=False)  # 从大到小按顺序排每个特征缺失的个数
    col_percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)  # 从大到小按顺序排每个特征缺失率
    return col_total, col_percent


def _rf_pre(X_):  # 返回新的特征和标签
    most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_most_frequent = most_frequent.fit_transform(X_)  # 采用众数填补缺失值
    X_most_frequent = pd.DataFrame(X_most_frequent, columns=X_.columns)  # 加上表头
    column = []
    for i in X_.columns:
        if len(X_most_frequent) != (X_most_frequent.loc[:, i] == 1).sum() + (
                X_most_frequent.loc[:, i] == 0).sum():  # 判断该特征是否只有0，1变量
            column.append(i)
    X_rf = X_.loc[:, column]
    column_0 = set(X_most_frequent.columns) - set(column)
    X_0 = X_.loc[:, column_0]
    # print(X_0.size)
    X_2 = pd.DataFrame()
    if (X_0.size > 0):
        X_2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1).fit_transform(X_0)
        X_2 = pd.DataFrame(X_2, columns=X_0.columns)  # 加上表头
    return X_rf, X_2  # 返回连续性特征和离散型特征


def _rf_imputation(X_rf, X_2):  # 采用随机森林补充缺失值，并合并特征
    a = X_rf.isnull().sum()
    df = pd.DataFrame(a)
    column_null = df[df.iloc[:, 0] != 0].index.values  # 获取有空值的列名
    column_notnull = df[df.iloc[:, 0] == 0].index.values  # 获取无空值的列名
    X = X_rf.loc[:, column_null]
    y_full = X_rf.loc[:, column_notnull]
    column = X.columns
    X.columns = range(0, X.shape[1])
    X_missing = X.reset_index(drop=True)
    sortindex = np.argsort(X_missing.isnull().sum(axis=0)).values
    for i in sortindex:
        # 构建我们的新特征矩阵和新标签
        fillc = X_missing.iloc[:, i]
        df = pd.concat([X_missing.iloc[:, X_missing.columns != i], pd.DataFrame(y_full)], axis=1)
        # 在新特征矩阵中，对含有缺失值的列，进行中位数的填补
        df_0 = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(df)
        # 找出我们的训练集和测试集
        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]
        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]
        # 用随机森林回归来填补缺失值
        rfc = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)
        X_missing.loc[X_missing.iloc[:, i].isnull(), i] = Ypredict
    X_missing.columns = column
    X_full = pd.concat([X_missing, y_full, X_2], axis=1)
    return X_full


def _knn_pre(X_):  # 返回新的特征和标签
    most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_most_frequent = most_frequent.fit_transform(X_)  # 采用众数填补缺失值
    X_most_frequent = pd.DataFrame(X_most_frequent, columns=X_.columns)  # 加上表头
    column = []
    for i in X_.columns:
        if len(X_most_frequent) != (X_most_frequent.loc[:, i] == 1).sum() + (
                X_most_frequent.loc[:, i] == 0).sum():  # 判断该特征是否只有0，1变量
            column.append(i)
    X_rf = X_.loc[:, column]
    column_0 = set(X_most_frequent.columns) - set(column)
    X_0 = X_.loc[:, column_0]
    # print(X_0.size)
    X_2 = pd.DataFrame()
    if (X_0.size > 0):
        X_2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1).fit_transform(X_0)
        X_2 = pd.DataFrame(X_2, columns=X_0.columns)  # 加上表头
    return X_rf, X_2  # 返回连续性特征和离散型特征


def _knn_imputation(X_rf, X_2):  # 用KNN方法去填充
    a = X_rf.isnull().sum()
    df = pd.DataFrame(a)
    column_null = df[df.iloc[:, 0] != 0].index.values  # 获取有空值的列名
    column_notnull = df[df.iloc[:, 0] == 0].index.values  # 获取无空值的列名
    X = X_rf.loc[:, column_null]
    y_full = X_rf.loc[:, column_notnull]
    column = X.columns
    X.columns = range(0, X.shape[1])
    X_missing = X.reset_index(drop=True)
    sortindex = np.argsort(X_missing.isnull().sum(axis=0)).values
    for i in sortindex:
        # 构建我们的新特征矩阵和新标签
        fillc = X_missing.iloc[:, i]
        df = pd.concat([X_missing.iloc[:, X_missing.columns != i], pd.DataFrame(y_full)], axis=1)
        # 在新特征矩阵中，对含有缺失值的列，进行中位数的填补
        df_0 = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(df)
        # 找出我们的训练集和测试集
        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]
        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]
        # 用KNN回归来填补缺失值
        knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
        rfc = knn.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)
        X_missing.loc[X_missing.iloc[:, i].isnull(), i] = Ypredict
    X_missing.columns = column
    X_full = pd.concat([X_missing, y_full, X_2], axis=1)
    return X_full


# 获取数值型和二值型变量
def _feature_get_n_b(df_input):
    num_features = []
    binary_features = []
    for i in df_input.columns.values:
        f_type = str(df_input[i].dtype)
        if 'float' in f_type:
            uni = df_input[i].unique()
            if (1.0 in uni) and (0.0 in uni) and (len(uni) == 2):
                binary_features.append(i)
            else:
                num_features.append(i)
        elif 'int' in f_type:
            uni = df_input[i].unique()
            if (1 in uni) and (0 in uni) and (len(uni) == 2):
                binary_features.append(i)
            else:
                num_features.append(i)

    return num_features, binary_features


def _feature_classification(df_input):
    continuous_features = []
    categorical_features = []
    time_features = []
    for i in df_input.columns.values:
        f_type = str(df_input[i].dtype)
        if 'float' in f_type:
            continuous_features.append(i)
        elif 'int' in f_type:
            continuous_features.append(i)
        elif 'time' in f_type:
            time_features.append(i)
        else:
            categorical_features.append(i)
    return continuous_features, categorical_features, time_features


def _dataframe_to_categorical(df_input, features):
    for feature in features:
        if np.issubdtype(df_input[feature], np.float) or np.issubdtype(df_input[feature], np.int):
            if len(df_input[feature].unique()) > 10:  # 分类数大于10
                df_result, str_result, input_o_desc, input_n_desc, \
                result_o_desc, result_n_desc = group_recoding(df_input=df_input[[feature]],
                                                              feature=feature, group_num=3, type=2)
                df_input[feature] = df_result[str(feature) + '_分组']

    return df_input

    # x:需要分组数
    # group_list：分组对照list
    # start_mum:起始数值
    # group_list_label :：分组对照list标签


def _group(x, group_list, start_mum=1, group_list_label=None):
    if (x is None)or (math.isnan(x)) or (x == ''):
        return np.nan
    for i in range(start_mum, len(group_list) + start_mum - 1):
        index = i - start_mum + 1
        if (x <= group_list[index]):
            if group_list_label is None:
                return i
            else:
                return group_list_label[index - 1]


def _round_dec(n, d=2,pvalue=False):
    """
    设置小数点位数.
    """
    s = '0.' + '0' * d
    if (np.isinf(n)) or (np.isnan(n)):
        return n
    else:
        if pvalue and n<0.001:
            return '<0.001'
        return Decimal(str(n)).quantize(Decimal(s), rounding=ROUND_HALF_UP)

def _describe(df_input):
    """
    对数据进行描述
    """
    continuous_features, categorical_features, time_features = _feature_classification(df_input)
    try:
        o_df_result = df_input[categorical_features].describe(include=[np.object]).T
        o_df_result.columns = ['总数', '分类项', '频率最高项', '频数']
    except ValueError:
        o_df_result = pd.DataFrame()
    try:
        n_df_result = df_input[continuous_features].describe(include=[np.number]).T
        n_df_result = n_df_result[['count', 'mean', '50%', '25%', '75%', 'min', 'max']].round(3)
        n_df_result.columns = ['总数', '均值', '中位数', '25%分位数', '75%分位数', '最小值', '最大值']
    except ValueError:
        n_df_result = pd.DataFrame()
    return o_df_result, n_df_result


# ------------------------------------------------------
# --------------数据处理函数------------------------
"""
缺失数据删除
df_input:处理数据Dataframe
miss_rate:缺失率float
miss_axis:1 行缺失，0 列缺失
"""""


def miss_data_delete(df_input, miss_rate, miss_axis):
    str_result = '无缺失率大于' + str(miss_rate) + '的数据'
    if (miss_axis == 1):
        miss_rowtotal, miss_rowper = _miss_row(df_input)
        list = miss_rowper[miss_rowper > miss_rate].index
        df_result = df_input.drop(list)
        # df_miss_rate = pd.concat([df_input, miss_rowper], axis=1)
        if len(list) > 0:
            list_new = map(lambda x: str(x), list)
            str_result = '剔除缺失率大于' + str(miss_rate) + '的案例，剔除案例的id为' + "、".join(list_new)
    if (miss_axis == 0):
        col_total, col_percent = _miss_col(df_input)
        list = col_percent[col_percent > miss_rate].index
        df_result = df_input.drop(list, axis=1)
        # df_miss_rate = df_input.append(col_percent, ignore_index=True)
        if len(list) > 0:
            list_new = map(lambda x: str(x), list)
            str_result = '剔除缺失率大于' + str(miss_rate) + '的变量，共剔除' + str(len(list)) + '个变量,是' + "、".join(list_new)
            for i, v in col_percent.items():
                if v <= miss_rate:
                    break
                else:
                    str_result += '\n%s缺失率为%.2f%%,共缺失样本%d例' % (
                        i, (v * 100), col_total[i]
                    )
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
相关系数，删除相关性过高变量
df_input:Dataframe 处理数据
features:list 特征
method:list {'pearson', 'kendall', 'spearman'}
thres:float 阈值
0.3-弱,0.1-0.3为弱相关,0.3-0.5为中等相关,0.5-1.0为强相关
"""


def get_var_low_colinear(df_input, features=None, method='pearson', thres=0.3):
    str_result = ''
    if features is None:
        continuous_features, categorical_features, time_features = _feature_classification(df_input)
        features = continuous_features + categorical_features
    df_input_temp = df_input[features].dropna()
    df_corr = df_input_temp.corr(method=method)
    features = df_corr.columns
    col_del = []
    y_range = list(range(1, len(features)))
    for y in y_range:
        for x in list(range(0, y)):
            if np.abs(df_corr.iloc[x, y]) > thres:
                str_result += '[%s,%s]的相关性为%.3f,大于阈值' % (
                df_corr.columns[y], df_corr.index.values[x], df_corr.iloc[x, y])
                if (df_corr.columns[y] not in col_del)& (df_corr.index.values[x] not in col_del):
                    col_del.append(df_corr.columns[y])
                    str_result += '剔除变量%s\n' % (df_corr.columns[y])
                else:
                    str_result += '无变量被剔除\n'
    if len(col_del) > 0:
        str_result += '通过' + method + '相关性检验，' + "、".join(col_del) + '相关系数大于' + str(thres) + ',因此相关性较强,剔除这些变量'
    df_result = df_input.drop(columns=col_del)
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
VIF系数删除共线性过高变量
df_input:Dataframe 处理数据
features:list 特征
thres:float 阈值 (默认10)
返回值：

"""


def get_var_low_vif(df_input, features=None, thres=10.0):
    if features is None:
        continuous_features, categorical_features, time_features = _feature_classification(df_input)
        features = continuous_features
    df_input_temp = df_input[features].dropna()
    col = list(range(df_input_temp.shape[1]))
    col_del = []
    dropped = True
    str_result = ''
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df_input_temp.iloc[:, col].values, ix)
               for ix in range(df_input_temp.iloc[:, col].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            str_result += '\n删除：%s  vif=%.3f' % (df_input_temp.columns[col[maxix]], maxvif)
            col_del.append(df_input_temp.columns[col[maxix]])
            del col[maxix]
            dropped = True
    col_left = list(df_input_temp.columns[col])
    if len(col_del) > 0:
        str_result = '通过共线性检验，' + "、".join(col_del) + '变量VIF系数大于' + str(thres) + '存在共线性' + str_result
    df_result = df_input.drop(columns=col_del)
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
智能数据填补
df_input:Dataframe 处理数据
features:list 特征
method:str 填补方式
constant：object 常数
"""


def miss_data_filling(df_input, features=None, method=None, constant=None, path=None):
    if features is None:
        continuous_features, categorical_features, time_features = _feature_classification(df_input)
        features = continuous_features
        df_temp = df_input[features]
    else:
        df_temp = df_input[features]
    if (method == 'randomforest'):
        x1, x2 = _rf_pre(df_temp)
        df_temp = _rf_imputation(x1, x2)
        method_str = dict_method[method]
    elif method in ["mean", "median", "most_frequent"]:
        imp_ = SimpleImputer(missing_values=np.nan, strategy=method)
        df_temp = pd.DataFrame(data=imp_.fit_transform(df_temp), columns=features)
        method_str = dict_method[method]
    elif method == 'constant':
        imp_ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=constant)
        df_temp = pd.DataFrame(data=imp_.fit_transform(df_temp), columns=features)
        method_str = '常数' + str(constant)
    elif method == 'interpolate':
        df_temp = df_temp.interpolate().fillna(method='bfill')  # 插值法不能对首行空值填充，后面加上向后填充方法
        method_str = dict_method[method]
    elif method == 'KNN':
        x1, x2 = _knn_pre(df_temp)
        df_temp = _knn_imputation(x1, x2)
        method_str = dict_method[method]
    else:
        return '请输入正确的方法'

    def distribution_curve(df_pre, df_pos, features, path):
        plot_name_list = []
        for feature in features:
            plt.clf()
            plt.figure(figsize=[6, 6])
            temp1 = df_pre[feature]
            temp2 = df_pos[feature]
            temp1.plot(kind='kde', label='pre')
            temp2.plot(kind='kde', label='post')
            plt.title(feature)
            plt.legend()
            savepath_temp = 'distribution_curve' + str(random.randint(1, 100)) + '.jpeg'
            plt.savefig(path + savepath_temp, bbox_inches='tight')
            plt.cla()
            plot_name_list.append(savepath_temp)
        return plot_name_list

    # df_result = pd.concat([df_input.drop(features,axis=1), df_temp], axis=1)
    df_result = df_input.copy()
    df_result[features] = df_temp
    plot_name_list = distribution_curve(df_input, df_result, features, path)
    str_result = '采用' + method_str + '填补的方式对' + "、".join(features) + '进行数据填补'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc, plot_name_list


"""
数据标准化函数
df_input:Dataframe 处理数据
features:list 特征
method:str
"""


def data_standardization(df_input, features, method='StandardScaler'):
    # global dict_method
    if method == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
    elif method == 'Normalizer':
        scaler = preprocessing.Normalizer()
    elif method == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler()
    elif method == 'MaxAbsScaler':
        scaler = preprocessing.MaxAbsScaler()
    elif method == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif method == 'sqrt':
        scaler = preprocessing.FunctionTransformer(np.sqrt)
    elif method == 'ln':
        scaler = preprocessing.FunctionTransformer(np.log)
    elif method == 'log2':
        scaler = preprocessing.FunctionTransformer(np.log2)
    elif method == 'log10':
        scaler = preprocessing.FunctionTransformer(np.log10)
    else:
        return "请输入正确的方法"
    str_result = '采用' + dict_method[method] + '对' + "、".join(features) + '变量进行处理'
    scaler.fit(df_input[features])
    temp = scaler.transform(df_input[features])
    df_temp = pd.DataFrame(temp, columns=features)
    df_result = df_input.drop(features, 1)
    df_result = pd.concat([df_result, df_temp], axis=1)
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
非数值异常处理
df_input:Dataframe 处理数据
features:list 特征
method:str
"""


def non_numerical_value_process(df_input, features, method='median'):
    df_feature = df_input.loc[:, features]
    df_feature1 = df_feature.copy()
    Str_message1 = '将列名为{}，第{}行非数值的错误数据{}'
    Str_message2 = '将列名为{}，第{}行的异常值数据{}'

    def _is_number(s):  # 判断字符串是否为数值
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def _set_Disposition(df, disposition):
        columns_name = df.columns
        if disposition == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif disposition == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif disposition == 'most_frequent':
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif disposition == 'zero':
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        new_array = imp.fit_transform(df)
        new_df = pd.DataFrame(new_array, columns=columns_name)
        return new_df

    def _set_null(column_list):
        # 将列中不是数值的错误值设置为空
        column_list1 = column_list.copy()
        for i in range(len(column_list)):
            if not _is_number(column_list[i]):
                column_list1[i] = np.nan
        return column_list1

    df_nan = df_feature1.apply(_set_null, axis=0)
    if method == 'nan':
        df_temp = df_nan
    else:
        df_temp = _set_Disposition(df_nan, method)
    df_result = pd.concat([df_input.drop(features, axis=1), df_temp], axis=1)
    str_result = '采用' + dict_method[method] + '方法对' + "、".join(features) + '中非数值数据进行处理。'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
异常偏离值处理
df_input:Dataframe 处理数据
features:list 特征
method:str 方法
ratio：float 异常偏离比例
"""


# def abnormal_deviation_process(df_input, features, method='median', ratio=1.5, path=None):
#     df_feature = df_input[features]

#     # df_feature1 = df_feature.copy()
#     def lower_upper_limit(c):
#         lower_q = np.nanquantile(c, 0.25, interpolation='lower')  # 下四分位数
#         higher_q = np.nanquantile(c, 0.75, interpolation='higher')  # 上四分位数
#         int_r = higher_q - lower_q
#         lower_limit = lower_q - ratio * int_r
#         high_limit = higher_q + ratio * int_r
#         return lower_limit, high_limit

#     def set_Disposition(columns, Disposition):
#         if Disposition == 'median':
#             a = np.median(columns)
#         elif Disposition == 'mean':
#             a = np.mean(columns)
#         elif Disposition == 'most_frequent':
#             a = Counter(columns).most_common()
#         elif Disposition == 'zero':
#             a = 0
#         elif Disposition == 'nan':
#             a = np.nan
#         return a

#     def replace_exception(column_list):
#         # 将列中异常值进行替换
#         column_list1 = column_list.copy()
#         lower_limit, high_limit = lower_upper_limit(column_list)
#         for i in range(len(column_list)):
#             try:
#                 if column_list[i] < lower_limit or column_list[i] > high_limit:
#                     column_list1[i] = set_Disposition(column_list, method)
#             except Exception as e:
#                 # print(77777)
#                 # print(e)
#                 pass
#         # print(column_list1)
#         return column_list1
#     # print(df_feature)
#     print(features)
#     print(method)
#     print(ratio)
    
#     df_temp = df_feature.apply(replace_exception, axis=0)
#     str_result = '采用' + dict_method[method] + '方法对' + "、".join(features) + '中异常偏离超过正常值范围' + str(ratio) + '倍值的数据进行处理。'
#     df_result = df_input.drop(features, 1)
#     df_result = pd.concat([df_result, df_temp], axis=1)
#     input_o_desc, input_n_desc = _describe(df_input)
#     result_o_desc, result_n_desc = _describe(df_result)
#     ax1 = plt.figure()
#     plt01 = x5.comparison_plot(df_input=df_feature, features=features, group=None, kind='box', concat_way='free',
#                               row_size=1, col_size=len(features), path=path)
#     ax2 = plt.figure()
#     plt02 = x5.comparison_plot(df_input=df_temp, features=features, group=None, kind='box', concat_way='free',
#                               row_size=1,
#                               col_size=len(features), path=path)
#     plt_list_path = []
#     plt_list_path.append(plt01[0])
#     plt_list_path.append(plt02[0])
#     # if len(features) == 1:
#     #     df_box = pd.concat([df_feature, df_temp], axis=1)
#     #     df_box.columns = [features[0], features[0] + "(处理后)"]
#     #     df_box.boxplot(  # 指定绘图数据
#     #                 patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
#     #                 showmeans=True,  # 以点的形式显示均值
#     #                 boxprops={'color': 'black', 'facecolor': 'steelblue'},  # 设置箱体属性，如边框色和填充色
#     #                 flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
#     #                 # 设置均值点的属性，如点的形状、填充色和点的大小
#     #                 meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4},
#     #                 # 设置中位数线的属性，如线的类型和颜色
#     #                 medianprops={'linestyle': '--', 'color': 'orange'},
#     #                 # labels=['']  # 删除x轴的刻度标签，否则图形显示刻度标签为1
#     #                 )
#     return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc, plt_list_path

def abnormal_deviation_process(df_input, features, method='median', ratio=1.5, path=None):
    df_feature = df_input[features]

    # df_feature1 = df_feature.copy()
    def lower_upper_limit(c):
        lower_q = np.nanquantile(c, 0.25, interpolation='lower')  # 下四分位数
        higher_q = np.nanquantile(c, 0.75, interpolation='higher')  # 上四分位数
        int_r = higher_q - lower_q
        lower_limit = lower_q - ratio * int_r
        high_limit = higher_q + ratio * int_r
        return lower_limit, high_limit

    def set_Disposition(columns, Disposition):
        if Disposition == 'median':
            a = np.median(columns)
        elif Disposition == 'mean':
            a = np.mean(columns)
        elif Disposition == 'most_frequent':
            a = Counter(columns).most_common()
        elif Disposition == 'zero':
            a = 0
        elif Disposition == 'nan':
            a = np.nan
        return a

    def replace_exception(column_list):
        # 将列中异常值进行替换
        column_list1 = column_list.copy()
        lower_limit, high_limit = lower_upper_limit(column_list)
        for i in range(len(column_list)):
            if column_list[i] < lower_limit or column_list[i] > high_limit:
                column_list1[i] = set_Disposition(column_list, method)
        return column_list1
    print(df_feature)
    df_temp = df_feature.apply(replace_exception, axis=0)
    str_result = '采用' + dict_method[method] + '方法对' + "、".join(features) + '中异常偏离超过正常值范围' + str(ratio) + '倍值的数据进行处理。'
    df_result = df_input.drop(features, 1)
    df_result = pd.concat([df_result, df_temp], axis=1)
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    ax1 = plt.figure()
    plt01 = x5.comparison_plot(df_input=df_feature, features=features, group=None, kind='box', concat_way='free',
                               row_size=1, col_size=len(features), path=path)
    ax2 = plt.figure()
    plt02 = x5.comparison_plot(df_input=df_temp, features=features, group=None, kind='box', concat_way='free',
                               row_size=1,
                               col_size=len(features), path=path)
    plt_list_path = []
    plt_list_path.append(plt01[0])
    plt_list_path.append(plt02[0])
    # if len(features) == 1:
    #     df_box = pd.concat([df_feature, df_temp], axis=1)
    #     df_box.columns = [features[0], features[0] + "(处理后)"]
    #     df_box.boxplot(  # 指定绘图数据
    #                 patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
    #                 showmeans=True,  # 以点的形式显示均值
    #                 boxprops={'color': 'black', 'facecolor': 'steelblue'},  # 设置箱体属性，如边框色和填充色
    #                 flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
    #                 # 设置均值点的属性，如点的形状、填充色和点的大小
    #                 meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4},
    #                 # 设置中位数线的属性，如线的类型和颜色
    #                 medianprops={'linestyle': '--', 'color': 'orange'},
    #                 # labels=['']  # 删除x轴的刻度标签，否则图形显示刻度标签为1
    #                 )
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc, plt_list_path
"""
分组编码函数
df_input:Dataframe 处理数据
feature:str 特征
groupnum： int 分组数
group_percentile： list 分组百分位数 ([0,25,50,75,100])
group_cut_value：list 数值切点
group_range_value:dict 范围编码
type:int 编码类型1 数值 2 文字
"""


def group_recoding(df_input, feature, group_num=None, group_percentile=None, group_cut_value=None,
                   group_range_value=None, type=1):
    # x:需要分组数
    # dict_：分组对照dict
    def _group_dict(x_, dict_):
        if (x_ is None) or (x_ == ''):
            return np.nan
        for k, v in dict_.items():
            if x_ <= float(v[1]) and x_ >= float(v[0]):
                return (str(k))
        return np.nan

    df_result = df_input.copy()
    if group_range_value is None:
        if (group_num is not None):
            group_cut_list = np.linspace(0, 1, num=group_num + 1)
            group_cut = np.quantile(df_result[feature].dropna(), group_cut_list,interpolation="lower")
        elif (group_percentile is not None):
            group_cut_list = group_percentile
            group_cut = np.quantile(df_result[feature].dropna(), group_cut_list,interpolation="lower")
        elif (group_cut_value is not None):
            group_cut = group_cut_value
        else:
            return "请输入分组数据"
        #group_cut = [_round_dec(c, 3) for c in group_cut]
        group_cut = list(group_cut)
        df_result[feature + '_分组'] = df_result.apply(lambda row: _group(row[feature], group_cut, 0), axis=1)
        str_result = '对' + feature + '进行分组重编码,重编码结果为[' + feature + '_分组],'
        labels = list(range(0, len(group_cut)-1))
        labels_str = []
        str_result += '组0：' + str(group_cut[0]) + '≤' + feature + '≤' + str(group_cut[1]) + ',病例数为' + str(
            list(df_result[feature + '_分组']).count(labels[0])) + '；'
        labels_str.append('≤' + str(group_cut[1]))
        i = 2
        for la in labels[1:]:
            str_result += '组' + str(i-1) + '：' + str(group_cut[i - 1]) + '＜' + feature + '≤' + str(
                group_cut[i]) + ',病例数为' + str(list(df_result[feature + '_分组']).count(la)) + '；'
            labels_str.append(str(group_cut[i - 1]) + '＜x≤' + str(group_cut[i]))
            i += 1
        if type == 2:
            df_result[feature + '_分组'] = df_result.apply(
                lambda row: _group(row[feature], group_cut, 1, group_list_label=labels_str), axis=1)
    else:
        df_result[feature + '_分组'] = df_result.apply(lambda row: _group_dict(row[feature], group_range_value), axis=1)
        str_result = ''
        for k, v in group_range_value.items():
            str_result += '组' + str(k) + '：' + str(_round_dec(float(v[0]))) + '≤' + feature + '≤' + str(_round_dec(float(v[1]))) + \
                          ',病例数为' + str(list(df_result[feature + '_分组']).count(str(k))) + '；'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
哑变量重编码函数
df_input:Dataframe 需处理的数据
features：list 需要处理的变量
"""


def dummies_recoding(df_input, features):
    rank = pd.get_dummies(df_input[features], prefix=features)
    df_result = pd.concat([df_input, rank.iloc[:, :]], axis=1)
    str_result = '对' + features + '进行哑变量重编码'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
PSM倾向性匹配
df_input:Dataframe 需处理的数据
features：list 匹配的变量
group：str 分组变量
ratio：int 少数类比多数类的比例，默认1(只能是整数)
precious：bool   是否精确匹配 （TRUE 则进行精确匹配,False则进行一般匹配）
"""


def psm_matching(df_input, features, group, ratio=1,precious=False):
    y_label = np.unique(df_input[group])
    df_input.loc[df_input[group] == y_label[0], group] = 0
    df_input.loc[df_input[group] == y_label[1], group] = 1
    Y_field = [group]
    field = features + Y_field
    data = df_input[field]
    Y = df_input[Y_field]
    count_0 = (Y == 0).sum().values[0]
    count_1 = (Y == 1).sum().values[0]
    list_count = [count_0, count_1]
    index = list_count.index(max(list_count))
    formula = '{} ~ {}'.format(Y_field[0], '+'.join(features))
    y_samp, X_samp = patsy.dmatrices(formula, data=df_input, return_type='dataframe')
    glm = GLM(y_samp, X_samp, family=sm.families.Binomial())
    res = glm.fit()
    pro = res.predict(X_samp)
    data['scores'] = pro
    nmatches = ratio
    test_scores = data[data[Y_field[0]] == 1 - index][['scores']]
    ctrl_scores = data[data[Y_field[0]] == index][['scores']]
    result = []
    for i in range(0, len(test_scores)):
        match_id = test_scores.index[i]
        score = test_scores.iloc[i]
        matches = abs(ctrl_scores - score).sort_values('scores').head(nmatches)
        try:
            chosen = np.random.choice(matches.index, nmatches, replace=False)
            if precious == True:
                mathide=[match_id]
                mathide.extend(chosen)
                match_chose=pd.DataFrame(data.loc[mathide])
                dup = match_chose[~match_chose.duplicated(features, keep=False)]
                if not dup.empty:
                    continue

        except Exception as e:
            return {'error': '无法匹配所需数量的数据'}
        result.append(match_id)
        result.extend(list(chosen))
        ctrl_scores = ctrl_scores.drop(chosen, axis=0)
    matched_data = data.loc[result]  # 匹配分数列
    matched_data['record_id'] = matched_data.index
    df_result = df_input.loc[result]  # 匹配结果dataframe
    num0 = (df_result[group] == 0).sum()
    num1 = (df_result[group] == 1).sum()
    str_result = '通过倾向评分匹配（Propensity Score Matching，简称PSM），以' + \
                 "，".join(features) + '作为评分项，按照1：' + str(ratio) + '的比例对' + \
                 group + '进行匹配，最终匹配结果为' + group + '=' + str(0) + '共' + str(num0) + '例，' + \
                 group + '=' + str(1) + '共' + str(num1) + '例。'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc



"""
样本均衡
df_input:Dataframe 需处理数据
group:str分组变量
ratio：dict 平衡比例
method:str方法
"""


def data_balance(df_input, group, ratio, method='SMOTE'):
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
    df_input=df_input.dropna(subset=[group])
    Ratio = ratio
    list_name = [group]
    continuous_features, categorical_features, time_features, = _feature_classification(df_input)
    features = continuous_features
    if group not in features:
        features += group
    df_temp = df_input[features].copy()
    df_temp = df_temp.fillna(0)
    Y = df_temp.loc[:, list_name]
    X = df_temp.drop(group, axis=1)
    column_nameX = X.columns
    try:
        if method == 'SMOTE':
            X_resampled, Y_resampled = SMOTE(sampling_strategy=Ratio).fit_sample(X, Y)
        elif method == 'ADASYN':
            X_resampled, Y_resampled = ADASYN(sampling_strategy=Ratio).fit_sample(X, Y)
        elif method == 'RandomOverSampler':
            X_resampled, Y_resampled = RandomOverSampler(sampling_strategy=Ratio).fit_sample(X, Y)
    except  Exception as e:
        return {'error': '过采样数据量必须大于原始数据量，无法匹配所需数量的数据'}
    try:
        if method == 'RandomUnderSampler':
            X_resampled, Y_resampled = RandomUnderSampler(sampling_strategy=Ratio).fit_sample(X, Y)
        elif method == 'ClusterCentroids':
            X_resampled, Y_resampled = ClusterCentroids(sampling_strategy=Ratio).fit_sample(X, Y)
    except Exception as e:
        return {'error': '欠采样数据量必须小于原始数据量，无法匹配所需数量的数据'}
    array_result = np.column_stack((X_resampled, Y_resampled))
    list_column_name = list(column_nameX)
    list_column_name.append(group)
    df_result = pd.DataFrame(array_result, columns=list_column_name)
    group_labels = np.unique(df_result[group])
    str_temp = ''
    for group_label in group_labels:
        str_temp += group + '(' + str(group_label) + ')=' + str((df_result[group] == group_label).sum()) + '例，'

    str_result = '采用' + dict_method[method] + '方法对数据进行平衡，使得少数类和多数类的比例为' + str(Ratio) + \
                 '，最终匹配结果为' + str_temp + \
                 '该方法会自动剔除包含空值的行，以及非数值变量的列。'
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
数据操作
df_input:Dataframe 需处理数据
features:list 需处理变量
type：int 数据操作类型
        1：批量修改小数点位数（features需是定量数据）
        2：剔除变量
decimal_num：int 小数点位数       
"""


def data_manipulate(df_input, features, type, decimal_num=2):
    def _decimal_num(df_input, continuous_features, decimal_num):
        str_result = ''
        if decimal_num == 0:
            df_input[continuous_features] = df_input[continuous_features].applymap(lambda x: _round_dec(x, 0)).astype(
                int)
            str_result += '将' + ','.join(continuous_features) + '转化为整数。'
        else:
            df_input[continuous_features] = df_input[continuous_features].applymap(
                lambda x: _round_dec(x, decimal_num)).astype(float)
            str_result += '将' + ','.join(continuous_features) + '转化为' + str(decimal_num) + '位小数。'
        return df_input, str_result

    def _delete_features(df_input, features):
        str_result = '将' + ','.join(features) + '剔除。'
        df_result = df_input.drop(columns=features)
        return df_result, str_result

    if type == 1:
        df_result, str_result = _decimal_num(df_input=df_input, continuous_features=features, decimal_num=decimal_num)
    elif type == 2:
        df_result, str_result = _delete_features(df_input=df_input, features=features)
    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc


"""
数据转化
df_input:Dataframe 需处理数据
features:list 需处理变量
type：int 转化类型
        1：定量转定类
        2：定类转定量
        3: 自动转化(features可以为空)
num:int  自动转化为定类数据的最大分类数阈值
"""


def data_transform(df_input, features, type, num=3):
    def _num_to_object(df_input, continuous_features):
        df_input[continuous_features] = df_input[continuous_features].astype(str)
        str_result = ','.join(continuous_features) + ',已转化为定类数据。'
        return df_input, str_result

    def _object_to_num(df_input, categorical_features):
        except_features = []
        transformed_features = []
        for categorical_feature in categorical_features:
            try:
                df_input[categorical_feature] = df_input[categorical_feature].astype(float)
                transformed_features.append(categorical_feature)
            except Exception:
                except_features.append(categorical_feature)
        str_result = ','.join(transformed_features) + ',已转化为定量数据'
        if len(except_features) > 0:
            str_result += ','.join(except_features) + '，无法转变为定量数据。'
        else:
            str_result += '。'
        return df_input, str_result

    def _auto_transform(df_input, features, num):
        str_result = ''
        num_to_ = []
        if features is None:
            features = df_input.columns
        for feature in features:
            f_type = str(df_input[feature].dtype)
            if f_type.find('float') or f_type.find('int'):
                uni = df_input[feature].unique()
                if (len(uni) <= num):
                    df_input[feature] = df_input[feature].astype(str)
                    num_to_.append(feature)
        str_result += '已将' + ','.join(num_to_) + '转化为定类数据。'
        return df_input, str_result

    df_original = df_input.copy()
    if type == 1:
        df_result, str_result = _num_to_object(df_input, features)
    elif type == 2:
        df_result, str_result = _object_to_num(df_input, features)
    elif type == 3:
        df_result, str_result = _auto_transform(df_input, features, num)
    input_o_desc, input_n_desc = _describe(df_original)
    result_o_desc, result_n_desc = _describe(df_result)
    return df_result, str_result, input_o_desc, input_n_desc, result_o_desc, result_n_desc
