# AnalysisFunction.
# V 1.0.11
# date 2020-3-20
# author：xuyuan


#V1.0.1 更新说明：更新data_describe，替换INDEX
#V1.0.2 更新说明：更新nonparametric_test_continuous_feature，空的df_result 会报错
#V1.0.3 更新说明：更新 chi_square_test，fisher_test，nonparametric_test_categorical_feature，存在单元格为0的变量，仍显示总览及分组概览
#V1.0.4 更新说明：更新 multi_anova的描述
#V1.0.5 更新说明：更新 了所有方法的grouplabel筛选方式
#V1.0.6 更新说明：优化了 multi_comp,更新了get_var_correlation：修改了入参和返回参数
#V1.0.7 更新说明：修改了数据描述缺失率的错误
#V1.0.8 更新说明：优化了 multi_comp，comprehensive_smart_analysis
#V1.0.9 更新说明：优化了nonparametric_test，comprehensive_smart_analysis
#V1.0.10 更新说明： 优化了多个方法的描述，多个分析方法增加了.dropna(subset=[group])过滤，相关性分析增加参数 annote
#V1.0.11 更新说明： 优化了表格展示内容

from scipy import stats
import pandas as pd
import math
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import datetime

import AnalysisFunction.X_5_SmartPlot as x5
# from AnalysisFunction
from AnalysisFunction.X_1_DataGovernance import _feature_classification
from AnalysisFunction.X_1_DataGovernance import _feature_get_n_b
from AnalysisFunction.X_1_DataGovernance import _round_dec
from AnalysisFunction.X_1_DataGovernance import _dataframe_to_categorical
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.outliers_influence import variance_inflation_factor


#-----------------内部函数------------------------
"""
正态性检验
"""
def _check_normality(testData):
    # 样本数<5000用Shapiro-Wilk算法检验正态分布性
    if len(testData) < 5000:
        statistic,p_value = stats.shapiro(testData)
        return  statistic,p_value,'Shapiro-Wilk'
    # 样本数大于5000用Kolmogorov-Smirnov test算法检验正态分布性
    if len(testData) >= 5000:
        statistic,p_value = stats.kstest(testData, 'norm')
        return  statistic,p_value,'Kolmogorov-Smirnov'



#------------------web函数-----------------------
"""
数据描述
df_input:需处理的Dataframe
features：list
"""
def data_describe(df_input,features=None,decimal_num=3):
    if(features is not None):
        df_input=df_input[features]
    continuous_features, categorical_features, time_features  = _feature_classification(df_input)
    try:
        o_df = df_input[categorical_features ].astype(str).describe(include=[np.object]).T
        o_df.columns = ['总数', '分类项', '频率最高项', '频数']
        o_df['缺失率%']=df_input[categorical_features].isnull().sum()/len(df_input[categorical_features])*100
        o_df[['缺失率%']] =o_df[['缺失率%']].applymap(lambda x: _round_dec(x, decimal_num))
    except ValueError:
        o_df = pd.DataFrame()
    try:
        n_df = df_input[continuous_features].describe(include=[np.number]).T
        n_df['缺失率%'] = df_input[continuous_features].isnull().sum() / len(df_input[continuous_features]) * 100
        n_df = n_df[['count', 'mean','50%', '25%', '75%','std','min','max','缺失率%']].applymap(lambda x: _round_dec(x, decimal_num))
        n_df.columns = ['总数', '均数','中位数', '25%分位数', '75%分位数','方差','最小值','最大值','缺失率%']
    except ValueError:
        n_df = pd.DataFrame()
    n_df.reset_index(inplace=True)
    n_df.rename(columns={'index': '变量'}, inplace=True)
    o_df.reset_index(inplace=True)
    o_df.rename(columns={'index': '变量'}, inplace=True)
    return o_df, n_df

"""
正态性验证
df_input:Dataframe
continuous_features：list (定量数据)
"""
def normal_test(df_input, continuous_features,decimal_num=3,savePath=None):
    lf=len(continuous_features)
    df_result = pd.DataFrame()
    columns = ['总数', '偏度','峰度','标准误','统计量','p值','方法']
    str_result = ''
    plot_name_list = []
    for feature in continuous_features:
        list_result = []  # 一行的结果，顺序参照columns
        temp_df = df_input[[feature]].dropna()  # 获得长度数据集
        temp = temp_df[feature]
        list_result.append(len(temp))
        std = temp.std()  # 获得数据集的标准差
        if lf==1:
            plt.figure(12)
            plt.figure(figsize=(10, 4), dpi=300)
            plt.subplot(121)
            mean = temp.mean()  # 获得数据集的平均值
            min_value=np.min(temp)
            max_value=np.max(temp)
            x = np.arange(min_value, max_value, (max_value-min_value)/100)
            y=np.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
            #plt.figure(figsize=(4, 4), dpi=300)
            plt.plot(x, y)
            plt.hist(temp, bins=12, rwidth=0.9, density=True)
            plt.title('distribution')
            plt.xlabel(feature)
            plt.ylabel('Probability')
            plt.subplot(122)
            #plt.figure(figsize=(4, 4), dpi=300)
            stats.probplot(temp,plot=plt)
            time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
                datetime.datetime.now().second)
            plot_name = "dist_plot" + time_ + ".jpeg"
            plt.savefig(savePath + plot_name, bbox_inches="tight")
            plot_name_list.append(plot_name)
            plt.close()
        else:
            plot_name=x5.dist_plot(df_input=df_input,features=[feature],title=str(feature)+'频率分布直方图',
                                   path=savePath)
            # plot_name=x5.dist_plot(df_input=df_input,features=continuous_features,title='频率分布直方图',
            #                        path=savePath,col_size=len(continuous_features),row_size=1)
            plot_name_list+=plot_name
        n, p,_method = _check_normality(temp)  # 判断正态
        skewness=temp.skew()
        kurtosis=temp.kurt()
        se=std/math.sqrt(len(temp))
        list_result.append(skewness)
        list_result.append(kurtosis)
        list_result.append(se)
        list_result.append(n)
        list_result.append(p)
        list_result.append(_method)
        df_result[feature]=list_result
        if p<0.05:
            str_result+='采用'+_method+'检验对'+feature+'进行正态性检验，统计量为'+str(_round_dec(n,decimal_num))+\
                       ',显著性水平为'+str(_round_dec(p,decimal_num))+'小于0.05,拒绝原假设，认为样本不服从正态分布,'
        elif p>=0.05:
            str_result+='采用'+_method+'检验对'+feature+'进行正态性检验，统计量为'+str(_round_dec(n,decimal_num))+\
                       ',显著性水平为'+str(_round_dec(p,decimal_num))+'大于0.05,接受原假设，认为样本来自服从正态分布的总体,'
        str_result+='样本偏度为：'+ str(_round_dec(skewness,decimal_num))+',峰度为：'+str(_round_dec(kurtosis,decimal_num))+',标准误：'+str(_round_dec(se,decimal_num))
        if abs(skewness)>(se*2) and abs(skewness)<(se*3):
            str_result+=',偏度为标准误的'+ str(_round_dec(abs(skewness)/se,decimal_num))+'倍，建议采用取根号值来转换为正态分布。\n'
        elif abs(skewness) >= (se * 3) :
            str_result += ',偏度为标准误的' + str(_round_dec(abs(skewness) / se, decimal_num)) + '倍，建议采用取对数的方式转换为正态分布。\n'
        else:
            str_result += ',偏度为标准误的' + str(_round_dec(abs(skewness) / se, decimal_num)) + '倍，轻度偏态分布。\n'
    df_result=df_result.T
    df_result.columns = columns
    df_result[['偏度','峰度','标准误','统计量','p值']]=df_result[['偏度','峰度','标准误','统计量','p值']].applymap(lambda x: _round_dec(x, decimal_num))
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index':'变量'},inplace=True)
    return df_result,str_result,plot_name_list

"""
方差齐性验证
df_input:Dataframe
group：str
continuous_features：list (定量数据)
method：str {'mean', 'median', 'trimmed'}
"""
def levene_test(df_input, group, continuous_features,group_labels=None, method='median',decimal_num=3):
    df_result = pd.DataFrame()
    columns = ['总数', '统计量', 'p值']
    str_result=''
    for feature in continuous_features:
        list_result = []
        df_temp = df_input[[feature, group]].dropna()
        if group_labels is None:
            group_labels = np.unique(df_temp[group])
        # temp_po = df_temp[df_temp[group] == group_labels[0]][feature]
        # temp_neg = df_temp[df_temp[group] == group_labels[1]][feature]
        labels_temp=[]
        for group_label in group_labels:
            labels_temp.append(df_temp[df_temp[group] == group_label][feature])
        n, p = stats.levene(*labels_temp,center=method)
        list_result.append(len(df_temp[feature]))
        list_result.append(n)
        list_result.append(p)
        df_result[feature]=list_result
        if p < 0.05:
            str_result += '对'+feature+'采用Levene检验进行方差齐性检验，统计量为' + str(_round_dec(n, decimal_num)) + \
                         ',显著性水平为' + str(_round_dec(p, decimal_num)) + '小于0.05,拒绝原假设，样本方差不齐。\n'
        elif p >= 0.05:
            str_result += '对'+feature+'采用Levene检验进行方差齐性检验，统计量为' + str(_round_dec(n, decimal_num)) + \
                         ',显著性水平为' + str(_round_dec(p, decimal_num)) + '大于0.05,接受原假设，认为样本方差齐。\n'
    df_result = df_result.T.applymap(lambda x: _round_dec(x, decimal_num))
    df_result.columns = columns
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index':'变量'},inplace=True)
    return df_result, str_result

"""
相关性分析
df_input:需处理的Dataframe
features:list
cor_method:str (pearson, kendall, spearman)
plot_method：str （heatmap，clustermap，pairplot）
hue_：str 分层变量  (定类数据)
cmap_style:str 色带 默认None，['coolwarm','YlGnBu','RdBu_r','greys']
"""
def get_var_correlation(df_input, features=None, cor_method='pearson', plot_method='heatmap', hue_=None,decimal_num=3,savepath=None,cmap_style=None,annot=False):
    if features is not None:
        df_temp=df_input[features]
    else:
        df_temp=df_input
    df_result = df_temp.corr(method=cor_method)

    print(df_result)
    plt.clf()
    plt.figure(figsize=(6,5),dpi=300)
    if plot_method=='heatmap':
        sns.heatmap(df_result,linewidths=0.3,cmap=cmap_style,annot=annot)
    elif plot_method=='clustermap':
        sns.clustermap(df_result,linewidths=0.3,cmap=cmap_style,annot=annot)
    elif plot_method == 'pairplot':
        features.append(hue_)
        sns.pairplot(df_input[features], hue=hue_)
    df_result=df_result.applymap(lambda x: _round_dec(x, decimal_num))
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index':'变量'},inplace=True)
    temp_name = 'corr' + str(random.randint(1, 100)) + '.jpeg'
    plt.savefig(savepath+temp_name,bbox_inches="tight")
    plt.close()
    plot_name_list=[]
    plot_name_list.append(temp_name)
    return df_result,plot_name_list

"""
共线性分析
df_input:需处理的Dataframe
features:list 
"""
def get_var_vif(df_input, features=None,decimal_num=3):
    if features is not None:
        df_input = df_input[features]
    else:
        n,b=_feature_get_n_b(df_input)
        df_input = df_input[n+b]
    df_input=df_input.dropna()
    df_input[df_input.shape[1]]=1
    #vif
    vif=[]
    for i in range(df_input.shape[1] - 1):
        vif.append(variance_inflation_factor(df_input.values, i))
    #result_out
    df_result=pd.DataFrame(df_input.columns[:-1, ])
    df_result.rename(columns={0:"变量名"},inplace=True)
    df_result["vif"]=vif
    df_result[["vif"]]= df_result[["vif"]].applymap(lambda x: _round_dec(x, decimal_num))
    df_result=df_result.sort_values(['vif'],ascending = False)
    df_result.reset_index(drop = True,inplace=True)
    return df_result

"""
卡方检验
df_input：DataFrame 输入的待处理数据
group：str 分组变量
categorical_features：list 定义分类变量 （定类数据）
group_labels：list 分组类别
decimals: int 小数点位数
"""
def chi_square_test(df_input, group, categorical_features,yates_correction=False,group_labels=None,show_method=False,decimal_num=3):
    str_result=''
    list_result_feature_name=[]
    list_result_total_num=[]
    list_result_chi=[]
    list_result_p=[]
    list_result_method = []
    list_result_p_str=[]
    list_result_error_str=[]
    df_result = pd.DataFrame()
    df_input=df_input.dropna(subset=[group])
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input=df_input[(df_input[group].isin(group_labels))]
    for categorical_feature in categorical_features:
        df_result_temp = pd.DataFrame()
        df_cross=pd.crosstab(df_input[categorical_feature],df_input[group])
        if np.any(df_cross == 0):  # 判断是否有格子中的数字为0
            list_result_error_str.append(categorical_feature)
            chi2=np.nan
            chi2_p=np.nan
        else:
            chi2, chi2_p, dof, expctd = stats.chi2_contingency(df_cross, correction=yates_correction)
        if (chi2_p <= 0.05):
            list_result_p_str.append(categorical_feature)
        list_result_feature_name.append(categorical_feature+' ,n(%)')
        list_result_total_num.append(len(df_input[[categorical_feature, group]].dropna()))
        list_result_chi.append(_round_dec(chi2, decimal_num))
        list_result_p.append(_round_dec(chi2_p, 3,True))
        if yates_correction ==False:
            list_result_method.append('Chi-square test')
        elif yates_correction ==True:
            list_result_method.append('Chi-square test(Yates\'correction)')
        for i in range(df_cross.shape[0]-1):#填充
            list_result_feature_name.append('')
            list_result_total_num.append('')
            list_result_chi.append('')
            list_result_p.append('')
            list_result_method.append('')
        df_cross['总览'] = df_cross.sum(axis=1)
        df_percent = df_cross.div(df_cross.sum(axis=0), axis=1).multiply(100).applymap(lambda x: _round_dec(x, decimal_num))
        df_percent = df_percent.applymap(lambda x: '(%s)' % x)
        for label in df_cross.columns:
            df_result_temp[label] = df_cross[label].map(str) + df_percent[label]
        if df_result.shape[0]<1:
            df_result=df_result_temp
        else:
            df_result=df_result.append(df_result_temp)
    dic = {}
    for label in group_labels:
        dic[label] = group+'\n' + str(label) + ' (n=' + str(df_input[df_input[group]==label].shape[0])+')'
    df_result['分类项'] = df_result.index.tolist()
    df_result['变量']=list_result_feature_name
    df_result['总数']=list_result_total_num
    df_result['统计量']=list_result_chi
    df_result['p']=list_result_p
    df_result['方法'] = list_result_method
    #重新排列Dataframe列顺序
    new_columns = ['变量','总数','分类项', '总览']
    for label in group_labels:
        new_columns.append(label)
    new_columns.extend(['统计量', 'p'])
    if show_method is True:
        new_columns.append('方法')
    df_result=df_result.reindex(columns=new_columns)
    df_result.rename(columns=dic, inplace=True)
    if len(list_result_p_str) > 0:
        str_result += ','.join(list_result_p_str)+'经卡方检验p<0.05,组间差异存在统计学意义。\n'
    if len(list_result_error_str)>0:
        str_result += ','.join(list_result_error_str)+ '，存在计数为0的单元格,无法分析；\n'
    return df_result,str_result

"""
fisher检验
df_input：DataFrame 输入的待处理数据
group：str 分组变量
categorical_features：list 定义分类变量（必须是二分类）
group_labels：list 分组类别（必须是二分类）
"""
def fisher_test(df_input, group, categorical_features,group_labels=None,show_method=False, decimal_num=3):
    str_result = ''
    list_result_feature_name = []
    list_result_total_num=[]
    list_result_chi = []
    list_result_p = []
    list_result_method = []
    list_result_p_str = []
    list_result_error_str = []
    list_result_error_str1 = []
    df_result = pd.DataFrame()
    df_input = df_input.dropna(subset=[group])
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    if len(group_labels)>2:
        return {'error': '分组变量中分组类别大于2组'}
    if len(group_labels)<2:
        return {'error': '分组变量中分组类别小于2组'}
    for categorical_feature in categorical_features:
        df_result_temp = pd.DataFrame()
        df_cross = pd.crosstab(df_input[categorical_feature], df_input[group])
        if df_cross.shape[0]>2:
            list_result_error_str1.append(categorical_feature) # 判断是分类数是否大于2
            continue
        if np.any(df_cross == 0):  # 判断是否有格子中的数字为0
            list_result_error_str.append(categorical_feature)
            chi2 = np.nan
            chi2_p = np.nan
        else:
            chi2, chi2_p = stats.fisher_exact(df_cross)
        if (chi2_p <= 0.05):
            list_result_p_str.append(categorical_feature)
        list_result_feature_name.append(categorical_feature + ' ,n(%)')
        list_result_total_num.append(len(df_input[[categorical_feature,group]].dropna()))
        list_result_chi.append(_round_dec(chi2, decimal_num))
        list_result_p.append(_round_dec(chi2_p, 3,True))
        list_result_method.append('Fisher exact test')
        for i in range(df_cross.shape[0] - 1):
            list_result_feature_name.append('')
            list_result_total_num.append('')
            list_result_chi.append('')
            list_result_p.append('')
            list_result_method.append('')
        df_cross['总览'] = df_cross.sum(axis=1)
        df_percent = df_cross.div(df_cross.sum(axis=0), axis=1).multiply(100).applymap(lambda x: _round_dec(x, decimal_num))
        df_percent = df_percent.applymap(lambda x: '(%s)' % x)
        for label in df_cross.columns:
            df_result_temp[label] = df_cross[label].map(str) + df_percent[label]
        if df_result.shape[0] < 1:
            df_result = df_result_temp
        else:
            df_result = df_result.append(df_result_temp)
    dic = {}
    for label in group_labels:
        dic[label] = group + '\n' + str(label) + ' (n=' + str(df_input[df_input[group]==label].shape[0]) + ')'
    df_result['分类项'] = df_result.index.tolist()
    df_result['总数'] = list_result_total_num
    df_result['变量'] = list_result_feature_name
    df_result['统计量'] = list_result_chi
    df_result['p'] = list_result_p
    df_result['方法'] = list_result_method
    new_columns = ['变量', '总数','分类项', '总览']
    for label in group_labels:
        new_columns.append(label)
    new_columns.extend(['统计量', 'p'])
    if show_method is True:
        new_columns.append('方法')
    df_result = df_result.reindex(columns=new_columns)
    df_result.rename(columns=dic, inplace=True)
    if len(list_result_p_str) > 0:
        str_result += ','.join(list_result_p_str) + '经fisher检验p<0.05,组间差异存在统计学意义。\n'
    if len(list_result_error_str) > 0:
        str_result += ','.join(list_result_error_str) + '，存在计数为0的单元格,无法分析；\n'
    if len(list_result_error_str1) > 0:
        str_result += ','.join(list_result_error_str1) + '，分类数大于2,无法分析；\n'
    return df_result, str_result


"""
t检验
df_input：DataFrame 输入的待处理数据
group：str 分组变量
continuous_features：list 定义连续变量   （定量数据）
group_labels：list 分组类别
related:bool 是否配对
"""
def two_sample_t_test(df_input, group, continuous_features, group_labels=None, relate=False, show_method=False, decimal_num=3):
    df_result = pd.DataFrame()
    df_input = df_input.dropna(subset=[group])
    str_result = ''
    columns = ['总数', '总览']
    new_labels = []
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    if len(group_labels)>2:
        return {'error': '分组变量中分组类别大于2组'}
    if len(group_labels)<2:
        return {'error': '分组变量中分组类别小于2组'}
    for label in group_labels:
        new_labels.append(str(label)+'_数量')
        new_labels.append(group +'\n' + str(label) + ' (n=' + str(df_input[df_input[group]==label].shape[0]) + ')')
    columns.extend(new_labels)
    columns.extend(['统计量', 'p'])
    if show_method is True:
        columns.append('方法')
    list_result_p = []
    for feature in continuous_features:
        df_temp = df_input[[feature, group]].dropna()
        list_result = []  # 一行的结果，顺序参照columns
        _method=''
        temp = df_temp[feature]
        temp_po=df_temp[df_temp[group]==group_labels[0]][feature]
        temp_neg=df_temp[df_temp[group]==group_labels[1]][feature]
        list_result.append(len(temp))
        temp_str = str(_round_dec(np.mean(temp),decimal_num)) + "(" + str(_round_dec(np.std(temp),decimal_num))+ ")"
        list_result.append(temp_str)
        for y_label in group_labels:
            temp = df_temp.loc[df_temp[group] == y_label][feature]
            temp_str = str(_round_dec(np.mean(temp),decimal_num)) + "(" + str(_round_dec(np.std(temp),decimal_num))+ ")"
            list_result.append(len(temp))
            list_result.append(temp_str)
        if relate :
            try:
                f, p = stats.ttest_rel(temp_po, temp_neg)
                _method='related t-test'
            except:
                str_result+=feature+'两组数量不一致无法分析'
                continue
        else:
            a, lev_p = stats.levene(temp_po, temp_neg)  # 判断方差齐性
            if (lev_p >= 0.05):
                f, p = stats.ttest_ind(temp_po, temp_neg)
                _method = 't-test '
            else:
                f, p = stats.ttest_ind(temp_po, temp_neg, equal_var=False)
                _method = 'Welch\'s t-test'
        list_result.append(_round_dec(f, decimal_num))  # 添加f值
        list_result.append(_round_dec(p, 3,True))  # 添加P值
        if show_method is True:
            list_result.append(_method)
        if (p <= 0.05):
            list_result_p.append(feature)
        df_result[feature] = pd.Series(list_result)
    if len(list_result_p) > 0:
        str_result += ','.join(list_result_p)+'经t检验p值<0.05，组间差异存在统计学意义。\n'
    df_result = df_result.T
    if df_result.shape[0]>0:
        df_result.columns = columns
    else:
        df_result=pd.DataFrame(columns=columns)
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': '变量'}, inplace=True)
    df_result['变量'] = ['%s ,mean(SD)' % s for s in df_result["变量"]]
    return df_result, str_result

"""
单因素方差分析
df_input：DataFrame 输入的待处理数据
group：str 分组变量
continuous_features：list 定义连续变量  （定量数据） 
group_labels：list 分组类别
"""
def one_way_anova(df_input,group,continuous_features,group_labels=None,show_method=False,decimal_num=3):
    df_result = pd.DataFrame()
    str_result = ''
    columns = ['总数', '总览']
    new_labels = []
    df_input = df_input.dropna(subset=[group])
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    for label in group_labels:
        new_labels.append(str(label)+'_数量')
        new_labels.append(group +'\n' + str(label) + ' (n=' + str(df_input[df_input[group]==label].shape[0]) + ')')
    columns.extend(new_labels)
    columns.extend(['统计量', 'p'])
    if show_method is True:
        columns.append('方法')
    list_result_p=[]
    for feature in continuous_features:
        df_temp = df_input[[feature, group]].dropna()
        list_result = []  # 一行的结果，顺序参照columns
        list_series=[]
        temp = df_temp[feature]
        list_result.append(len(temp))
        temp_str = str(_round_dec(np.mean(temp),decimal_num)) + "(" + str(_round_dec(np.std(temp),decimal_num))+ ")"
        list_result.append(temp_str)
        for l in group_labels:
            temp = df_temp.loc[df_temp[group] == l][feature]
            temp_str = str(_round_dec(np.mean(temp),decimal_num)) + "(" + str(_round_dec(np.std(temp),decimal_num))+ ")"
            list_result.append(len(temp))
            list_result.append(temp_str)
            list_series.append(temp)
        f, p = stats.f_oneway(*list_series)
        list_result.append(_round_dec(f, decimal_num))
        list_result.append(_round_dec(p, 3,True))
        if show_method is True:
            list_result.append('one way anova')
        df_result[feature] = pd.Series(list_result)
        if (p<= 0.05):
            list_result_p.append(feature)
    if len(list_result_p) > 0:
        str_result +=  ','.join(list_result_p)+'经ANOVA检验p值<0.05，组间差异存在统计学意义。\n'
    df_result = df_result.T
    df_result.columns = columns
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': '变量'}, inplace=True)
    df_result['变量'] = ['%s ,mean(SD)' % s for s in df_result["变量"]]
    return df_result,str_result

"""
多因素方差分析
df_input：DataFrame 输入的待处理数据
group：str 应变量（定量数据）
features：list 自变量 
formula_type：int
1：常规 2： 计算相互作用
anova_test：str
统计方法{"F", "Chisq", "Cp"}
anova_typ: int
展示格式{1,2,3}
"""
def multi_anova(df_input, group, features, formula_type=1,anova_test='F',anova_typ=3,decimal_num=3):
    df_input = df_input.dropna(subset=[group])
    formula = group+'~'
    if formula_type==1:
        formula+='+'.join(features)
    elif formula_type==2:
        formula+='*'.join(features)
    model = smf.ols(formula=formula, data=df_input).fit()
    df_result = sm.stats.anova_lm(model,test=anova_test,typ=anova_typ)
    df_result=df_result.applymap(lambda x: _round_dec(x, decimal_num))
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': '变量'}, inplace=True)
    new_columns={'sum_sq':'平方和','mean_sq':'均方','PR(>F)':'p'}
    df_result.rename(columns=new_columns, inplace=True)
    list_p=[]
    for i in range(1,df_result.shape[0]):
       if df_result.loc[i,'p']<0.05:
           list_p.append(df_result.loc[i,'变量'])
    str_result='研究'+str(group)+'的影响因素，通过多因素方差分析同时对'+','.join(features)+'变量进行分析，'
    str_result+='从分析结果中可以看出'+','.join(list_p)+'P值小于0.05存在统计学意义，说明'+','.join(list_p)+'对'+str(group)+'有显著影响。'
    return df_result,str_result

"""
中位数差异分析
df_input：DataFrame 输入的待处理数据
group：str 分组变量
group_labels：list 分组类别
continuous_features：list 定义连续变量（定量数据）   
iter_n：int 重采样次数
"""
def median_difference(df_input,group,continuous_features,group_labels=None,iter_n=100,decimal_num=3):
    df_input = df_input.dropna(subset=[group])
    df_result = pd.DataFrame()
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    if len(group_labels)>2:
        return {'error': '分组变量中分组类别大于2组'}
    if len(group_labels)<2:
        return {'error': '分组变量中分组类别小于2组'}
    columns = ['中位数差值', 'std','95%CI']
    for feature in continuous_features:
        list_result = []  # 一行的结果，顺序参照columns
        df_temp = df_input[[feature, group]].dropna()
        temp_po = df_temp[df_temp[group] == group_labels[0]][feature]
        temp_neg = df_temp[df_temp[group] == group_labels[1]][feature]
        difference_median = []
        for i in range(iter_n):
            difference_median.append(
                np.median(temp_po.sample(frac=0.5, replace=True)) -
                np.median(temp_neg.sample(frac=0.5, replace=True)))
        list_result.append(str(_round_dec(np.quantile(difference_median, q=0.5), decimal_num)))
        list_result.append(str(_round_dec(np.std(difference_median), 3)))
        list_result.append('['+str(_round_dec(np.quantile(difference_median, q=0.025), decimal_num))+
                           ','+str(_round_dec(np.quantile(difference_median, q=0.975), decimal_num))+']')
        df_result[feature] = pd.Series(list_result)
    df_result=df_result.T
    df_result.columns = columns
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': '变量'}, inplace=True)
    str_result='采用bootstrap方法对两组样本进行随机抽样，并分别计算中位数，重复'+str(iter_n)+\
               '次，生成一个中位数差值组成的新样本，计算新样本的方差及95%置信' \
               '区间，置信区间采用百分位数法计算。'
    return df_result,str_result

"""
非参数检验(定量数据)
df_input：DataFrame 输入的待处理数据
group：str 分组变量
continuous_features：list 定义连续变量   （定量数据） 
method:str 方法名['Mannwhitney-U','Wilcoxon','Kruskal-Wallis','Friedman']
group_labels：list 分组标签
"""
def nonparametric_test_continuous_feature(df_input,group,continuous_features,method,group_labels=None,show_method=False,decimal_num=3):
    df_result = pd.DataFrame()
    df_input = df_input.dropna(subset=[group])
    str_result = ''
    columns = ['总数','总览']
    # columns.append('总览' + '\n' + ' (n=' + str(df_input.shape[0]) + ')')
    new_labels = []
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    if method in ['Mannwhitney-U','Wilcoxon']:#两组检验
        if len(group_labels)>2:
            return {'error': '分组变量中分组类别大于2组'}
    if len(group_labels)<2:
        return {'error': '分组变量中分组类别小于2组'}
    for label in group_labels:
        # n = sum(df_input[group] == label)
        new_labels.append(str(label)+'_数量')
        new_labels.append(group + '\n' + str(label) + ' (n=' + str(df_input[df_input[group] == label].shape[0]) + ')')
    columns.extend(new_labels)
    columns.extend(['统计量', 'p'])
    if show_method is True:
        columns.append('方法')
    list_result_p=[]
    list_error=[]
    for feature in continuous_features:
        df_temp = df_input[[feature, group]].dropna()
        list_result = []  # 一行的结果，顺序参照columns
        list_series = []
        temp = df_temp[feature]
        temp_str = str(np.quantile(temp, q=0.5, interpolation='nearest')) + "[" + \
                   str(np.quantile(temp, q=0.25, interpolation='nearest')) + "," + \
                   str(np.quantile(temp, q=0.75, interpolation='nearest')) + "]"
        list_result.append(len(temp))
        list_result.append(temp_str)
        for label in group_labels:
            temp_ = df_temp.loc[df_temp[group] == label][feature]
            temp = [float(i) for i in temp_]
            if len(temp)>0:
                temp_str = str(np.quantile(temp, q=0.5, interpolation='nearest')) + "[" + \
                           str(np.quantile(temp, q=0.25, interpolation='nearest')) + "," + \
                           str(np.quantile(temp, q=0.75, interpolation='nearest')) + "]"
            else:
                temp_str='NaN(NaN)'
            list_result.append(len(temp))
            list_result.append(temp_str)
            list_series.append(temp)
        if method=='Mannwhitney-U':
            try:
                u_static, pval = stats.mannwhitneyu(list_series[0], list_series[1], alternative='two-sided')
                n1 = np.size(list_series[0])
                n2 = np.size(list_series[1])
                static = (u_static - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            except:
                list_error.append(feature)
                continue
        elif method=='Wilcoxon':
            try:
                static, pval = stats.wilcoxon(list_series[0], list_series[1])
            except:
                list_error.append(feature)
                continue
        elif method=='Kruskal-Wallis':
            try:
                static, pval = stats.mstats.kruskal(*list_series)

            except:
                list_error.append(feature)
                continue
        elif method=='Friedman':
            try:
                static, pval  = stats.friedmanchisquare(*list_series)
            except:
                list_error.append(feature)
                continue
        else:
            return {'error': '请传入正确的方法'}
        list_result.append(_round_dec(static, decimal_num))
        list_result.append(_round_dec(pval, 3,True))
        if show_method is True:
            list_result.append(method)
        df_result[feature] = pd.Series(list_result)
        if (pval <= 0.05):
            list_result_p.append(feature)
    if len(list_result_p) > 0:
        str_result += ','.join(list_result_p)+'经'+method+'检验p值<0.05，组间差异存在统计学意义。\n'
    if len(list_error) > 0:
        str_result += ','.join(list_error)+'数据错误，无法进行'+method+'检验。\n'
    df_result = df_result.T
    if df_result.empty:
        df_result=pd.DataFrame(columns=columns)
    else:
        df_result.columns = columns
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': '变量'}, inplace=True)
    df_result['变量'] = ['%s ,median[IQR]' % s for s in df_result["变量"]]
    return df_result, str_result

"""
非参数检验(定类数据)
df_input：DataFrame 输入的待处理数据
group：str 分组变量
categorical_features：list 定义分类变量   （定类数据） 
method:str 方法名['cochrans_q','McNemar']
group_labels：list 分组标签
"""
def nonparametric_test_categorical_feature(df_input, group, categorical_features,method,group_labels=None,show_method=False,decimal_num=3):
    str_result=''
    list_result_feature_name=[]
    list_result_total_num = []
    list_result_static=[]
    list_result_p=[]
    list_result_metod = []
    list_result_p_str=[]
    list_error=[]
    df_result = pd.DataFrame()
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    else:
        df_input = df_input[(df_input[group].isin(group_labels))]
    for categorical_feature in categorical_features:
        df_result_temp = pd.DataFrame()
        df_cross=pd.crosstab(df_input[categorical_feature],df_input[group])
        if np.any(df_cross == 0):  # 判断是否有格子中的数字为0
            list_error.append(categorical_feature)
            static = np.nan
            pval = np.nan
            list_result_metod.append('')
        else:
            if method=='cochrans_q':
                try:
                    b = cochrans_q(df_cross.values)
                    static=b.statistic
                    pval=b.pvalue
                    list_result_metod.append('cochran\'s Q')
                except:
                    list_error.append(categorical_feature)
                    continue
            elif method == 'McNemar':
                try:
                    b = mcnemar(df_cross.values)
                    static = b.statistic
                    pval = b.pvalue
                    list_result_metod.append('McNemar')
                except Exception :
                    list_error.append(categorical_feature)
                    continue
            else:
                return {'error': '请传入正确的方法'}
        if (pval <= 0.05):
            list_result_p_str.append(categorical_feature)
        list_result_feature_name.append(categorical_feature+' ,n(%)')
        list_result_total_num.append(len(df_input[[categorical_feature,group]].dropna()))
        list_result_static.append(_round_dec(static, decimal_num))
        list_result_p.append(_round_dec(pval, 3,True))
        for i in range(df_cross.shape[0]-1):#填充
            list_result_feature_name.append('')
            list_result_total_num.append('')
            list_result_static.append('')
            list_result_p.append('')
            list_result_metod.append('')
        df_cross['总览'] = df_cross.sum(axis=1)
        df_percent = df_cross.div(df_cross.sum(axis=0), axis=1).multiply(100).applymap(lambda x: _round_dec(x, decimal_num))
        df_percent = df_percent.applymap(lambda x: '(%s)' % x)
        for label in df_cross.columns:
            df_result_temp[label] = df_cross[label].map(str) + df_percent[label]
        if df_result.shape[0]<1:
            df_result=df_result_temp
        else:
            df_result=df_result.append(df_result_temp)
    dic = {}
    for label in group_labels:
        dic[label] = group+'\n' + str(label) + ' (n=' + str(df_input[df_input[group]==label].shape[0])+')'
    df_result['分类项'] = df_result.index.tolist()
    df_result['变量']=list_result_feature_name
    df_result['总数']=list_result_total_num
    df_result['统计量']=list_result_static
    df_result['p']=list_result_p
    df_result['方法'] = list_result_metod

    #重新排列Dataframe列顺序
    new_columns = ['变量','总数','分类项','总览']
    for label in group_labels:
        new_columns.append(label)
    new_columns.extend(['统计量', 'p'])
    if show_method is True:
        new_columns.append('方法')
    df_result=df_result.reindex(columns=new_columns)
    df_result.rename(columns=dic, inplace=True)
    if len(list_result_p_str) > 0:
        str_result += ','.join(list_result_p_str)+'经'+method+'检验p<0.05,组间差异存在统计学意义。\n'
    if len(list_error)>0:
        str_result += ','.join(list_error)+ '，存在计数为0的单元格或错误的数据,无法分析；\n'
    return df_result,str_result


"""
多重比较
df_input：DataFrame 输入的待处理数据
group:str 分组变量 （定类数据）
continuous_feature：str自变量（定量数据）
method：str 多重比较方法
plot_type: int {1:'箱型图',2:'小提琴图'}
"""
def multi_comp(df_input, group, continuous_feature,path, method='tukeyhsd',plot_type=1):
    df_temp=df_input[[continuous_feature, group]].dropna()
    try:
        df_temp[continuous_feature] = df_input[continuous_feature].astype('Float32')
        df_temp[group] = df_input[group].astype('str')
    except Exception as e:
        print(e)
        return {'error':'数据错误'}
    mc=MultiComparison(df_temp[continuous_feature], df_temp[group])
    str_result='采用'+str(method)+'方法，分析不同'+str(group)+'分组下'+str(continuous_feature)+'的差异。'
    if method=='tukeyhsd':
        df_result_temp=mc.tukeyhsd()._results_table.data
        df_result=pd.DataFrame(df_result_temp[1:],columns=df_result_temp[0])
        df_result['分类项'] = [' % s-' % i for i in df_result["group1"]]
        df_result['分类项'] =df_result['分类项']+df_result['group2'].map(str)
        df_result=df_result[['分类项','meandiff','lower','upper','p-adj']]
    #箱型图
    plot_name_list=[]
    plt.figure(figsize=[5,5],dpi=300)
    if plot_type==1:
        sns.boxplot(x=group, y=continuous_feature,hue=None, data=df_temp)
    else:
        sns.violinplot(x=group, y=continuous_feature, hue=None, data=df_temp)
    temp_name='box' +str(random.randint(1,100))+ '.jpeg'
    savepath_temp = path + temp_name
    plt.savefig(savepath_temp, bbox_inches='tight')
    plot_name_list.append(temp_name)
    plt.cla()
    #森林图
    plt.figure(figsize=[5,5], dpi=300)
    path_list_temp=x5.forest_plot(df_result,'分类项','meandiff',['lower','upper'],direct='vertical',fig_size=[5,5],title='differece 95% CI',path=path,axvline_tick=0)
    plot_name_list=plot_name_list+path_list_temp
    plt.close()
    return str_result,df_result,plot_name_list


"""
两组样本自动分析
df_input: Dataframe 传入
group：str 分组变量
group_labels：list 分组类别
categorical_features：list 分类变量名称
continuous_features：list 连续变量名称
relate: bool 是否为配对样本
show_method:bool 是否展示方法
decimal_num： int 小数点位数
"""
def _two_groups_smart_analysis(df_input,group,categorical_features,continuous_features,group_labels=None,relate=False,show_method=True,decimal_num=3):
    str_result=''
    str_result_error=''
    columns=['变量','总数','分类项','总览']
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    for label in group_labels:
        columns.append(group + '\n' + str(label) + ' (n=' + str(df_input[df_input[group] == label].shape[0]) + ')')
    columns.extend(['统计量', 'p'])
    if show_method is True:
        columns.append( '方法')
    df_result = pd.DataFrame(columns=columns)
    if categorical_features is not None:
        if relate:# -------------------------配对样本------------------------------
            # McNemar test
            df_temp,str_temp=nonparametric_test_categorical_feature(df_input=df_input,group=group,categorical_features=categorical_features,method='McNemar',group_labels=group_labels,show_method=show_method,decimal_num=decimal_num)
            df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
            str_result+=str_temp
        else:# -------------------------独立样本------------------------------
            # 卡方、fisher
            for categorical_feature in categorical_features:
                df_cross=pd.crosstab(df_input[categorical_feature],df_input[group])
                # if np.any(df_cross == 0):  # 判断是否有格子中的数字为0
                #     str_result_error += categorical_feature + '存在计数为0的单元格无法统计，'
                #     continue
                df_cross_exp = stats.contingency.expected_freq(df_cross)
                if (len(df_input[categorical_feature]) >= 40 & np.sum(df_cross_exp < 5) == 0):
                    df_temp, str_temp = chi_square_test(df_input=df_input, group=group,
                                                                               categorical_features=[categorical_feature],
                                                                               group_labels=group_labels,
                                                                               yates_correction=False,show_method=show_method,
                                                                               decimal_num=decimal_num)
                    df_result=pd.concat([df_result,df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
                elif (df_cross.shape[0]==2 and df_cross.shape[1]==2) and \
                        (len(df_input[categorical_feature]) < 40 | np.sum(df_cross_exp < 1) > 0):
                    df_temp, str_temp = fisher_test(df_input=df_input, group=group,
                                                        categorical_features=[categorical_feature],
                                                        group_labels=group_labels,show_method=show_method,
                                                        decimal_num=decimal_num)
                    df_result=pd.concat([df_result,df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
                else:
                    df_temp, str_temp = chi_square_test(df_input=df_input, group=group,
                                                        categorical_features=[categorical_feature],
                                                        group_labels=group_labels,
                                                        yates_correction=True,show_method=show_method,
                                                        decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
    if continuous_features is not None:
        if relate:
            for continuous_feature in continuous_features:
                n, p, _method = _check_normality(df_input[continuous_feature])  # 判断正态
                if p < 0.05:
                    # Wilcoxon test
                    df_temp, str_temp = nonparametric_test_continuous_feature(df_input=df_input, group=group,
                                                                               continuous_features=[continuous_feature],
                                                                               method='Wilcoxon', group_labels=group_labels,
                                                                               show_method=show_method, decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
                else:
                    #t 配对 t
                    df_temp, str_temp = two_sample_t_test(df_input=df_input, group=group,
                                                          continuous_features=[continuous_feature],
                                                          group_labels=group_labels, relate=relate,
                                                                              show_method=show_method,
                                                                              decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
        else:
            for continuous_feature in continuous_features:
                n, p, _method = _check_normality(df_input[continuous_feature])  # 判断正态
                if p < 0.05:
                    # U test
                    df_temp, str_temp = nonparametric_test_continuous_feature(df_input=df_input, group=group,
                                                                               continuous_features=[continuous_feature],
                                                                               method='Mannwhitney-U', group_labels=group_labels,
                                                                               show_method=show_method, decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
                else:
                    #t检验
                    df_temp, str_temp = two_sample_t_test(df_input=df_input, group=group,
                                                          continuous_features=[continuous_feature],
                                                          group_labels=group_labels, relate=relate,
                                                          show_method=show_method,
                                                          decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
    str_result=str_result_error+str_result
    df_result=df_result[columns]
    return df_result,str_result


"""
多组样本自动分析
df_input: Dataframe 传入
group：str 分组变量
group_labels：list 分组类别
categorical_features：list 分类变量名称
continuous_features：list 连续变量名称
relate: bool 是否为配对样本
show_method:bool 是否展示方法
decimal_num： int 小数点位数
"""
def _multi_groups_smart_analysis(df_input,group,categorical_features,continuous_features,group_labels=None,relate=False,show_method=True,decimal_num=3):
    str_result=''
    str_result_error=''
    columns=['变量','总数','分类项','总览']
    if (group_labels is None):
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    for label in group_labels:
        columns.append(group + '\n' + str(label) + ' (n=' + str(df_input[df_input[group] == label].shape[0]) + ')')
    columns.extend(['统计量', 'p'])
    if show_method is True:
        columns.append( '方法')
    df_result = pd.DataFrame(columns=columns)
    if categorical_features is not None:
        if relate:# -------------------------配对样本------------------------------
            # Cochran's Q test
            df_temp,str_temp=nonparametric_test_categorical_feature(df_input=df_input,group=group,categorical_features=categorical_features,method='cochrans_q',group_labels=group_labels,show_method=show_method,decimal_num=decimal_num)
            df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
            str_result+=str_temp
        else:# -------------------------独立样本------------------------------
            # 卡方、fisher
            for categorical_feature in categorical_features:
                df_cross=pd.crosstab(df_input[categorical_feature],df_input[group])
                # if np.any(df_cross == 0):  # 判断是否有格子中的数字为0
                #     str_result_error += categorical_feature + '存在计数为0的单元格无法统计，'
                #     continue
                df_cross_exp = stats.contingency.expected_freq(df_cross)
                _ratio = df_cross_exp.size * 0.2
                if (np.sum(df_cross_exp < 5)<_ratio):
                    df_temp, str_temp = chi_square_test(df_input=df_input, group=group,
                                                                               categorical_features=[categorical_feature],
                                                                               group_labels=group_labels,
                                                                               yates_correction=False,show_method=show_method,
                                                                               decimal_num=decimal_num)
                    df_result=pd.concat([df_result,df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
                else:
                    df_temp, str_temp = chi_square_test(df_input=df_input, group=group,
                                                        categorical_features=[categorical_feature],
                                                        group_labels=group_labels,
                                                        yates_correction=True,show_method=show_method,
                                                        decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
    if continuous_features is not None:
        if relate:
            # Friedman
            df_temp, str_temp = nonparametric_test_continuous_feature(df_input=df_input, group=group,
                                                                       continuous_features=continuous_features,
                                                                       method='Friedman', group_labels=group_labels,
                                                                       show_method=show_method, decimal_num=decimal_num)
            df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
            str_result += str_temp
        else:
            for continuous_feature in continuous_features:
                n, p, _method = _check_normality(df_input[continuous_feature])  # 判断正态
                if p >= 0.05:
                    list_series=[]
                    for l in group_labels:
                        temp_ = df_input.loc[df_input[group] == l][continuous_feature]
                        temp = [float(i) for i in temp_]
                        list_series.append(temp)
                    a, lev_p = stats.levene(*list_series)  # 判断方差齐性
                    if (lev_p >= 0.05):  # 方差齐
                        # 单因素方差检验
                        df_temp, str_temp = one_way_anova(df_input=df_input, group=group,
                                                                                   continuous_features=[continuous_feature],
                                                                                   group_labels=group_labels,
                                                                                   show_method=show_method, decimal_num=decimal_num)
                        df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                        str_result += str_temp
                    else:
                        df_temp, str_temp = nonparametric_test_continuous_feature(df_input=df_input, group=group,
                                                                                  continuous_features=[
                                                                                      continuous_feature],
                                                                                  method='Kruskal-Wallis',
                                                                                  group_labels=group_labels,
                                                                                  show_method=show_method,
                                                                                  decimal_num=decimal_num)
                        df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                        str_result += str_temp
                else:
                    #Kruskal-Wallis检验
                    df_temp, str_temp = nonparametric_test_continuous_feature(df_input=df_input, group=group,
                                                          continuous_features=[continuous_feature],method='Kruskal-Wallis',
                                                          group_labels=group_labels,
                                                          show_method=show_method,
                                                          decimal_num=decimal_num)
                    df_result = pd.concat([df_result, df_temp], axis=0, ignore_index=True)
                    str_result += str_temp
    df_result=df_result[columns]
    str_result=str_result_error+str_result
    return df_result,str_result

"""
综合智能统计分析
df_input:需处理的Dataframe
group：分组变量str
categorical_features：分类变量名称list （可以放入定类或定量数据）
continuous_features：连续变量名称list  （只能放定量数据）
group_labels：分组标签list None,
show_method：bool 是否展示检验方法
relate：bool False  
"""
def comprehensive_smart_analysis(df_input,group,categorical_features=None,continuous_features=None,group_labels=None,show_method=True,relate=False,decimal_num=3):
    def miss_count(count,total):
        try:
            return total-int(count)
        except:
            return count
    df_input=df_input.dropna(subset=[group])
    if categorical_features is not None:
        df_input=_dataframe_to_categorical(df_input,categorical_features)
    if group_labels is None:
        group_labels = df_input[group].unique()
        try:
            group_labels.sort()
        except Exception:
            print(Exception)
    if (categorical_features is None) and (continuous_features is None):
        df_temp=df_input.drop(group, 1)
        continuous_features,categorical_features,time_features= _feature_classification(df_temp)
    if len(group_labels)==2:
        df_result, str_result=_two_groups_smart_analysis(df_input=df_input,group=group,categorical_features=categorical_features,continuous_features=continuous_features,group_labels=group_labels,show_method=show_method,relate=relate,decimal_num=decimal_num)
    elif len(group_labels)>2:
        df_result, str_result=_multi_groups_smart_analysis(df_input=df_input,group=group,categorical_features=categorical_features,continuous_features=continuous_features,group_labels=group_labels,show_method=show_method,relate=relate,decimal_num=decimal_num)
    else:
        return {'error':'分组变量中分类项少于2'}
    str_result_="利用综合智能基线分析研究不同"+str(group)+"分组中各个指标的差异是否具有统计学意义，总的有效样本为"+str(df_input.shape[0])+'例，其中'
    for la in group_labels:
        str_result_ += str(group)+'='+str(la)+'：病例数为' + str(list(df_input[group]).count(la)) + '；'
    str_result_+='\n（综合智能基线分析将根据样本的分布情况、方差齐性、样本量智能选择分析方法）：\n'
    str_result=str_result_+str_result

    df_result['总数']=df_result['总数'].apply(lambda x:miss_count(x,df_input.shape[0]))
    df_result.rename(columns={'总览': '总览' + '\n' + ' (n=' + str(df_input.shape[0]) + ')','总数':'缺失'}, inplace=True)
    return df_result, str_result


