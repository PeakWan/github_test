# AnalysisFunction.
# V 1.0.18
# date 2020-5-26
# author：xuyuan

#V1.0 更新说明：更新lifelines 版本 0.25.5
#V1.0.1 更新说明：更新智能组分析（smart_goup_analysis） 增加了入参出参，点线图改为误差图
#V1.0.2 更新说明：更新智能组分析（smart_goup_analysis） 修复了分层不能传定量变量
#V1.0.3 更新说明：更新了多个方法的描述，smart_goup_analysis增加了描述返回：str_result
#V1.0.4 更新说明：import X_2,更新了smart_goup_analysis:修改了表格返回dict_df_result，更新了一些描述
#V1.0.5 更新说明：更新了survival_estimating_models，更新了smart_goup_analysis
#V1.0.6 更新说明：更新了分层分析和趋势分析，增加了try
#v1.0.7 更新说明：更新多模型回归、分层分析和趋势分析 不同模型的系数不同，logist用or 其他用β, 删除线性回归、逻辑回归方法
#V1.0.8 更新说明：更新多模型回归、分层分析和趋势分析  增加模型 cox  添加 入参time_variable
#V1.0.9 更新说明：更新两组ROC：修改了入参和返回参数；优化了smart_goup_analysis
#V1.0.10 更新说明：更新ROC：自动剔除缺失值 ；更新多模型分析、分层分析：获取label时自动剔除空值; 更新趋势分析：label自动转为int；更新生存分析：自动剔除缺失值
#V1.0.11 更新说明：新增了单因素\多因素分析
#v1.0.12 更新说明：优化了平滑曲线，优化了单因素\多因素分析描述,所有方法增加参数style
#v1.0.13 更新说明：优化智能组分析（smart_goup_analysis）
#v1.0.14 更新说明：优化智能组分析（smart_goup_analysis）新增参数palette_style，新增返回return {error}
#v1.0.15 更新说明：优化roc(two_groups_roc）新增参数palette_style
#v1.0.16 更新说明：单因素/多因素、分层、趋势分析新增返回return {error}，优化了多模型的描述
#v1.0.17 更新说明： 更新COX参数估计的方法为“breslow”，修改COX的样本量计算方式
#v1.0.18 更新说明： 多模型分层、单因素多因素分析定类变量增加控制
#v1.0.19 更新说明：hz 更改smooth_curve_fitting_analysis（平滑曲线（线性相加模型）分析）plt.legend(loc='auto')中auto 改成best
#v1.0.19 更新说明：hz 更改two_groups_roc（ROC曲线）多分类问题及处理原先二分类只能处理0,1问题
#v1.0.20 更新说明：hz 更改R_cox_regression,中时间依赖Roc 分位点整数问题R代码修改

import datetime
import time
import pandas as pd
import numpy as np
import random
import json
import statsmodels.api as sma
import statsmodels.formula.api as smf
from AnalysisFunction.X_1_DataGovernance import _feature_get_n_b
from AnalysisFunction.X_1_DataGovernance import _feature_classification
from AnalysisFunction.X_1_DataGovernance import _round_dec
from AnalysisFunction.X_1_DataGovernance import _group
from AnalysisFunction.X_2_DataSmartStatistics import comprehensive_smart_analysis
import AnalysisFunction.X_5_R_SmartPlot as x5r
import AnalysisFunction.X_5_SmartPlot as x5
from sklearn.metrics import roc_curve,auc,confusion_matrix
import matplotlib.pyplot as plt
import math
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.utils import median_survival_times
from lifelines.utils import survival_table_from_events
from lifelines import CoxPHFitter
from lifelines.plotting import plot_lifetimes
from lifelines.statistics import multivariate_logrank_test
import scipy.stats as stats
import seaborn as sns
import itertools
from pygam import LinearGAM
import re

from scipy import interp
from sklearn.preprocessing import label_binarize
import traceback



plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#--------------通用函数---------------
Classfication_method = {'LogisticRegression': 'LogisticRegression(logistics回归)'
    , 'XGBClassifier': 'XGBClassifier(极限梯度提升树)'
    , 'RandomForestClassifier': 'RandomForestClassifier(随机森林)'
    , 'SVC': 'SVC(支持向量机)', 'KNN': 'KNN(最邻近结点算法)'}
Regression_method = {'LinearRegression': 'LinearRegression(线性回归)', 'XGBRegressor': 'XGBRegressor(XGboost回归)'
    , 'RandomForestRegressor': 'RandomForestRegressor(随机森林回归)', 'LinearSVR': 'LinearSVR(线性支持向量回归)'
    , 'KNeighborsRegressor': 'KNeighborsRegressor(KNN回归)'}
Cluster_method={'KMeans':'KMeans(K均值聚类)'
        ,'Birch':'Birch(优化后的层次聚类)'
        ,'SpectralClustering':'SpectralClustering(谱聚类)'
        ,'AgglomerativeClustering':'AgglomerativeClustering(自底而上的层次聚类)'
        ,'GMM':'mixture.GaussianMixture(基于高斯混合模型的最大期望聚类)'}

dict_models={'logit':'二元logistics回归','mulclass':'有序多分类logistics回归','ols':'线性回归','poisson':'泊松回归','rlm':'稳健线性模型','glm':'广义线性模型','cox':'cox回归'}
#计算标准误
def _standard_error(sample):
    std=np.std(sample,ddof=0)
    standard_error=std/math.sqrt(len(sample))
    return standard_error
def _is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(str(num))
    return True if result else False
#将一列95%CI的数据转为两列
def _data_format_ci(x_series):
    nd_array = []
    for x in x_series:
        try:
            nd_array.append(json.loads(x))
        except Exception:
            nd_array.append([None,None])
    return nd_array
#系数转化
def coef_conversion(model_name,coef):

    if model_name in ['logit','cox']:
        return np.exp(coef)
    else:
        return coef
def is_filter(n):
    filter_lis = ["",None," ",np.nan,str(np.nan),str(None)]
    return str(n) not in filter_lis


def variables_control(df,variable_list,method=""):##测试临时由10修改为15   20210715
    if method=='logit': count_max=15
    elif method == 'ols': count_max = 15
    elif method == 'cox': count_max = 15
    else:count_max=15
    for variable in variable_list:
        if len(df[variable].unique())>count_max:
            return(variable+'变量的分类项过多(N>15)，建议转变为连续变量或哑变量后进行分析。')
    return False

def get_variables(value, split_symbol):
    variables = []
    for ivalue in value:
        x = re.split(split_symbol, ivalue)
        for ix in x:
            if ix not in variables:
                variables.append(ix)
    return variables



#-----------单因素\多因素分析------------------------------------


def _categorical_multivariate_analysis(df_input, categorical_exposure_variable, dependent_variable, model_name,
                                       adjust_variables, categorical_exposure_variable_ref=None,time_variable=None,  decimal_num=3):

    if adjust_variables is None:
        adjust_variables=[]
    if time_variable is not None:
        df_input = df_input[[categorical_exposure_variable] + [dependent_variable] + adjust_variables+[time_variable]].dropna()
    else:
        df_input = df_input[[categorical_exposure_variable] + [dependent_variable] + adjust_variables].dropna()
    try:
        df_input[categorical_exposure_variable]=df_input[categorical_exposure_variable].astype(int)
    except Exception:
        print(Exception)
    exposure_feature_label=df_input[categorical_exposure_variable].unique().tolist()
    exposure_feature_label = list(filter(is_filter, list(exposure_feature_label)))
    try:
        exposure_feature_label.sort()
    except Exception:
        print(Exception)
    exposure_feature_temp_label=exposure_feature_label.copy()
    if (categorical_exposure_variable_ref is None) or (categorical_exposure_variable_ref is np.nan):
        categorical_exposure_variable_ref=exposure_feature_label[0]
    exposure_feature_temp_label.remove(categorical_exposure_variable_ref) #生成哑变量标签
    exposure_feature_model_label=[categorical_exposure_variable + '_' + str(x) for x in exposure_feature_temp_label]
    exposure_feature_all_label = [categorical_exposure_variable + '_' + str(x) for x in exposure_feature_label]
    if adjust_variables is not None:
        features= exposure_feature_model_label + adjust_variables
    else:
        features = exposure_feature_model_label
    df_input_temp = pd.concat([df_input, pd.get_dummies(df_input[categorical_exposure_variable],
                                                        prefix=categorical_exposure_variable).iloc[:, :]], axis=1)
    #print(df_input_temp)
    if model_name=='cox':
        formula = time_variable + '~' + '+ '.join(features)
        #print(formula)
        result_model = smf.phreg(formula=formula, data=df_input_temp, status=dependent_variable, ties='breslow').fit()
        df_result = pd.DataFrame()
        list_temp_name = []
        list_temp_N = []
        list_temp_value = []
        list_temp_CI = []
        list_temp_p = []
        list_temp_name.append(categorical_exposure_variable)
        list_temp_N.append(np.nan)
        list_temp_value.append(np.nan)
        list_temp_CI.append(np.nan)
        list_temp_p.append(np.nan)
        ref_label_i=0 #用于标志是否循环到参考标签
        for i,label in enumerate(exposure_feature_all_label):
            #print(label)
            list_temp_name.append(re.sub(categorical_exposure_variable + '_', '', label))
            list_temp_N.append(result_model.model.surv.n_obs)
            if label in exposure_feature_model_label:
                list_temp_value.append(
                    '%s' % (_round_dec(coef_conversion(model_name, result_model.params[i-ref_label_i]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[i-ref_label_i][0]),
                               decimal_num),
                    _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[i-ref_label_i][1]),
                               decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[i-ref_label_i], decimal_num)))
            else:
                list_temp_value.append(np.nan)
                list_temp_CI.append(np.nan)
                list_temp_p.append(np.nan)
                ref_label_i=1

    else:
        if model_name == 'logit':
            model_family = sma.families.Binomial()
        elif model_name == 'ols':
            model_family = sma.families.Gaussian()
        elif model_name == 'poisson':
            model_family = sma.families.Poisson()
        else:
            model_family = sma.families.Gaussian()
        formula = dependent_variable + '~' + '+ '.join(features)
        result_model = smf.glm(formula=formula, family=model_family, data=df_input_temp).fit()
        df_result = pd.DataFrame()
        list_temp_name=[]
        list_temp_N=[]
        list_temp_value=[]
        list_temp_CI = []
        list_temp_p=[]
        list_temp_name.append(categorical_exposure_variable)
        list_temp_N.append(np.nan)
        list_temp_value.append(np.nan)
        list_temp_CI.append(np.nan)
        list_temp_p.append(np.nan)
        for label in exposure_feature_all_label:
            list_temp_name.append(re.sub(categorical_exposure_variable+'_','',label))
            list_temp_N.append(sum(df_input_temp[label]))
            if label in exposure_feature_model_label:
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[label]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[label, 0]), decimal_num),
                    _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[label, 1]),
                               decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[label], decimal_num)))
            else:
                list_temp_value.append(np.nan)
                list_temp_CI.append(np.nan)
                list_temp_p.append(np.nan)
    df_result['name'] = list_temp_name
    df_result['N'] = list_temp_N
    df_result['β'] = list_temp_value
    df_result['95%CI'] = list_temp_CI
    df_result['P-value'] = list_temp_p
    return df_result

def _continue_multivariate_analysis(df_input, continue_exposure_variables, dependent_variable, model_name='logit',
                                    adjust_variables=None, time_variable=None, decimal_num=3):
    df_result=pd.DataFrame()
    list_temp_name = []
    list_temp_N=[]
    list_temp_value=[]
    list_temp_CI = []
    list_temp_p = []
    list_temp_N_adjuested=[]
    list_temp_value_adjuested=[]
    list_temp_CI_adjuested = []
    list_temp_p_adjuested = []
    list_error=[]
    if model_name=='cox':
        for exposure_variable in continue_exposure_variables:
            # ---计算单变量结果
            try:
                formula = time_variable + '~' + exposure_variable
                model_result = smf.phreg(formula=formula, data=df_input, status=dependent_variable, ties='breslow').fit()
                list_temp_name.append(exposure_variable)
                list_temp_N.append(model_result.model.surv.n_obs)
                list_temp_value.append(
                    '%s' % (_round_dec(coef_conversion(model_name, model_result.params[0]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (_round_dec(
                    coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][0]),
                    decimal_num)
                                                 , _round_dec(
                    coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][1]),
                    decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[0], decimal_num)))
                if adjust_variables is not None:
                    formula = time_variable + '~' + exposure_variable + '+' + '+'.join(adjust_variables)
                    model_result = smf.phreg(formula=formula, data=df_input, status=dependent_variable, ties='breslow').fit()
                    list_temp_N_adjuested.append(model_result.model.surv.n_obs)
                    list_temp_value_adjuested.append('%s' % (
                        _round_dec(coef_conversion(model_name, model_result.params[0]), decimal_num)))
                    list_temp_CI_adjuested.append('[%s,%s]' % (
                        _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None)[0][0]),decimal_num)
                        , _round_dec(coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][1]), decimal_num)))
                    list_temp_p_adjuested.append(str(_round_dec(model_result.pvalues[0], decimal_num)))
            except Exception as e:
                print(e)
                list_error.append(exposure_variable)

    else:
        if model_name == 'logit':
            model_family = sma.families.Binomial()
        elif model_name == 'ols':
            model_family = sma.families.Gaussian()
        elif model_name=='poisson':
            model_family=sma.families.Poisson()
        else:
            model_family = sma.families.Gaussian()

        for exposure_variable in continue_exposure_variables:
            try:
            #---计算单变量结果
                formula = dependent_variable + '~' + exposure_variable
                model_result = smf.glm(formula=formula, family=model_family, data=df_input).fit()
                list_temp_name.append(exposure_variable)
                list_temp_N.append(model_result.nobs)
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,model_result.params[exposure_variable]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (_round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num)
                                                , _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[exposure_variable], decimal_num)))
                if adjust_variables is not None:
                    formula = dependent_variable + '~' + exposure_variable + '+' + '+'.join(adjust_variables)
                    model_result = smf.glm(formula=formula, family=model_family, data=df_input).fit()
                    list_temp_N_adjuested.append(model_result.nobs)
                    list_temp_value_adjuested.append('%s' % (_round_dec(coef_conversion(model_name,model_result.params[exposure_variable]), decimal_num)))
                    list_temp_CI_adjuested.append('[%s,%s]' % (
                        _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num)
                        , _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
                    list_temp_p_adjuested.append(str(_round_dec(model_result.pvalues[exposure_variable], decimal_num)))
            except Exception as e:
                print(e)
                list_error.append(exposure_variable)
    df_result[0] = list_temp_name
    df_result[1] = list_temp_N
    df_result[2] = list_temp_value
    df_result[3] = list_temp_CI
    df_result[4] = list_temp_p
    columns = ['name', 'N', 'β', '95%CI', 'P-value']
    if adjust_variables is not None:
        df_result[5] = list_temp_N_adjuested
        df_result[6] = list_temp_value_adjuested
        df_result[7] = list_temp_CI_adjuested
        df_result[8] = list_temp_p_adjuested
        columns = ['name','N', 'β', '95%CI', 'P-value','N_adjuested', 'β_adjuested', '95%CI_adjuested', 'P-value_adjuested']
    df_result.columns=columns
    return df_result,list_error

def multivariate_analysis(df_input, continue_exposure_variables, categorical_exposure_variables, dependent_variable, model_name='logit',
                          adjust_variables=None, categorical_exposure_variable_ref=None,time_variable=None, decimal_num=3, savePath=None,style=1):
    """
    df_input:Dataframe
    continue_exposure_variables: list 自变量(定量)
    categorical_exposure_variables: list 自变量(定类)
    dependent_variable： str 因变量
    model_name: str 回归模型名称    {'logit':'逻辑回归','ols'：'线性回归','poisson':'泊松回归','cox'：‘cox回归’}
    adjust_variables： list 调整变量
    categorical_exposure_variable_ref: list 自变量(定类)参考标签
    time_variable: str 时间变量
    decimal_num：int
    savePath： str 图片路径
    style:int 图片风格
    """
    plot_name_list = []
    list_error=[]
    df_input=df_input.dropna(subset=[dependent_variable])
    if model_name=='logit':
        dv_unique=pd.unique(df_input[dependent_variable])
        if set(dv_unique)!=set([0,1]):
            return {'error': '因变量只允许取值为0、1的二分类数据。'+str(dependent_variable)+':'+str(dv_unique)}

    #-------连续变量-----------
    df_result,list_error_temp= _continue_multivariate_analysis(df_input=df_input, continue_exposure_variables=continue_exposure_variables,
                                                 dependent_variable=dependent_variable, model_name=model_name,
                                                 adjust_variables=adjust_variables, time_variable=time_variable,
                                                 decimal_num=decimal_num)
    if len(list_error_temp)>0:
        list_error=list_error+list_error_temp
    #-------分类变量-----------
    # 控制分类变量的分类数
    str_variables_control = variables_control(df_input, categorical_exposure_variables, model_name)
    if str_variables_control:
        return {'error': str_variables_control}

    if categorical_exposure_variable_ref is None:
        categorical_independent_variables_ref= [np.nan] * len(categorical_exposure_variables)
    for i,categorical_exposure_variable in enumerate(categorical_exposure_variables):
        try:
            df_temp= _categorical_multivariate_analysis(df_input=df_input, categorical_exposure_variable=categorical_exposure_variable, dependent_variable=dependent_variable, model_name=model_name, adjust_variables=None,
                                                   categorical_exposure_variable_ref=categorical_exposure_variable_ref[i],time_variable=time_variable,decimal_num=decimal_num)
            if adjust_variables is not None:
                df_temp=pd.concat([df_temp, _categorical_multivariate_analysis(df_input=df_input, categorical_exposure_variable=categorical_exposure_variable, dependent_variable=dependent_variable, model_name=model_name, adjust_variables=adjust_variables,
                                                                         categorical_exposure_variable_ref=categorical_exposure_variable_ref[i],time_variable=time_variable,decimal_num=decimal_num).drop(['name'], axis=1)], axis=1)
                columns = ['name', 'N', 'β', '95%CI', 'P-value', 'N_adjuested', 'β_adjuested', '95%CI_adjuested',
                           'P-value_adjuested']
                df_temp.columns = columns
            if df_result.empty:
               df_result=df_temp
            else:
               df_result=df_result.append(df_temp,ignore_index=True)
        except Exception as e:
            print(repr(e))
            print(traceback.print_exc())
            list_error.append(categorical_exposure_variable)

    df_temp_forest=pd.DataFrame()
    df_temp_forest['name'] = df_result['name']
    df_temp_forest['β'] = df_result['β']
    df_temp_forest[['95%CI_l', '95%CI_h']] = pd.DataFrame(_data_format_ci(df_result['95%CI']))
    try:
        if model_name in ['logit']:
            title_name =  'OR(95%CI)'
            zero = 1
        elif model_name in ['cox']:
            title_name =' HR(95%CI)'
            zero=1
        else:
            title_name = ' β(95%CI)'
            zero=0
        print(df_temp_forest)
        froest_plot_path = x5r.R_froest_plot(df_input=df_temp_forest, title=title_name, name='name',
                                             mean='β',lowerb="95%CI_l", upperb="95%CI_h",
                                             savePath=savePath, grawid=3, tickfont=1, xlabfont=1, style=style,zero=zero)
    except Exception as e:
        print(e)
        froest_plot_path = ''
    plot_name_list.append(froest_plot_path)

    if adjust_variables is not None:
        df_temp_forest = pd.DataFrame()
        df_temp_forest['name'] = df_result['name']
        df_temp_forest['β'] = df_result['β_adjuested']
        df_temp_forest[['95%CI_l', '95%CI_h']] = pd.DataFrame(_data_format_ci(df_result['95%CI_adjuested']))
        try:
            if model_name in ['logit']:
                title_name = 'OR_adjuested(95%CI)'
                zero = 1
            elif model_name in ['cox']:
                title_name = ' HR_adjuested(95%CI)'
                zero = 1
            else:
                title_name = ' β_adjuested(95%CI)'
                zero = 0
            froest_plot_path = x5r.R_froest_plot(df_input=df_temp_forest, title=title_name, name='name',
                                                 mean='β', lowerb="95%CI_l", upperb="95%CI_h",
                                                 savePath=savePath, grawid=3, tickfont=1, xlabfont=1, style=style,zero=zero)
        except Exception as e:
            print(e)
            froest_plot_path = ''
        plot_name_list.append(froest_plot_path)

    #增加描述
    str_result = '构建' + dict_models[model_name] + '模型,进行单因素、多因素分析。\n'
    str_result+= '观察' + ','.join(continue_exposure_variables+categorical_exposure_variables) + '与因变量' + str(dependent_variable) + '之间的关系，\n'
    if adjust_variables is not None:
        str_result+='Adjusted Model ：模型的协变量为'+','.join(adjust_variables)+'\n'
    if len(list_error) > 0:
        str_result += ','.join(list_error) + '无法进行单因素和多因素分析。'
    # 替换表头
    if model_name in ['logit']:
        df_result.columns = [x.replace('β', 'OR') for x in df_result.columns]
    elif model_name in ['cox']:
        df_result.columns = [x.replace('β', 'HR') for x in df_result.columns]
    return df_result, str_result, plot_name_list


#--------------多模型回归分析-----------------------------------
def _multi_models_regression(df_input, exposure_variable, dependent_variables, label_name, model_name,
                             adjust_features_model_1=None, adjust_features_model_2=None, time_variable=None, decimal_num=3):
    df_result=pd.DataFrame()
    list_temp_N=[]
    list_temp_value=[]
    list_temp_CI = []
    list_temp_p = []
    if model_name=='cox':
        for dependent_feature in dependent_variables:
            # ---计算单变量结果
            formula = time_variable + '~' + exposure_variable
            model_result = smf.phreg(formula=formula, data=df_input, status=dependent_feature, ties='breslow').fit()
            list_temp_N.append('')
            list_temp_N.append(model_result.model.surv.n_obs)
            list_temp_value.append('')
            list_temp_value.append(
                '%s' % (_round_dec(coef_conversion(model_name, model_result.params[0]), decimal_num)))
            list_temp_CI.append('')
            list_temp_CI.append('[%s,%s]' % (_round_dec(
                coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][0]),
                decimal_num)
                                             , _round_dec(
                coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][1]),
                decimal_num)))
            list_temp_p.append('')
            list_temp_p.append(str(_round_dec(model_result.pvalues[0], decimal_num)))
            if adjust_features_model_1 is not None:
                formula = time_variable + '~' + exposure_variable + '+' + '+'.join(adjust_features_model_1)
                model_result = smf.phreg(formula=formula, data=df_input, status=dependent_feature, ties='breslow').fit()
                list_temp_N.append(model_result.model.surv.n_obs)
                list_temp_value.append('%s' % (
                    _round_dec(coef_conversion(model_name, model_result.params[0]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None)[0][0]),decimal_num)
                    , _round_dec(coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][1]), decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[0], decimal_num)))
            if adjust_features_model_2 is not None:
                formula = time_variable + '~' + exposure_variable + '+' + '+'.join(
                    adjust_features_model_1 + adjust_features_model_2)
                model_result = smf.phreg(formula=formula, data=df_input, status=dependent_feature, ties='breslow').fit()
                list_temp_N.append(model_result.model.surv.n_obs)
                list_temp_value.append('%s' % (
                    _round_dec(coef_conversion(model_name, model_result.params[0]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][0]),
                               decimal_num)
                    , _round_dec(coef_conversion(model_name, model_result.conf_int(alpha=0.05, cols=None)[0][1]),
                                 decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[0], decimal_num)))
    else:
        if model_name == 'logit':
            model_family = sma.families.Binomial()
        elif model_name == 'ols':
            model_family = sma.families.Gaussian()
        elif model_name=='poisson':
            model_family=sma.families.Poisson()
        else:
            model_family = sma.families.Gaussian()

        for dependent_feature in dependent_variables:
            #---计算单变量结果
            formula = dependent_feature + '~' + exposure_variable
            model_result = smf.glm(formula=formula, family=model_family, data=df_input).fit()
            list_temp_N.append('')
            list_temp_N.append(model_result.nobs)
            list_temp_value.append('')
            list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,model_result.params[exposure_variable]), decimal_num)))
            list_temp_CI.append('')
            list_temp_CI.append('[%s,%s]' % (_round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num)
                                            , _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
            list_temp_p.append('')
            list_temp_p.append(str(_round_dec(model_result.pvalues[exposure_variable], decimal_num)))
            if adjust_features_model_1 is not None:
                formula = dependent_feature + '~' + exposure_variable + '+' + '+'.join(adjust_features_model_1)
                model_result = smf.glm(formula=formula, family=model_family, data=df_input).fit()
                list_temp_N.append(model_result.nobs)
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,model_result.params[exposure_variable]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num)
                    , _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[exposure_variable], decimal_num)))
            if adjust_features_model_2 is not None:
                formula = dependent_feature + '~' + exposure_variable + '+' + '+'.join(adjust_features_model_1 + adjust_features_model_2)
                model_result = smf.glm(formula=formula, family=model_family, data=df_input).fit()
                list_temp_N.append(model_result.nobs)
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,model_result.params[exposure_variable]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num)
                    ,_round_dec(coef_conversion(model_name,model_result.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
                list_temp_p.append(str(_round_dec(model_result.pvalues[exposure_variable], decimal_num)))
    df_result[0]=list_temp_N
    df_result[1]=list_temp_value
    df_result[2]=list_temp_CI
    df_result[3] = list_temp_p
    columns = ['N', label_name + '(β)', '95%CI', 'P-value']
    df_result.columns=columns
    return df_result
def multi_models_regression(df_input, exposure_variable, dependent_variables, model_name='logit',
                            adjust_features_model_1=None, adjust_features_model_2=None, stratification_variable=None, time_variable=None, decimal_num=3, savePath=None,style=1):
    """
    df_input:Dataframe
    exposure_feature: str暴露变量(定量)
    dependent_features： list 因变量
    model_name: str 回归模型名称
    {'logit':'逻辑回归','ols'：'线性回归','poisson':'泊松回归'}
    adjust_features_model_1： list 调整变量1
    adjust_features_model_2： list 调整变量2
    stratification_feature:  str 分层变量(定类)
    decimal_num：int
    """
    plot_name_list = []
    df_result=pd.DataFrame()
    df_result_index =[]
    str_result_error_features=''
    if model_name=='logit':
        dependent_variables_error = []
        dependent_variables_new = []
        for dependent_feature in dependent_variables:
            df_unique_temp=pd.unique(df_input[dependent_feature].dropna())
            if set(df_unique_temp)==set([0,1]):
                dependent_variables_new.append(dependent_feature)
            else:
                dependent_variables_error.append(dependent_feature)
        dependent_variables=dependent_variables_new
        if len(dependent_variables_error)>0:
            str_result_error_features+='Logist 回归分析因变量只能包含0、1，'+','.join(dependent_variables_error)+'无法进行分析。'

    for dependent_feature in dependent_variables:
        df_result_index.append(dependent_feature)
        df_result_index.append('Crude Model')
        if adjust_features_model_1 is not None:
            df_result_index.append('Adjusted Model 1')
        if adjust_features_model_2 is not None:
            df_result_index.append('Adjusted Model 2')
    df_result['Model']=df_result_index
    # -----------计算总体结果-------------
    df_result_temp = _multi_models_regression(df_input=df_input, exposure_variable=exposure_variable, dependent_variables=dependent_variables, label_name='总体', model_name=model_name,
                                              adjust_features_model_1=adjust_features_model_1, adjust_features_model_2=adjust_features_model_2, time_variable=time_variable, decimal_num=decimal_num)
    df_result = pd.concat([df_result, df_result_temp], axis=1)
    df_temp_forest=pd.DataFrame()
    df_temp_forest['Model']=df_result['Model']
    df_temp_forest['总体(β)']=df_result['总体(β)'].replace('',np.nan)
    df_temp_forest[['95%CI_l','95%CI_h']]=pd.DataFrame(_data_format_ci(df_result['95%CI']))
    try:
        if model_name in ['logit']:
            title_name='OR(95%CI)'
            zero = 1
        elif model_name in ['cox']:
            title_name = 'HR(95%CI)'
            zero=1
        else:
            title_name = 'β(95%CI)'
            zero=0
        froest_plot_path=x5r.R_froest_plot(df_input=df_temp_forest,title=title_name,name ='Model',mean = '总体(β)',lowerb= "95%CI_l" ,
                                           upperb="95%CI_h",savePath=savePath,grawid=3,tickfont=1, xlabfont=1,style=style,zero=zero)
    except Exception as e:
        print(e)
        froest_plot_path = ''
    plot_name_list.append(froest_plot_path)
    #-----------计算分层结果-------------
    if stratification_variable is not None:
        uniq_labels = df_input[stratification_variable].unique()
        # 控制分层数
        if len(uniq_labels)>5:
            return {'error': "分层数过多"}
        uniq_labels = list(filter(is_filter, list(uniq_labels)))
        try:
            uniq_labels.sort()
        except Exception:
            print(Exception)
        for label in uniq_labels:
            df_input_temp = df_input[df_input[stratification_variable] == label]
            df_result_temp = _multi_models_regression(df_input=df_input_temp, exposure_variable=exposure_variable, dependent_variables=dependent_variables, label_name=str(stratification_variable) + '_' + str(label), model_name=model_name,
                                                      adjust_features_model_1=adjust_features_model_1, adjust_features_model_2=adjust_features_model_2, time_variable=time_variable, decimal_num=decimal_num)
            df_result = pd.concat([df_result, df_result_temp], axis=1)
            df_temp_forest = pd.DataFrame()
            df_temp_forest['Model'] = df_result['Model']
            df_temp_forest[str(stratification_variable) + '_' + str(label)+'(β)'] = df_result_temp[str(stratification_variable) + '_' + str(label)+'(β)'].replace('', np.nan)
            df_temp_forest[['95%CI_l', '95%CI_h']] = pd.DataFrame(_data_format_ci(df_result_temp['95%CI']))
            try:
                if model_name in ['logit']:
                    title_name = str(stratification_variable) + '_' + str(label)+' OR(95%CI)'
                    zero = 1
                elif model_name in ['cox']:
                    title_name = str(stratification_variable) + '_' + str(label)+' HR(95%CI)'
                    zero = 1
                else:
                    title_name = str(stratification_variable) + '_' + str(label)+' β(95%CI)'
                    zero = 0
                froest_plot_path = x5r.R_froest_plot(df_input=df_temp_forest, title=title_name, name='Model',
                                                     mean=str(stratification_variable) + '_' + str(label)+'(β)', lowerb="95%CI_l", upperb="95%CI_h",
                                                     savePath=savePath, grawid=3, tickfont=1, xlabfont=1, style=style,zero=zero)
            except Exception as e:
                print(e)
                froest_plot_path = ''
            plot_name_list.append(froest_plot_path)

    str_result = '构建' + dict_models[model_name] + '模型\n'
    str_result+= '观察' + str(exposure_variable) + '变化与因变量' + ','.join(dependent_variables) + '之间的关系，'
    if stratification_variable is not None:
        str_result+='分层变量为'+str(stratification_variable)+'。\n'
    if adjust_features_model_1 is not None:
        str_result+='Model 1：模型的协变量为'+','.join(adjust_features_model_1)+'\n'
    if adjust_features_model_2 is not None:
        str_result+='Model 2：模型的协变量为'+','.join(adjust_features_model_1+adjust_features_model_2)
    str_result+=str_result_error_features

    if model_name in ['logit']:
        df_result.columns=[x.replace('β', 'OR') for x in df_result.columns]
    elif model_name in ['cox']:
        df_result.columns=[x.replace('β', 'HR') for x in df_result.columns]
    return df_result,str_result,plot_name_list

#--------------分层分析-----------------------------------
def _stratification_regression(df_input, exposure_variable, dependent_variable, stratification_variable, model_name, adjust_features, time_variable=None, decimal_num=3):
    df_result=pd.DataFrame()
    list_temp_name=[]
    list_temp_N=[]
    list_temp_value=[]
    list_temp_CI = []
    list_temp_p = []



    if model_name=='cox':
        if stratification_variable is not None:
            list_temp_name.append(stratification_variable)
            list_temp_N.append('')
            list_temp_value.append('')
            list_temp_CI.append('')
            list_temp_p.append('')
            labels = df_input[stratification_variable].unique()
            labels = list(filter(is_filter, list(labels)))
            try:
                labels.sort()
            except Exception:
                print(Exception)
            for label in labels:
                df_input_temp = df_input[df_input[stratification_variable] == label]
                #---计算单变量结果
                if adjust_features is not None:
                    formula = time_variable + '~' + '+ '.join([exposure_variable] + adjust_features)
                else:
                    formula = time_variable + '~' + exposure_variable
                result_model = smf.phreg(formula=formula, data=df_input_temp, status=dependent_variable, ties='breslow').fit()
                list_temp_name.append(label)
                list_temp_N.append(result_model.model.surv.n_obs)
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[0]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % ( _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None)[0][0]), decimal_num),
                                                  _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None)[0][1]), decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[0], decimal_num)))
        else:
            if adjust_features is not None:
                formula = time_variable + '~' + '+ '.join([exposure_variable] + adjust_features)
            else:
                formula = time_variable + '~' + exposure_variable
            result_model = smf.phreg(formula=formula, data=df_input, status=dependent_variable, ties='breslow').fit()
            list_temp_name.append('总体')
            list_temp_N.append(result_model.model.surv.n_obs)
            list_temp_value.append(
                '%s' % (_round_dec(coef_conversion(model_name, result_model.params[0]), decimal_num)))
            list_temp_CI.append('[%s,%s]' % (
            _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[0][0]), decimal_num),
            _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[0][1]), decimal_num)))
            list_temp_p.append(str(_round_dec(result_model.pvalues[0], decimal_num)))
    else:
        model_family = sma.families.Gaussian()
        if model_name == 'logit':
            model_family = sma.families.Binomial()
        elif model_name == 'ols':
            model_family = sma.families.Gaussian()
        elif model_name == 'poisson':
            model_family = sma.families.Poisson()
        if stratification_variable is not None:
            list_temp_name.append(stratification_variable)
            list_temp_N.append('')
            list_temp_value.append('')
            list_temp_CI.append('')
            list_temp_p.append('')
            labels = df_input[stratification_variable].unique()
            labels = list(filter(is_filter, list(labels)))
            try:
                labels.sort()
            except Exception:
                print(Exception)
            for label in labels:
                df_input_temp = df_input[df_input[stratification_variable] == label]
                #---计算单变量结果
                if adjust_features is not None:
                    formula = dependent_variable + '~' + '+ '.join([exposure_variable] + adjust_features)
                else:
                    formula = dependent_variable + '~' + exposure_variable
                result_model = smf.glm(formula=formula, family=model_family, data=df_input_temp).fit()
                list_temp_name.append(label)
                list_temp_N.append(result_model.nobs)
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[exposure_variable]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % ( _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num),
                                                  _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[exposure_variable], decimal_num)))
        else:
            if adjust_features is not None:
                formula = dependent_variable + '~' + '+ '.join([exposure_variable] + adjust_features)
            else:
                formula = dependent_variable + '~' + exposure_variable
            result_model = smf.glm(formula=formula, family=model_family, data=df_input).fit()
            list_temp_name.append('总体')
            list_temp_N.append(result_model.nobs)
            list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[exposure_variable]), decimal_num)))
            list_temp_CI.append('[%s,%s]' % (
            _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 0]), decimal_num),
            _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[exposure_variable, 1]), decimal_num)))
            list_temp_p.append(str(_round_dec(result_model.pvalues[exposure_variable], decimal_num)))
    df_result['分层']=list_temp_name
    df_result['N']=list_temp_N
    df_result['β'] =list_temp_value
    df_result['95%CI']=list_temp_CI
    df_result['P-value'] =list_temp_p

    return df_result
def stratification_regression(df_input, exposure_variable, dependent_variable, stratification_variables=None, model_name='logit', adjust_variables=None, time_variable=None, decimal_num=3, savePath=None,style=1):
    """
    df_input:Dataframe
    exposure_feature: str 暴露变量(定量)
    dependent_feature： str 因变量
    stratification_features： list 分层变量(定类)
    model_name: str 回归模型名称
    {'logit':'逻辑回归','ols'：'线性回归','poisson':'泊松回归'}
    adjust_features：调整变量 list
    decimal_num：int
    """
    df_result=pd.DataFrame()
    list_error=[]
    plot_name_list = []
    df_input=df_input.dropna(subset=[dependent_variable])
    if model_name=='logit':
        dv_unique=pd.unique(df_input[dependent_variable])
        if set(dv_unique)!=set([0,1]):
            return {'error': '因变量只允许取值为0、1的二分类数据。'+str(dependent_variable)+':'+str(dv_unique)}
    # ------总体-------------
    df_temp = _stratification_regression(df_input=df_input, exposure_variable=exposure_variable,
                                         dependent_variable=dependent_variable,
                                         stratification_variable=None, model_name=model_name,
                                         adjust_features=None, time_variable=time_variable, decimal_num=decimal_num)
    if adjust_variables is not None:
        df_temp_adjust = _stratification_regression(df_input=df_input, exposure_variable=exposure_variable,
                                                    dependent_variable=dependent_variable,
                                                    stratification_variable=None,
                                                    model_name=model_name, adjust_features=adjust_variables,
                                                    time_variable=time_variable,
                                                    decimal_num=decimal_num)
        df_temp_adjust.columns = ['分层', 'N(adjusted)', 'β(adjusted)', '95%CI(adjusted)', 'P-value(adjusted)']
        df_temp = pd.concat([df_temp, df_temp_adjust[['N(adjusted)', 'β(adjusted)', '95%CI(adjusted)', 'P-value(adjusted)']]], axis=1)
        # df_temp = pd.concat([df_temp,df_temp_adjust],axis=1)
    df_result = df_result.append(df_temp, ignore_index=True)
    #-------分层-------------
    if stratification_variables is not None:
        for stratification_variable in stratification_variables:
            try:
                df_temp=_stratification_regression(df_input=df_input, exposure_variable=exposure_variable, dependent_variable=dependent_variable,
                                                   stratification_variable=stratification_variable, model_name=model_name, adjust_features=None,
                                                   time_variable=time_variable, decimal_num=decimal_num)
                if adjust_variables is not None:
                    df_temp_adjust=_stratification_regression(df_input=df_input, exposure_variable=exposure_variable, dependent_variable=dependent_variable,
                                                              stratification_variable=stratification_variable, model_name=model_name, adjust_features=adjust_variables,
                                                              time_variable=time_variable, decimal_num=decimal_num)
                    df_temp_adjust.columns=['分层','N(adjusted)','β(adjusted)','95%CI(adjusted)','P-value(adjusted)']
                    df_temp=pd.concat([df_temp,df_temp_adjust[['N(adjusted)','β(adjusted)','95%CI(adjusted)','P-value(adjusted)']]],axis=1)
                df_result=df_result.append(df_temp,ignore_index=True)
            except Exception as e:
                print(traceback.print_exc())
                list_error.append(stratification_variable)
    df_temp_forest = pd.DataFrame()
    df_temp_forest['分层'] = df_result['分层']
    df_temp_forest['β'] = df_result['β'].replace('', np.nan)
    df_temp_forest[['95%CI_l', '95%CI_h']] = pd.DataFrame(_data_format_ci(df_result['95%CI']))
    try:
        if model_name in ['logit']:
            title_name='Crude Model OR(95%CI)'
            zero=1
        elif model_name in ['cox']:
            title_name = 'Crude Model HR(95%CI)'
            zero=1
        else:
            title_name = 'Crude Model β(95%CI)'
            zero=0
        froest_plot_path=x5r.R_froest_plot(df_input=df_temp_forest,title=title_name,name ='分层',mean = "β",lowerb= "95%CI_l" ,
                                           upperb="95%CI_h",savePath=savePath,grawid=3,tickfont=1, xlabfont=1,style=style,zero=zero)
    except Exception as e:
        print(e)
        froest_plot_path = ''
    plot_name_list.append(froest_plot_path)

    if adjust_variables is not None:
        df_temp_forest = pd.DataFrame()
        df_temp_forest['分层'] = df_result['分层']
        df_temp_forest['β(adjusted)'] = df_result['β(adjusted)'].replace('', np.nan)
        df_temp_forest[['95%CI_l(adjusted)', '95%CI_h(adjusted)']] = pd.DataFrame(_data_format_ci(df_result['95%CI(adjusted)']))
        try:
            # 程序等待时间防止存入图片名称相同
            # time.sleep(1)
            if model_name in ['logit']:
                title_name = 'Adjusted Model OR(95%CI)'
                zero = 1
            elif model_name in ['cox']:
                title_name = 'Adjusted Model HR(95%CI)'
                zero = 1
            else:
                title_name = 'Adjusted Model β(95%CI)'
                zero = 0
            froest_plot_path01=x5r.R_froest_plot(df_input=df_temp_forest,title=title_name,name ='分层',mean ="β(adjusted)",lowerb= "95%CI_l(adjusted)" ,
                                                 upperb="95%CI_h(adjusted)",savePath=savePath,grawid=3,tickfont=1, xlabfont=1,style=style,zero=zero)
        except Exception as e:
            print(e)
            froest_plot_path01 = ''
        plot_name_list.append(froest_plot_path01)

    str_result = '构建' + dict_models[model_name] + '模型\n'
    str_result += '观察' + str(exposure_variable) + '变化与因变量' + str(dependent_variable) + '之间的关系，\n'
    if stratification_variables is not None:
        str_result +='对样本进行分层处理，以观察不同'+','.join(stratification_variables)+'分层样本中' + str(exposure_variable) + '对' + str(dependent_variable) + '影响的差异\n'
    if adjust_variables is not None :
        str_result += 'adjusted Model ：调整模型增加的协变量为' + ','.join(adjust_variables) + '\n'
    if len(list_error)>0:
        str_result +=','.join(list_error)+'无法进行分层分析'
    #替换表头
    if model_name in ['logit']:
        df_result.columns=[x.replace('β', 'OR') for x in df_result.columns]
    elif model_name in ['cox']:
        df_result.columns=[x.replace('β', 'HR') for x in df_result.columns]
    return df_result,str_result,plot_name_list

#--------------趋势回归分析-----------------------------------
def _categorical_trend_regression(df_input, categorical_exposure_variable, dependent_variable, model_name, adjust_variables, categorical_exposure_variable_ref=None,time_variable=None,  decimal_num=3):

    if adjust_variables is None:
        adjust_variables=[]
    if time_variable is not None:
        df_input = df_input[[categorical_exposure_variable] + [dependent_variable] + adjust_variables+[time_variable]].dropna()
    else:
        df_input = df_input[[categorical_exposure_variable] + [dependent_variable] + adjust_variables].dropna()
    try:
        df_input[categorical_exposure_variable]=df_input[categorical_exposure_variable].astype(int)
    except Exception:
        print(Exception)
    exposure_feature_label=df_input[categorical_exposure_variable].unique().tolist()
    exposure_feature_label = list(filter(is_filter, list(exposure_feature_label)))
    try:
        exposure_feature_label.sort()
    except Exception:
        print(Exception)
    exposure_feature_temp_label=exposure_feature_label.copy()
    if (categorical_exposure_variable_ref is None) or (categorical_exposure_variable_ref is np.nan):
        categorical_exposure_variable_ref=exposure_feature_label[0]
    exposure_feature_temp_label.remove(categorical_exposure_variable_ref) #生成哑变量标签
    exposure_feature_model_label=[categorical_exposure_variable + '_' + str(x) for x in exposure_feature_temp_label]
    exposure_feature_all_label = [categorical_exposure_variable + '_' + str(x) for x in exposure_feature_label]
    if adjust_variables is not None:
        features= exposure_feature_model_label + adjust_variables
    else:
        features = exposure_feature_model_label
    df_input_temp = pd.concat([df_input, pd.get_dummies(df_input[categorical_exposure_variable],
                                                        prefix=categorical_exposure_variable).iloc[:, :]], axis=1)
    #print(df_input_temp)
    if model_name=='cox':
        formula = time_variable + '~' + '+ '.join(features)
        print(formula)
        result_model = smf.phreg(formula=formula, data=df_input_temp, status=dependent_variable, ties='breslow').fit()
        df_result = pd.DataFrame()
        list_temp_name = []
        list_temp_N = []
        list_temp_value = []
        list_temp_CI = []
        list_temp_p = []
        list_temp_name.append(categorical_exposure_variable)
        list_temp_N.append('')
        list_temp_value.append('')
        list_temp_CI.append('')
        list_temp_p.append('')
        ref_label_i=0 #用于标志是否循环到参考标签
        for i,label in enumerate(exposure_feature_all_label):
            #print(label)
            list_temp_name.append(re.sub(categorical_exposure_variable + '_', '', label))
            list_temp_N.append(result_model.model.surv.n_obs)
            if label in exposure_feature_model_label:
                list_temp_value.append(
                    '%s' % (_round_dec(coef_conversion(model_name, result_model.params[i-ref_label_i]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[i-ref_label_i][0]),
                               decimal_num),
                    _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[i-ref_label_i][1]),
                               decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[i-ref_label_i], decimal_num)))
            else:
                list_temp_value.append('')
                list_temp_CI.append('')
                list_temp_p.append('')
                ref_label_i=1
        # ----添加趋势检验--------
        try:
            df_input[categorical_exposure_variable] = df_input[categorical_exposure_variable].astype(float)
            if adjust_variables is not None:
                formula = time_variable + '~' + '+ '.join([categorical_exposure_variable] + adjust_variables)
            else:
                formula = time_variable + '~' + categorical_exposure_variable
            result_model = smf.phreg(formula=formula, data=df_input, status=dependent_variable, ties='breslow').fit()
            list_temp_name.append('P for trend')
            list_temp_N.append(result_model.model.surv.n_obs)
            list_temp_value.append('%s' % (
                _round_dec(coef_conversion(model_name, result_model.params[0]),
                           decimal_num)))
            list_temp_CI.append('[%s,%s]' % (
                _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[0][0]), decimal_num),
                _round_dec(coef_conversion(model_name, result_model.conf_int(alpha=0.05, cols=None)[0][1]), decimal_num)))
            list_temp_p.append(str(_round_dec(result_model.pvalues[0], decimal_num)))
        except Exception:
            print(categorical_exposure_variable + '无法进行趋势分析')
    else:
        if model_name == 'logit':
            model_family = sma.families.Binomial()
        elif model_name == 'ols':
            model_family = sma.families.Gaussian()
        elif model_name == 'poisson':
            model_family = sma.families.Poisson()
        else:
            model_family = sma.families.Gaussian()
        formula = dependent_variable + '~' + '+ '.join(features)
        result_model = smf.glm(formula=formula, family=model_family, data=df_input_temp).fit()
        df_result = pd.DataFrame()
        list_temp_name=[]
        list_temp_N=[]
        list_temp_value=[]
        list_temp_CI = []
        list_temp_p=[]
        list_temp_name.append(categorical_exposure_variable)
        list_temp_N.append('')
        list_temp_value.append('')
        list_temp_CI.append('')
        list_temp_p.append('')
        for label in exposure_feature_all_label:
            list_temp_name.append(re.sub(categorical_exposure_variable+'_','',label))
            list_temp_N.append(sum(df_input_temp[label]))
            if label in exposure_feature_model_label:
                list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[label]), decimal_num)))
                list_temp_CI.append('[%s,%s]' % (
                    _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[label, 0]), decimal_num),
                    _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[label, 1]),
                               decimal_num)))
                list_temp_p.append(str(_round_dec(result_model.pvalues[label], decimal_num)))
            else:
                list_temp_value.append('')
                list_temp_CI.append('')
                list_temp_p.append('')
        # ----添加趋势检验--------
        try:
            df_input[categorical_exposure_variable]=df_input[categorical_exposure_variable].astype(float)
            if adjust_variables is not None:
                formula = dependent_variable + '~' + '+ '.join([categorical_exposure_variable] + adjust_variables)
            else:
                formula = dependent_variable + '~' + categorical_exposure_variable
            result_model = smf.glm(formula=formula, family=model_family, data=df_input).fit()
            list_temp_name.append('P for trend')
            list_temp_N.append(result_model.nobs)
            list_temp_value.append('%s' % (_round_dec(coef_conversion(model_name,result_model.params[categorical_exposure_variable]), decimal_num)))
            list_temp_CI.append('[%s,%s]' % (
                _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[categorical_exposure_variable, 0]), decimal_num),
                _round_dec(coef_conversion(model_name,result_model.conf_int(alpha=0.05, cols=None).loc[categorical_exposure_variable, 1]),
                           decimal_num)))
            list_temp_p.append(str(_round_dec(result_model.pvalues[categorical_exposure_variable], decimal_num)))
        except Exception:
            print(categorical_exposure_variable+'无法进行趋势分析')

    df_result['变量'] = list_temp_name
    df_result['N'] = list_temp_N
    df_result['β'] = list_temp_value
    df_result['95%CI'] = list_temp_CI
    df_result['P-value'] = list_temp_p
    return df_result
def _continuous_trend_regression(df_input, continuous_exposure_variable, dependent_variable, model_name, adjust_variables,time_variable=None,decimal_num=3):
    if adjust_variables is None:
        adjust_variables = []
    if time_variable is not None:
        df_input = df_input[
            [continuous_exposure_variable] + [dependent_variable] + adjust_variables + [time_variable]].dropna()
    else:
        df_input = df_input[[continuous_exposure_variable] + [dependent_variable] + adjust_variables].dropna()
    if len(df_input[continuous_exposure_variable].unique())<=5:
        df_result=_categorical_trend_regression(df_input=df_input, categorical_exposure_variable=continuous_exposure_variable, dependent_variable=dependent_variable, model_name=model_name,
                                      adjust_variables=adjust_variables, categorical_exposure_variable_ref=None, time_variable=time_variable,decimal_num=decimal_num)
    else:
        group_cut=np.quantile(df_input[continuous_exposure_variable],[0,1/3,2/3,1], interpolation='nearest')
        grop_cut_label=['<'+str(group_cut[1]),str(group_cut[1])+'-'+str(group_cut[2]),'>'+str(group_cut[2])]
        df_input[continuous_exposure_variable] = df_input.apply(lambda row: _group(row[continuous_exposure_variable], group_cut, 1), axis=1)
        df_result=_categorical_trend_regression(df_input=df_input, categorical_exposure_variable=continuous_exposure_variable,
                                      dependent_variable=dependent_variable, model_name=model_name,
                                      adjust_variables=adjust_variables, categorical_exposure_variable_ref=None,time_variable=time_variable,
                                      decimal_num=decimal_num)
        df_result['变量']=[str(continuous_exposure_variable)]+grop_cut_label+['P for trend']
    return df_result
def trend_regression(df_input, categorical_exposure_variables, continuous_exposure_variables, dependent_variable, model_name='logit', adjust_variable=None,
                     categorical_independent_variables_ref=None, time_variable=None,decimal_num=3,savePath=None,style=1):
    """
    df_input:Dataframe
    categorical_exposure_variables :list定类自变量(定类)
    continuous_exposure_variables : list定量自变量(定量)
    dependent_variable：因变量 str
    model_name:回归模型名称 str
    {'logit':'逻辑回归','ols'：'线性回归','poisson':'泊松回归'}
    adjust_variable：list 调整变量
    categorical_independent_variables_ref: 参考标签 list
    decimal_num：int
    """
    plot_name_list = []
    df_result=pd.DataFrame()
    list_error=[]

    df_input=df_input.dropna(subset=[dependent_variable])
    if model_name=='logit':
        dv_unique=pd.unique(df_input[dependent_variable])
        if set(dv_unique)!=set([0,1]):
            return {'error': '因变量只允许取值为0、1的二分类数据。'+str(dependent_variable)+':'+str(dv_unique)}

    if categorical_independent_variables_ref is None:
        categorical_independent_variables_ref= [np.nan] * len(categorical_exposure_variables)
    for i,categorical_exposure_variable in enumerate(categorical_exposure_variables):
        try:
            df_temp= _categorical_trend_regression(df_input=df_input, categorical_exposure_variable=categorical_exposure_variable, dependent_variable=dependent_variable, model_name=model_name, adjust_variables=None,
                                                   categorical_exposure_variable_ref=categorical_independent_variables_ref[i],time_variable=time_variable,decimal_num=decimal_num)
            if adjust_variable is not None:
               df_temp=pd.concat([df_temp, _categorical_trend_regression(df_input=df_input, categorical_exposure_variable=categorical_exposure_variable, dependent_variable=dependent_variable, model_name=model_name, adjust_variables=adjust_variable,
                                                                         categorical_exposure_variable_ref=categorical_independent_variables_ref[i],time_variable=time_variable,decimal_num=decimal_num).drop(['变量'], axis=1)], axis=1)
            if df_result.empty:
               df_result=df_temp
            else:
               df_result=df_result.append(df_temp,ignore_index=True)
        except Exception as e:
            print(repr(e))
            print(traceback.print_exc())
            list_error.append(categorical_exposure_variable)
    for continuous_exposure_variable in continuous_exposure_variables:
        try:
            df_temp = _continuous_trend_regression(df_input=df_input,continuous_exposure_variable=continuous_exposure_variable,
                                                    dependent_variable=dependent_variable, model_name=model_name,
                                                    adjust_variables=None,time_variable=time_variable,decimal_num=decimal_num)
            if adjust_variable is not None:
                df_temp = pd.concat([df_temp, _continuous_trend_regression(df_input=df_input, continuous_exposure_variable=continuous_exposure_variable,
                                                                           dependent_variable=dependent_variable, model_name=model_name,
                                                                           adjust_variables=adjust_variable,time_variable=time_variable,decimal_num=decimal_num).drop(
                    ['变量'], axis=1)], axis=1)
            if df_result.empty:
                df_result = df_temp
            else:
                df_result = df_result.append(df_temp, ignore_index=True)
        except Exception as e:
            print(repr(e))
            print(traceback.print_exc())
            list_error.append(continuous_exposure_variable)
    str_result = '观察自变量' + ','.join((categorical_exposure_variables+continuous_exposure_variables)) + '的变化趋势与因变量' + str(dependent_variable) + '之间的关系，\n'
    #str_result+= '定类变量将自动转化为定量变量进行趋势性检验，定量变量将转自动分为3组进行趋势线检验\n'
    str_result+= '定类变量将自动转化为定量变量进行趋势性检验，定量变量(连续型)将自动分为3等分组进行趋势线检验\n'
    str_result += '构建' + dict_models[model_name] + '模型\n'
    if len(list_error)>0:
        str_result +=','.join(list_error)+'无法进行趋势分析'
    if  adjust_variable is not None:
        str_result += 'Model ：调整模型的协变量为' + ','.join(adjust_variable) + '\n'
        df_result.columns = ['变量', 'N', 'β', '95%CI', 'P-value', 'N(adjusted)', 'β(adjusted)', '95%CI(adjusted)',
                             'P-value(adjusted)']
    df_temp_forest = pd.DataFrame()
    df_temp_forest['变量'] = df_result['变量']
    df_temp_forest['β'] = df_result['β'].replace('', np.nan)
    df_temp_forest[['95%CI_l', '95%CI_h']] = pd.DataFrame(_data_format_ci(df_result['95%CI']))

    try:
        if model_name in ['logit']:
            title_name = 'Crude Model OR(95%CI)'
            zero=1
        elif model_name in ['cox']:
            title_name = 'Crude Model HR(95%CI)'
            zero=1
        else:
            title_name = 'Crude Model β(95%CI)'
            zero=0
        froest_plot_path=x5r.R_froest_plot(df_input=df_temp_forest,title=title_name,name ='变量',mean = "β",lowerb= "95%CI_l" ,
                                           upperb="95%CI_h",savePath=savePath,grawid=3,tickfont=1, xlabfont=1,style=style,zero=zero)
    except Exception as e:
        print(e)
        froest_plot_path = ''
    plot_name_list.append(froest_plot_path)
    if  adjust_variable is not None:
        df_temp_forest = pd.DataFrame()
        df_temp_forest['变量'] = df_result['变量']
        df_temp_forest['β(adjusted)'] = df_result['β(adjusted)'].replace('', np.nan)
        df_temp_forest[['95%CI_l(adjusted)', '95%CI_h(adjusted)']] = pd.DataFrame(
            _data_format_ci(df_result['95%CI(adjusted)']))
        try:
            if model_name in ['logit']:
                title_name = 'Adjusted Model OR(95%CI)'
                zero = 1
            elif model_name in ['cox']:
                title_name = 'Adjusted Model HR(95%CI)'
                zero = 1
            else:
                title_name = 'Adjusted Model β(95%CI)'
                zero = 0
            froest_plot_path01=x5r.R_froest_plot(df_input=df_temp_forest,title=title_name,name ='变量',mean ="β(adjusted)",lowerb= "95%CI_l(adjusted)" ,
                                                 upperb="95%CI_h(adjusted)",savePath=savePath,grawid=3,tickfont=1, xlabfont=1,style=style,zero=zero)
        except Exception as e:
            print(e)
            froest_plot_path01 = ''
        plot_name_list.append(froest_plot_path01)

    # 替换表头

    if model_name in ['logit']:
        df_result.columns = [x.replace('β', 'OR') for x in df_result.columns]
    elif model_name in ['cox']:
        df_result.columns = [x.replace('β', 'HR') for x in df_result.columns]
    return df_result,str_result,plot_name_list

#--------------平滑曲线（线性相加模型）分析-----------------------------------
def smooth_curve_fitting_analysis(df_input, exposure_variable, dependent_variable, inflection_list=None,
                                  adjust_variable=None, decimal_num=3,savePath=None,style=1):
    """
    df_input:Dataframe
    exposure_variable:暴露变量 str
    dependent_variable：因变量 str
    inflection_list：折点 list (输入数字list)
    adjust_variable：调整变量 list
    decimal_num:int 小数点
    """
    plot_name_list = []
    # plt.clf()
    # plt.figure()
    def _group(x, group_list,):
        for i in range(1, len(group_list)):
            if (x <= group_list[i]):
                if i==1:
                    result_str='≤%.1f'%group_list[i]
                elif  i==(len(group_list)-1):
                    result_str='>%.1f'%group_list[i-1]
                else:
                    result_str='%.1f<X≤%.1f'%(group_list[i-1],group_list[i])
                return result_str

    def GAM_simple(df_input, exposure_feature, dependent_feature, adjust_features=None,savePath=None):
        features = []
        features.append(exposure_feature)
        features.append(dependent_feature)
        if adjust_features is not None:
            features += adjust_features
        df_temp = df_input[features].dropna()
        if adjust_features is not None:
            x = df_temp[[exposure_feature] + adjust_features]
        else:
            x = df_temp[[exposure_feature]]
        y = df_temp[dependent_feature]
        gam = LinearGAM().fit(x, y)
        XX = gam.generate_X_grid(term=0)
        plt.figure()
        plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
        plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX, width=.95)[1], c='r', ls='--')
        plt.title('广义相加模型(GAM)');
        time_=str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().second)
        plot_name="广义相加模型(GAM)"+time_+".jpeg"
        plt.savefig(savePath+plot_name, bbox_inches="tight")
        plot_name_list = []
        plot_name_list.append(plot_name)
        plt.close()
        return plot_name_list
    ax0=GAM_simple(df_input=df_input, exposure_feature=exposure_variable, dependent_feature=dependent_variable, adjust_features=adjust_variable,savePath=savePath)
    plot_name_list.append(ax0[0])
    plt.figure()
    ax1=sns.lmplot(x=exposure_variable, y=dependent_variable, data=df_input,
                   scatter=True, scatter_kws={'s':2}, order=3)
    plt.title('多项式回归模型(n=3)')
    time_=str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().second)
    plot_name="多项式回归模型(n=3)"+time_+".jpeg"
    plt.savefig(savePath+plot_name, bbox_inches="tight")
    plot_name_list.append(plot_name)
    plt.close()
    if inflection_list is not None:
        inflection_list= [df_input[exposure_variable].min()] + inflection_list
        inflection_list.append(df_input[exposure_variable].max())
        df_input['折点分组'] = df_input.apply(lambda row: _group(row[exposure_variable], inflection_list), axis=1)
        plt.figure()
        ax2=sns.lmplot(x=exposure_variable, y=dependent_variable, data=df_input, hue='折点分组', scatter_kws={'s':2}, legend=False)
        plt.legend(loc='best')
        plt.title('折点分段线性拟合模型')
        plot_name="折点分段线性拟合模型"+time_+".jpeg"
        plt.savefig(savePath+plot_name, bbox_inches="tight")
        plot_name_list.append(plot_name)
        plt.close()
        df_result,str_result_,plt1=stratification_regression(df_input=df_input, exposure_variable=exposure_variable, dependent_variable=dependent_variable,
                                                       stratification_variables=['折点分组'], model_name='ols', adjust_variables=adjust_variable
                                                       , decimal_num=decimal_num,savePath=savePath,style=style)
        for i in plt1:
            plot_name_list.append(i)
    else:
        df_result, str_result_,plt2 = stratification_regression(df_input=df_input, exposure_variable=exposure_variable,
                                                          dependent_variable=dependent_variable, stratification_variables=None, model_name='ols',
                                                          adjust_variables=adjust_variable, decimal_num=decimal_num,savePath=savePath,style=style)
        for i in plt2:
            plot_name_list.append(i)
    str_result='探索自变量%s和因变量%s之间的非线性关系，构建非线性模型：\n 1：广义相加模型(GAM); 2：多项式回归模型（n=3）'%(exposure_variable,dependent_variable)
    if inflection_list is not None:
        str_result+='3:折点分段线性拟合模型'
    # plt.close()
    # print(plot_name_list)
    return plot_name_list,df_result,str_result

# ---------二组独立样本ROC分析----------------
def two_groups_roc(df_input, features, group, title=None,decimal_num=3,savePath=None,palette_style='nejm'):
    """
    df_input:Dataframe
    features:自变量list
    group：因变量str
    title: 标题 str
    decimal_num:小数点位数
    savePath:存储路径
    """
    if len(features) > 10:
        return {'error':'分析变量数不能超过10个'}
    palette_dict = {
        'lancet': ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF",
                   "#ADB6B6FF", "#1B1919FF"],
        'nejm': ["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF",
                 "#BC3C29FF"],
        'jama': ["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF", "#80796BFF", "#374E55FF",
                 "#DF8F44FF"],
        'npg': ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF",
                "#7E6148FF", "#B09C85FF"]}
    color_camp = palette_dict[palette_style]#palette_dict['jama']
    df_result = pd.DataFrame(columns=['特征', 'N', 'AUC值', '灵敏度', '特异度', '约登指数', '最佳阈值'])
    plot_name_list = []
    plt.clf()
    plt.figure(figsize=[7, 6], dpi=600)
    data_class = len(pd.value_counts(df_input[group]))#类的数量
    if data_class > 9:
        return {'error':'分析变量类的数目不能超过9个'}
    data_feature_class = len(features)
    if data_class > 2 and data_feature_class == 1:  #处理变量数等于1的，分类数大于2的
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()
        feature = features[0]

        #y_result = label_binarize(df_input[group], classes=[i for i in range(data_class)])# 将标签二值化
        u = np.unique(np.array(df_input[group]))
        y_result = label_binarize(df_input[group], classes=[ii for ii in u])  # 将标签二值化
        for i in range(data_class):
            df_feature_result = pd.concat([df_input[features], pd.DataFrame(y_result,
                                columns=[ii for ii in range(data_class)])], axis=1).dropna()

            y = df_feature_result[i]
            x = df_feature_result[feature]
            fpr[i], tpr[i], thresholds[i] = roc_curve(y, x)
            roc_auc[i] = _round_dec(auc(fpr[i], tpr[i]), decimal_num)
            yuden = (tpr[i] - fpr[i])
            yuden_max = _round_dec(max(yuden), decimal_num)
            yuden_index = np.argmax(yuden)
            yuden_Sensitivity = _round_dec(tpr[i][yuden_index], decimal_num)  # 灵敏性
            yuden_Specificity = _round_dec((1 - fpr[i][yuden_index]), decimal_num)  # 特异性
            cut_off = _round_dec(thresholds[i][yuden_index], 3)
            new = pd.DataFrame({'特征': feature+str(u[i]), 'N': len(x), 'AUC值': roc_auc[i], '灵敏度': yuden_Sensitivity,
                                '特异度': yuden_Specificity, '约登指数': yuden_max,
                                '最佳阈值': cut_off}, index=["0"])
            df_result = df_result.append(new, ignore_index=True)
            # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
            plt.plot(fpr[i], tpr[i], lw=1.5, color=color_camp[i],
                     label=feature + '-' + str(int(u[i]))+'(AUC = %0.2f)' % (roc_auc[i]))

        #X_input = pd.Series(np.zeros(shape=df_input[feature].shape))
        X_input = np.stack([df_input[feature] for _ in range(data_class)], axis=1)
        #for i in range(data_class):
        #    X_input = pd.concat([X_input, df_input[feature]], axis=1)
        fpr["micro"], tpr["micro"], thresholds['micro'] = roc_curve(np.array(y_result).ravel(), np.array(X_input).ravel())

        roc_auc["micro"] = _round_dec(auc(fpr["micro"], tpr["micro"]), decimal_num)
        yuden = (tpr["micro"] - fpr["micro"])
        yuden_max = _round_dec(max(yuden), decimal_num)
        yuden_index = np.argmax(yuden)
        yuden_Sensitivity = _round_dec(tpr["micro"][yuden_index], decimal_num)  # 灵敏性
        yuden_Specificity = _round_dec((1 - fpr["micro"][yuden_index]), decimal_num)  # 特异性
        cut_off = _round_dec(thresholds["micro"][yuden_index], 3)
        new = pd.DataFrame({'特征': feature+"-micro", 'N': len(df_input[feature]), 'AUC值': roc_auc["micro"], '灵敏度': yuden_Sensitivity,
                            '特异度': yuden_Specificity, '约登指数': yuden_max,
                            '最佳阈值': cut_off}, index=["0"])
        df_result = df_result.append(new, ignore_index=True)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr["micro"], tpr["micro"], lw=1.5, color=color_camp[data_class+1],
                 label="micro-average-"+feature + '(AUC = %0.2f)' % (roc_auc["micro"]))

        all_fpr, all_fpr_index = np.unique(np.concatenate([fpr[ii] for ii in range(data_class)]), return_index=True)
        a_thresholds = np.concatenate([thresholds[ii] for ii in range(data_class)])
        all_thresholds = a_thresholds[all_fpr_index]
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(data_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= data_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        thresholds["macro"] = all_thresholds
        roc_auc["macro"] = _round_dec(auc(fpr["macro"], tpr["macro"]), decimal_num)
        yuden = (tpr["macro"] - fpr["macro"])
        yuden_max = _round_dec(max(yuden), decimal_num)
        yuden_index = np.argmax(yuden)
        yuden_Sensitivity = _round_dec(tpr["macro"][yuden_index], decimal_num)  # 灵敏性
        yuden_Specificity = _round_dec((1 - fpr["macro"][yuden_index]), decimal_num)  # 特异性
        cut_off = _round_dec(thresholds["macro"][yuden_index], 3)
        new = pd.DataFrame(
            {'特征': feature+"-macro", 'N': len(df_input[feature]), 'AUC值': roc_auc["macro"], '灵敏度': yuden_Sensitivity,
             '特异度': yuden_Specificity, '约登指数': yuden_max,
             '最佳阈值': cut_off}, index=["0"])
        df_result = df_result.append(new, ignore_index=True)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr["macro"], tpr["macro"], lw=1.5, color=color_camp[data_class+2],
                 label="macro-average-" + feature + '(AUC = %0.2f)' % (roc_auc["macro"]))

    elif data_class > 2 and data_feature_class > 1:#处理变量数大于1的，分类数大于2的
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()

        u = np.unique(np.array(df_input[group]))
        y_result = label_binarize(df_input[group], classes=[ii for ii in u])  # 将标签二值化
        for i, feature in enumerate(features):
            df_feature_result = pd.concat([df_input[feature], pd.DataFrame(y_result,
                                columns=[ii for ii in range(data_class)])], axis=1).dropna()
            for j in range(data_class):
                y = df_feature_result[j]
                x = df_feature_result[feature]
                fpr[j], tpr[j], thresholds[j] = roc_curve(y, x)

            X_input = np.stack([df_input[feature] for _ in range(data_class)], axis=1)
            fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(np.array(y_result).ravel(), np.array(X_input).ravel())
            roc_auc["micro"] = _round_dec(auc(fpr["micro"], tpr["micro"]), decimal_num)
            yuden = (tpr["micro"] - fpr["micro"])
            yuden_max = _round_dec(max(yuden), decimal_num)
            yuden_index = np.argmax(yuden)
            yuden_Sensitivity = _round_dec(tpr["micro"][yuden_index], decimal_num)  # 灵敏性
            yuden_Specificity = _round_dec((1 - fpr["micro"][yuden_index]), decimal_num)  # 特异性
            cut_off = _round_dec(thresholds["micro"][yuden_index], 3)
            new = pd.DataFrame(
                {'特征': feature+"-micro", 'N': len(df_input[feature]), 'AUC值': roc_auc["micro"], '灵敏度': yuden_Sensitivity,
                 '特异度': yuden_Specificity, '约登指数': yuden_max,
                 '最佳阈值': cut_off}, index=["0"])
            df_result = df_result.append(new, ignore_index=True)
            # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
            plt.plot(fpr["micro"], tpr["micro"], lw=1.5, color=color_camp[i],
                     label="micro-average-"+feature + '(AUC = %0.2f)' % (roc_auc["micro"]))

            all_fpr, all_fpr_index = np.unique(np.concatenate([fpr[ii] for ii in range(data_class)]), return_index=True)
            a_thresholds = np.concatenate([thresholds[ii] for ii in range(data_class)])
            all_thresholds = a_thresholds[all_fpr_index]
            mean_tpr = np.zeros_like(all_fpr)
            for j in range(data_class):
                mean_tpr += interp(all_fpr, fpr[j], tpr[j])
            mean_tpr /= data_class
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            thresholds["macro"] = all_thresholds

            roc_auc["macro"] = _round_dec(auc(fpr["macro"], tpr["macro"]), decimal_num)
            yuden = (tpr["macro"] - fpr["macro"])
            yuden_max = _round_dec(max(yuden), decimal_num)
            yuden_index = np.argmax(yuden)
            yuden_Sensitivity = _round_dec(tpr["macro"][yuden_index], decimal_num)  # 灵敏性
            yuden_Specificity = _round_dec((1 - fpr["macro"][yuden_index]), decimal_num)  # 特异性
            cut_off = _round_dec(thresholds["macro"][yuden_index], 3)
            new = pd.DataFrame(
                {'特征': feature+"-macro", 'N': len(df_input[feature]), 'AUC值': roc_auc["macro"], '灵敏度': yuden_Sensitivity,
                 '特异度': yuden_Specificity, '约登指数': yuden_max,
                 '最佳阈值': cut_off}, index=["0"])
            df_result = df_result.append(new, ignore_index=True)
            # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
            plt.plot(fpr["macro"], tpr["macro"], lw=1.5, color=palette_dict['jama'][i],
                     label="macro-average-" + feature + '(AUC = %0.2f)' % (roc_auc["macro"]))

    elif data_class == 2:
        for i, feature in enumerate(features):
            df_input_temp = df_input[[feature, group]].dropna()
            u = np.unique(np.array(df_input_temp[group]))
            y = label_binarize(df_input_temp[group], classes=[ii for ii in u])  # 将标签二值化
            #y = df_input_temp[group]
            x = df_input_temp[feature]
            fpr, tpr, thresholds = roc_curve(y, x)
            roc_auc = _round_dec(auc(fpr, tpr),decimal_num)
            yuden = (tpr - fpr)
            yuden_max = _round_dec(max(yuden),decimal_num)
            yuden_index = np.argmax(yuden)
            yuden_Sensitivity = _round_dec(tpr[yuden_index],decimal_num)#灵敏性
            yuden_Specificity = _round_dec((1-fpr[yuden_index]),decimal_num)#特异性
            cut_off=_round_dec(thresholds[yuden_index],3)
            new = pd.DataFrame({'特征':feature,'N':len(x),'AUC值':roc_auc,'灵敏度':yuden_Sensitivity,
                          '特异度':yuden_Specificity,  '约登指数':yuden_max,
                          '最佳阈值':cut_off},index=["0"])
            df_result = df_result.append(new, ignore_index=True)
            # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
            plt.plot(fpr, tpr, lw=1.5, color=color_camp[i],
                 label=feature+'(AUC = %0.2f)' % (roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='base line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # plt.style.use('seaborn')
    if title is None:
        title = str(group)+'ROC曲线'
    plt.title(title)
    plt.legend(loc="lower right")
    ax_ = plt.gca()
    ax_.spines['top'].set_visible(False)  # 去掉上边框
    ax_.spines['right'].set_visible(False)  # 去掉右边框
    plot_name = "roc.jpeg"
    plt.savefig(savePath + plot_name, bbox_inches="tight")
    plot_name_list.append(plot_name)
    plt.close()
    str_result = '生成接受者操作特性曲线（receiver operating characteristic curve，ROC曲线），同时自动计算各个变量的约登指数和最佳阈值。'
    return df_result,str_result,plot_name_list

#---------------------------多组生存估算模型----------------------------
def _lifetime_plot(df_input,time_column,event_column,sorted=False):
    """
    生存时间绘图
    df_input:Dataframe
    time_column:str
    event_column：str(生存为0，死亡为1)
    """
    n=df_input.shape[0]
    max_time=max(df_input[time_column])
    max_event_time=max(df_input[df_input[event_column] == 1][time_column])
    observed_lifetimes=df_input[time_column]
    death_observed=df_input[event_column]
    ax = plot_lifetimes(observed_lifetimes, event_observed=death_observed,sort_by_duration=sorted)
    ax.set_xlim(0, max_time)
    ax.vlines(max_event_time, 0, n, lw=2, linestyles='--')
    ax.set_xlabel("time")
    ax.set_title('样本生存时间线图')
    return plt
def _Univariate_Estimating_Models(df_input, time_column, event_column, model_type):
    # df_result=pd.DataFrame()
    df_columns = ['总数', '事件数', '删失数', '删失率', '中位数', '95%CI（下限）', '95%CI(上限）']
    N = len(df_input[event_column])
    event_N = df_input[df_input[event_column] == 1].shape[0]
    censored_N = N - event_N
    censored_rate = round(censored_N / N, 3)
    eum = KaplanMeierFitter()
    eum.fit(df_input[time_column], event_observed=df_input[event_column], label=event_column)
    median_ = eum.median_survival_time_
    median_confidence_interval_ = median_survival_times(eum.confidence_interval_)
    df_result_data = [[N, event_N, censored_N, censored_rate, median_, median_confidence_interval_.iloc[0, 0],
                       median_confidence_interval_.iloc[0, 1]]]
    df_result = pd.DataFrame(data=df_result_data, columns=df_columns)
    result_str = event_column + '的患者生存时间中位数为' + str(median_) + '，中位数95%置信区间为[' + \
                 str(median_confidence_interval_.iloc[0, 0]) + ',' + str(median_confidence_interval_.iloc[0, 1]) + ']'
    if (model_type == 1):
        eum.plot()
        plt.title('Kaplan-Meier生存曲线')
    if (model_type == 2):
        eum = NelsonAalenFitter()
        eum.fit(df_input[time_column], event_observed=df_input[event_column], label=event_column)
        eum.plot()
        plt.title('Cumulative hazard风险累积曲线')
    return df_result, result_str, plt
def _Multi_Univariate_Estimating_Models(df_input, time_column, event_column, group_column, model_type):
    labels = df_input[group_column].unique()
    try:
        labels.sort()
    except Exception:
        print(Exception)
    fig = plt.figure()
    ax = plt.subplot(111)
    result_str = ''
    df_result=pd.DataFrame()
    for label in labels:
        temp=df_input[event_column][df_input[group_column] == label]
        N = len(temp)
        event_N = sum(temp)
        censored_N = N - event_N
        censored_rate = round(censored_N / N, 3)
        eum = KaplanMeierFitter()
        eum.fit(df_input[time_column][df_input[group_column] == label],
                temp,
                label=group_column + str(label))
        if model_type==1:
            ax = eum.plot(ax=ax)
        median_confidence_interval_ = median_survival_times(eum.confidence_interval_)
        result_str +=  str(group_column) +'为'+ str(label) + '的患者中' + event_column + '存活中位数为' + str(
            eum.median_survival_time_) + '存活50%对应的存活时间95%置信区间：[' + \
                     str(median_confidence_interval_.iloc[0, 0]) + ',' + str(median_confidence_interval_.iloc[0, 1]) + ']\n'
        df_result[label]=[N, event_N, censored_N, censored_rate, eum.median_survival_time_, median_confidence_interval_.iloc[0, 0],
        median_confidence_interval_.iloc[0, 1]]
    df_result=df_result.T
    df_result.columns=['总数', '事件数', '删失数', '删失率', '中位数', '95%CI（下限）', '95%CI(上限）']
    plt.title('Kaplan-Meier生存曲线')
    if (model_type == 2):
        for label in labels:
            eum = NelsonAalenFitter()
            eum.fit(df_input[time_column][df_input[group_column] == label],
                    df_input[event_column][df_input[group_column] == label],
                    label=group_column + str(label))
            ax = eum.plot(ax=ax)
        plt.title('Cumulative hazard风险累积曲线')
    log_rank_result = multivariate_logrank_test(df_input[time_column], df_input[group_column], df_input[event_column])
    result_str += ',' + str(len(labels)) + '组患者的生存时间的Log Rank秩和检验统计量为' +str(round(log_rank_result.test_statistic,3))+\
                  ',P(sig.)='+str(round(log_rank_result.p_value,3))+'。'
    if log_rank_result.p_value>0.05:
        result_str+='按照Log Rank秩和检验结果，认为不同'+group_column+'的患者生时间没有差异。'
    elif log_rank_result.p_value<=0.05:
        result_str+='按照Log Rank秩和检验结果，认为不同'+group_column+'的患者生时间存在差异。'

    return df_result,result_str,plt
def survival_estimating_models(df_input,time_column,event_column,group_column=None,model_type=1,sorted=False,intervals=4):
    """
    多组生存估算模型
    df_input:Dataframe
    time_column:str 时间变量
    event_column：str(生存为0，死亡为1)
    group_column：str 分组变量
    model_type: int  1 km曲线 2 cumulative_hazard,3 样本生存时间曲线
    说明：选3需要设置是否排序
    """
    dict_df_result = {}
    if group_column is None:
        df_input=df_input[[time_column,event_column]].dropna()
    else:
        df_input=df_input[[time_column,event_column,group_column]].dropna()

    if model_type==3:
        df_result=None
        str_result=''
        plt=_lifetime_plot(df_input,time_column,event_column,sorted)
    else:
        if group_column is None:
            df_result1,str_result,plt= _Univariate_Estimating_Models(df_input,time_column,event_column,model_type)
        else:
            df_result1,str_result,plt =_Multi_Univariate_Estimating_Models(df_input, time_column, event_column, group_column, model_type)
        df_result2 = survival_table_from_events(df_input[time_column], df_input[event_column],
                                               collapse=True, intervals=intervals)
        df_result2.columns = ['数量', '事件发生', '删失', '留存数']
        # df_result2=df_result2.T
        df_result2.reset_index(inplace=True)
        if group_column is not None:
            df_result1.reset_index(inplace=True)
            dic_rename = {'index': group_column}
            df_result1.rename(columns=dic_rename, inplace=True)
        dict_df_result['生存中位数表']=df_result1
        dict_df_result['生存删失表'] = df_result2

    return dict_df_result,str_result,plt


#---------------------------生存回归模型----------------------------
def cox_model(df_input, time_column, event_column,groups=None, features=None):
    """
    COX回归分析
    df_input:Dataframe
    time_column:str 时间变量
    event_column：str(生存为0，死亡为1)
    groups:分组 list 默认None
    features：list 默认None
    """
    if features is not None:
        features.append(time_column)
        features.append(event_column)
        if groups is not None:
            df_input=df_input[features+groups]
        else:
            df_input = df_input[features]
    df_input=df_input.dropna()
    cph = CoxPHFitter()
    cph.fit(df_input,duration_col=time_column, event_col=event_column,show_progress=False)
    df_result=cph.summary
    df_result=df_result[['coef','se(coef)','exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','z','p']]
    df_result.columns=['B','se','HR','HR lower 95%','HR upper 95%','z','p']
    df_result=df_result.round(3)
    str_result='总观测人数：'+str(cph.weights.sum())+',发生事件人数：'+str(cph.weights[cph.event_observed > 0].sum())+\
               ',COX模型的最大对数似然值(log_likelihood)为'+str(round(cph.log_likelihood_,3))+\
               ',赤池信息量准则值(AIC)为'+str(round(cph.AIC_partial_,3))+\
               ',一致性(Concordance)为'+str(round(cph.concordance_index_,3))+'。\n从系数统计表中可以看出,'
    labels_temp = []
    if groups is not None:
        for gc in groups:
            labels_temp.append(df_input[gc].unique())
            if df_result.loc[gc,'p']<0.05:
                str_result+=gc+'是影响'+event_column+'结局的独立因素(P='+str(round(df_result.loc[gc,'p'],3))+')。\n'
            str_result+=gc+'的HR='+str(df_result.loc[gc,'HR'])+'(95%CI:'+str(df_result.loc[gc,'HR lower 95%'])+\
                        '-'+str(df_result.loc[gc,'HR lower 95%'])+')，'
            if df_result.loc[gc,'HR']>=1:
                str_result+='HR大于1说明与对照组相比'+gc+'是'+event_column+'发生的危险因素。\n'
            else:str_result+='HR小于1说明与对照组相比'+gc+'是'+event_column+'发生的保护因素。\n'
        labels = list(itertools.product(*labels_temp))
        ax1 = cph.plot()#fmt="o", color="blue", ecolor='grey'
        plt.title('变量系数森林图')
        ax2 = cph.plot_partial_effects_on_outcome(covariates=groups, values=labels, cmap='coolwarm')
        plt.title('COX生存曲线')
    else:
        ax1 = cph.plot()
        plt.title('变量系数森林图')
        # ax2 = cph.plot_covariate_groups(covariates=None, values=None)
        # plt.title('COX生存曲线')
    df_result.index.name = '变量名'
    df_result.reset_index(inplace=True)
    return df_result,str_result,plt

#---------------------------智能分组分析----------------------------
def smart_goup_analysis(df_input, x_feature, y_feature,path,group_=None, group_num=5, group_cut=None, type=1,decimal_num=3, palette_style='nejm'):
    """
    智能分组分析
    df_input：DataFrame 输入的待处理数据
    x_features：str 自变量：百分位分组变量 （定量数据或定类）
    y_features：str 因变量 （定量数据）
    gourp_：str 分层变量   （定类数据）
    group_num：int 分组数
    group_cut：list 数值分割点（需要从最小值开始，最大值结束）
    type：int 类型 1 （Mean±SD） 2  （Median[Q1-Q3]）
    decimal_num: int  小数点位数
    palette_style:str 图片风格参数 （nejm、lancet、jama、npg）
    """
    # -------------智能分组分析内部函数------------------
    palette_dict={'lancet':["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF", "#1B1919FF"],
    'nejm':["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF", "#BC3C29FF"],
    'jama':["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF", "#80796BFF", "#374E55FF", "#DF8F44FF"],
    'npg':["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF"]}
    color_camp=palette_dict[palette_style]
    def _groupdata_format(df_input, value_feature, group_feature, label, type, color, decimal_num=3):
        if type == 1:
            df_result = pd.DataFrame(columns=['分组', label + '(N)', '均数', '方差'])
            labels = df_input[group_feature].unique()
            try:
                labels.sort()
            except Exception:
                print(Exception)
            N_temp = []
            mean_temp = []
            std_temp = []
            for i in labels:
                list_temp = df_input.loc[df_input[group_feature] == i][value_feature]
                N_temp.append(len(list_temp))
                mean_temp.append(np.mean(list_temp))
                std_temp.append(np.std(list_temp))
            df_result['分组'] = labels
            df_result[label + '(N)'] = N_temp
            df_result['均数'] = mean_temp
            df_result['方差'] = std_temp
            ax = plt.errorbar(labels, mean_temp, fmt="s:", capsize=4,label=label, yerr=std_temp,c=color)
            df_result[[label + '(N)', '均数', '方差']] = df_result[[label + '(N)', '均数', '方差']].applymap(
                lambda x: _round_dec(x, decimal_num))
            df_result[label + '(Mean±SD)'] = df_result['均数'].astype(str) + '±' + df_result['方差'].astype(str)
            df_result.drop(columns=['均数', '方差'], inplace=True)
        elif type == 2:
            df_result = pd.DataFrame(columns=['分组', label + '(N)', '中位数', 'Q1', 'Q3'])
            labels = df_input[group_feature].unique()
            try:
                labels.sort()
            except Exception:
                print(Exception)
            N_temp = []
            median_temp = []
            q1_temp = []
            q3_temp = []
            for i in labels:
                list_temp = df_input.loc[df_input[group_feature] == i][value_feature]
                N_temp.append(len(list_temp))
                median_temp.append(np.median(list_temp))
                q1_temp.append(np.quantile(list_temp, q=0.25, interpolation='nearest'))
                q3_temp.append(np.quantile(list_temp, q=0.75, interpolation='nearest'))
            df_result['分组'] = labels
            df_result[label + '(N)'] = N_temp
            df_result['中位数'] = median_temp
            df_result['Q1'] = q1_temp
            df_result['Q3'] = q3_temp
            yerr = np.array([np.subtract(median_temp, q1_temp), np.subtract(q3_temp, median_temp)])
            ax = plt.errorbar(labels, median_temp, fmt="s:", capsize=4,label=label, yerr=yerr,c=color)
            df_result[[label + '(N)', '中位数', 'Q1', 'Q3']] = df_result[[label + '(N)', '中位数', 'Q1', 'Q3']].applymap(
                lambda x: _round_dec(x, decimal_num))
            df_result[label + '(Median[Q1-Q3])'] = df_result['中位数'].astype(str) + '[' + df_result['Q1'].astype(
                str) + ',' + df_result['Q3'].astype(str) + ']'
            df_result.drop(columns=['中位数', 'Q1', 'Q3'], inplace=True)
        return df_result, ax
    def _p_trend_for_smart_goup_analysis(df_input, dependent_variable, independent_variable):
        formula = dependent_variable + '~' + independent_variable
        result_model = smf.glm(formula=formula, family=sma.families.Gaussian(), data=df_input).fit()
        or_ = result_model.params[independent_variable]
        ci_0 = result_model.conf_int(alpha=0.05, cols=None).loc[independent_variable, 0]
        ci_1 = result_model.conf_int(alpha=0.05, cols=None).loc[independent_variable, 1]
        p = result_model.pvalues[independent_variable]
        return or_, ci_0, ci_1, p
    if group_ is not None:
        df_input = df_input[[x_feature, y_feature, group_]].dropna()
        if len(df_input[group_].unique())>9:
            return {'error': '分组数过多'}
    else:
        df_input = df_input[[x_feature, y_feature]].dropna()
    #如果xfeature是数值
    if df_input[x_feature].dtype in ['int64','float64']:
        if group_cut is None:
            group_cut_list = np.linspace(0, 1, num=group_num + 1)
            group_cut = np.quantile(df_input[x_feature], group_cut_list)
        labels = list(range(1, len(group_cut)))#分组标签 （数字）
        labels_plt = []#分组标签 （图）
        labels_tale = []#分组标签 （表格）
        for i in range(1,len(labels)+1):
            # labels_plt.append(str(_round_dec(group_cut[i-1]))+'-' + str(_round_dec(group_cut[i])))
            labels_plt.append(str(_round_dec(group_cut[i])))
            labels_tale.append(str(_round_dec(group_cut[i - 1])) + '-' +str(_round_dec(group_cut[i])))
        labels_dict={}#分组标签字典
        for l in labels:
            labels_dict[l]=labels_tale[l-1]
        df_input[x_feature + '_分组'] = df_input.apply(lambda row: _group(row[x_feature], group_cut, 1), axis=1)
        df_input[x_feature + '_分组']=df_input[x_feature + '_分组'].astype(int)
    #如果x_feature不是数值
    else:
        labels_plt = []  # 分组标签 （图）
        labels_tale = []  # 分组标签 （表格）
        labels_dict = {}  # 分组标签字典
        labels_plt=df_input[x_feature].unique()# 分组标签 （非数字）（图）
        try:
            labels_plt.sort()
        except Exception:
            print(Exception)
        labels_tale=labels_plt#（表格）
        labels = list(range(1, len(labels_plt)+1))  # 分组标签 （数字）
        labels_dict = {}  # 分组标签字典(数字：标签)
        for l in labels:
            labels_dict[l] = labels_tale[l - 1]
        labels_dict_T={value: key for key, value in labels_dict.items()}
        df_input[x_feature + '_分组'] = df_input[x_feature].map(labels_dict_T)
    #初始化结果变量
    df_result1=pd.DataFrame()
    list_name_for_trend = []
    list_or_for_trend = []
    list_ci_for_trend = []
    list_p_for_trend=[]
    list_p_for_trend_error=[]
    str_result=''
    dict_df_result = {}
    plot_name_list=[]
    #创建画布
    plt.clf()
    plt.figure(figsize=(8,6),dpi=600)
    #--整体--
    df_result,ax=_groupdata_format(df_input=df_input,value_feature=y_feature,group_feature=x_feature + '_分组',label='Total',type=type,color=color_camp[0],decimal_num=decimal_num)
    #计算p for trend
    or_, ci_0, ci_1, p=_p_trend_for_smart_goup_analysis(df_input=df_input, dependent_variable=y_feature, independent_variable=x_feature + '_分组')
    list_name_for_trend.append('总体')
    list_or_for_trend.append('%s' %_round_dec(or_,decimal_num))
    list_ci_for_trend.append('[%s,%s]' %(_round_dec(ci_0, decimal_num),_round_dec(ci_1, decimal_num)))
    list_p_for_trend.append('%s' %_round_dec(p,decimal_num))
    #--分组--
    if group_ is not None:
        plt.clf()
        g_lables=df_input[group_].unique()
        try:
            g_lables.sort().sort()
        except Exception:
            print(Exception)
        color_index=0
        for g_lable in g_lables:
            try:
                df_result_temp,ax=_groupdata_format(df_input=df_input[df_input[group_]==g_lable],value_feature=y_feature,
                                                    group_feature=x_feature + '_分组',label=group_+'_'+str(g_lable),type=type,
                                                    color=color_camp[color_index],decimal_num=decimal_num)
                df_result = pd.merge(df_result, df_result_temp, on='分组', how='left')
                # 计算p for trend
                or_, ci_0, ci_1, p = _p_trend_for_smart_goup_analysis(df_input=df_input[df_input[group_]==g_lable], dependent_variable=y_feature,
                                                                      independent_variable=x_feature + '_分组')
                list_name_for_trend.append(group_+'_'+str(g_lable))
                list_or_for_trend.append('%s' % _round_dec(or_, decimal_num))
                list_ci_for_trend.append('[%s,%s]' % (_round_dec(ci_0, decimal_num), _round_dec(ci_1, decimal_num)))
                list_p_for_trend.append('%s' % _round_dec(p, decimal_num))
                color_index+=1
            except Exception:
                list_p_for_trend_error.append(str(g_lable))
    df_result['分组']=df_result['分组'].map(labels_dict)
    plt.legend(loc='lower right', fontsize=10)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    # plt.ylim(4, 9)
    plt.xticks(labels,labels_plt)
    plt.grid(axis='y',ls='--',alpha=0.5)
    ax_ = plt.gca()
    ax_.spines['top'].set_visible(False)  # 去掉上边框
    ax_.spines['right'].set_visible(False)  # 去掉右边框
    plt.title('')
    savepath_temp=path+'智能分组分析'+'.jpeg'
    a = '智能分组分析.jpeg'
    plot_name_list.append(a)
    plt.savefig(savepath_temp,bbox_inches = 'tight')
    plt.close()
    df_result1['分组']=list_name_for_trend
    df_result1['β']=list_or_for_trend
    df_result1['95%CI']=list_ci_for_trend
    df_result1['P for trend']=list_p_for_trend
    str_result+='研究%s(Y轴)在%s(X轴)不同区间的变化趋势,同时构建线性回归模型'\
               %(y_feature,x_feature)
    if group_ is not None:
        try:
            str_result +='同时比较不同%s下的%s是否存在差异，' %(group_,y_feature)
            #如果有分层变量，则比较不同层间差异是否显著
            df_result_2, str_result_temp=comprehensive_smart_analysis(df_input=df_input,group=group_,continuous_features=[y_feature],show_method=True)
            dict_df_result['组间差异分析']=df_result_2
            str_result+=str_result_temp
        except:
            str_result +='无法比较组间差异\n'
    if len(list_p_for_trend_error)>0:
        str_result+='(在不同的分层中，分层变量为'+','.join(list_p_for_trend_error)+'时，样本量过小，无法计算P for trend.)'
    dict_df_result['智能分组分析总览'] = df_result
    dict_df_result['各层趋势分析'] = df_result1

    return dict_df_result,str_result,plot_name_list
