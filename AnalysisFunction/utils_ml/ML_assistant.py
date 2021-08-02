"""
    2021/01/13: Owen Yang 增加变量筛选 filtering()
"""

import datetime

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

from xgboost import plot_importance
from sklearn.model_selection import learning_curve

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import explained_variance_score as EVS
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .params import ToShow


"""
    Assistance functions
"""


def filtering(df, conditions):
    """
    2021/01/13: Owen Yang 变量筛选初版

    入参：
        pandas.dataframe::df:
            原始 dataframe

        tuple(str, dict)::conditions:
            str: 并且 / 或者 / 无
            dict: {
                str::col : (
                    str::rel, 
                    [[str::conditino1, float::value1], [str::condition2, float::value2]]
                ),
                ... ...
            }

    返回：
        筛选后的字符串
    """
    if (conditions is None):
        return df
    rel, col_dic = conditions
    def single_col(df, col, tp):
        def logic(series, string, value):
            if (string == "大于"):
                return (series > value)
            if (string == "小于"):
                return (series < value)
            if (string == "大于等于"):
                return (series >= value)
            if (string == "小于等于"):
                return (series <= value)
            if (string == "等于"):
                return (series == value)
            if (string == "不等于"):
                return (series != value)

        child_rel, ls = tp
        if (child_rel == "并且"):
            return logic(df[col], ls[0][0], ls[0][1]) & logic(df[col], ls[1][0], ls[1][1])
        elif (child_rel == "或者"):
            return logic(df[col], ls[0][0], ls[0][1]) | logic(df[col], ls[1][0], ls[1][1])
        else:
            return logic(df[col], ls[0][0], ls[0][1])

    if (rel == "并且"):
        indices = pd.Series([True, ] * len(df.index))
        for col, tp in col_dic.items():
            indices &= single_col(df, col, tp)
    elif (rel == "或者"):
        indices = pd.Series([False, ] * len(df.index))
        for col, tp in col_dic.items():
            indices |= single_col(df, col, tp)
    else:
        col, tp = list(col_dic.items())[0]
        indices = single_col(df, col, tp)
    return df[indices]


def save_fig(path, name, suffix, fig):
    now = datetime.datetime.now()
    time_ = str(now.hour) + str(now.minute) + str(now.second)
    plot_name = name + '_' + time_ + suffix
    fig.savefig(path + plot_name,bbox_inches = 'tight')
    return plot_name


def dic2str(dic, name=None):
    res_str = ""
    show = ToShow[name]
    for k, v in dic.items():
        if k in show:
            res_str += '\t' + str(k) + "（" + show[k] + "）" + ": " + str(v) + '\n'
    return res_str


def round_dec(n, d):
    """ 设置小数点位数 """
    if type(n) != type(1.0) : 
        return n
    s = '0.' + '0' * d
    return Decimal(str(n)).quantize(Decimal(s), rounding=ROUND_HALF_UP)
    


def plot_learning_curve(
        estimator,
        X, y,
        fig=None,  # 选择子图
        ylim=None,  # 设置纵坐标的取值范围
        cv=None,  # 交叉验证
        n_jobs=None,  # 设定所要使用的线程
        scoring=None,
    ):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, 
        X, y, 
        shuffle = True, 
        cv = cv,
        n_jobs = n_jobs,
        scoring = scoring,
    )

    if fig is None:
        fig = plt.figure(figsize=(6, 4), dpi=600)
    ax = plt.gca()
    
    ax.set_title(estimator.__class__.__name__ + " Learning Curve")
    if ylim is not None: 
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel(scoring)
    ax.grid(which='major', axis='y', linestyle='-.')

    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Set")
    ax.plot(train_sizes, np.mean(test_scores, axis=1),  'o-', color="g", label="Validation Set")
    
    ax.legend(loc="best")
    return ax


def make_class_metrics_dict(values=None):
    """
        Make a dictionary for classfication metrics.

        If values are given, then k-v pairs:{str-float/double}
        If values is None, then k-v pairs:{str-list}
    """
    if values is None:
        metric_dic = {
            'AUC': [],
            '准确度': [],
            '灵敏度': [],
            '特异度': [],
            '阳性预测值': [],
            '阴性预测值': [],
            'F1分数': [],
            # '约登指数': [],
            # '阈值': [],
        }
    else:
        assert len(values) == 7
        metric_dic = {
            'AUC': values[0],
            '准确度': values[1],
            '灵敏度': values[2],
            '特异度': values[3],
            '阳性预测值': values[4],
            '阴性预测值': values[5],
            'F1分数': values[6],
            # '约登指数': values[7],
            # '阈值': values[8],
        }
    return metric_dic


def make_regr_metrics_dict(values=None):
    """
        Make a dictionary for regression metrics.

        If values are given, then k-v pairs:{str-float/double}
        If values is None, then k-v pairs:{str-list}
    """
    if values is None:
        metric_dic = {
            'R方': [],
            '均方误差': [],
            '解释方差回归': [],
        }
    else:
        assert len(values) == 3
        metric_dic = {
            'R方': values[0],
            '均方误差': values[1],
            '解释方差回归': values[2],
        }
    return metric_dic


def regression_metric_evaluate(clf, Xtest, Ytest):
    """
        Input:
            clf: regression model/estimator
            Xtest: variable
            Ytest: target variable

        Return:
            metric_dic: metrics dictionary
            df_result: mean values of each metric
    """
    pred = clf.predict(Xtest)

    R2  = clf.score(Xtest, Ytest)
    mse = MSE(Ytest, pred)
    evs = EVS(Ytest, pred)
    metric_dic = make_regr_metrics_dict([R2, mse, evs])

    df_result = pd.DataFrame(metric_dic, index=['Mean'])
    return df_result, metric_dic


def classification_metric_evaluate(clf, Xtest, Ytest, binary=True):
    """
        Input:
            clf: classification model/estimator
            Xtest: variable
            Ytest: target variable

        Return:
            fpr: False Positive Rate
            tpr: True Positive Rate
            metric_dic: metrics dictionary
            df_result: mean values of each metric
    """
    if not binary:
        return multiclass_metric_evaluate(clf, Xtest, Ytest)

    prob = clf.predict_proba(Xtest)[:, 1]
    fpr, tpr, thresholds = roc_curve(Ytest, prob, pos_label=1)

    AUC = auc(fpr, tpr)
    youden = max(tpr - fpr) #约登指数

    maxindex = (tpr - fpr).tolist().index(youden)
    threshold = thresholds[maxindex] #阈值

    sen = tpr[maxindex] #灵敏度
    spe = 1 - fpr[maxindex] #特异度
    
    prob[prob > threshold] = 1
    prob[prob <= threshold] = 0

    cm = confusion_matrix(Ytest, prob, labels=[1, 0]) #混淆矩阵
    pre = cm[0, 0] / cm[:, 0].sum() #阳性预测值/精确度
    npv = cm[1, 1] / cm[:, 1].sum() #阴性预测值
    acc = (cm[1, 1] + cm[0, 0]) / cm.sum() #准确度

    f1_score = 2*pre* sen/(pre+sen)

    metric_dic = make_class_metrics_dict([
        AUC, acc, sen, spe, pre, npv, f1_score, 
        #youden, threshold,
    ])
    df_result = pd.DataFrame(metric_dic, ['Mean'])
    return fpr, tpr, metric_dic, df_result


def multiclass_metric_evaluate(clf, Xtest, Ytest):
    # precision, sensitivity, f1, _ = precision_recall_fscore_support(Ytest, clf.predict(Xtest), average='macro')

    cms = multilabel_confusion_matrix(Ytest, clf.predict(Xtest)) #混淆矩阵
    sen = np.asarray([cm[1,1]/(cm[1, :].sum() + 1e-3) for cm in cms]).mean()
    pre = np.asarray([cm[1,1]/(cm[:, 1].sum() + 1e-3) for cm in cms]).mean()
    spe = np.asarray([cm[0,0]/(cm[0, :].sum() + 1e-3) for cm in cms]).mean()
    npv = np.asarray([cm[0,0]/(cm[:, 0].sum() + 1e-3) for cm in cms]).mean()

    f1c = 2 * pre*sen / (pre+sen)
    acc = np.asarray([(cm[1, 1]+cm[0, 0])/(cm.sum() + 1e-3) for cm in cms]).mean()

    AUC = roc_auc_score(Ytest, clf.predict_proba(Xtest), average='macro', multi_class='ovo')

    metric_dic = make_class_metrics_dict([
        AUC, acc, sen, spe, pre, npv, f1c, 
    ])
    df_result = pd.DataFrame(metric_dic, ['Mean'])
    return None, None, metric_dic, df_result
    




def ci(data):
    """
        calculate confidence interval

        Input:
            data: given AUC values

        Return:
            a, b : left-boundary, right boundary
    """
    sample_mean = np.mean(data)
    se = stats.sem(data)
    a = sample_mean - 1.96 * se
    b = sample_mean + 1.96 * se
    return a, b
