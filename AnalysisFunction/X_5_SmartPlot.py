# V 1.0.15
# date 2021-6-29
# author：xuyuan

# V1.0.1 更新说明：更新forest_plot y轴从0开始
# V1.0.2 更新说明：增加了方法comparison_plot
# V1.0.3 更新说明：增加了方法point_line_plot
# V1.0.4 更新说明：更新了comparison_plot，point_line_plot
# V1.0.5更新说明：优化了comparison_plot，point_line_plot的画图文字显示问题，去掉了空白框
# V1.0.6更新说明：  1 comparison_plot，point_line_plot中优化了“hue”超过10个变量的显示问题
                # 2 新增堆叠柱状图、频率分布直方图
# V1.0.7 更新说明： 1.饼图突出了前几位，并更改了颜色分布
                # 2.线混合图修正标题位置
                # 3.分类散点图加入hue,dodge参数，并加入【simhei】显示中文
                # 4.比较图 x轴刻度标签转为水平，优化图例位置 修改上下列的重叠问题
                # 5.点线图 修改上下列的重叠问题
                # 6.频率分布直方图 加入了KDE密度曲线 增加了多行多列功能
                # 7.箱型图 修改当group！=None或hue！=None时的显示一个图形的情况
                # 8.散点图 加入透明度参数alpha
                # 9.小提琴图 增加了一些参数,添加了orient和scale
                # 10.堆叠柱状图 增加direction参数
# V1.0.8 更新说明： 优化了点线图中只有一个特征时的分层和添加相关系数
# V1.0.9 更新说明： pie_graph，comparison_plot，point_line_plot，stackbar_plot新增参数 palette_style
# V1.0.10 更新说明：优化了point_line_plot，
# V1.0.11 更新说明：优化了pie_graph
# v1.0.12 更新说明：优化了dist_plot
# v1.0.13 更新说明：优化了forest_plot 
# v1.0.14 更新说明：优化了一些方法
# v1.0.15 更新说明：散点图中相关性方法写错了，改成pearson
"""
    2021/02/09 Owen:
        forest_plot() 修改了垂直排列时的默认x轴范围为[0, 1]
        现在X5所有函数都plt.clf()，调用时会清除主程序中cached图，不是好的调用方式，之后应该进行一次整体调整。
"""

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import learning_curve
import math
import datetime

from AnalysisFunction.utils_ml import save_fig

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

palette_dict = {
    'lancet': ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF",
               "#1B1919FF"],
    'nejm': ["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF",
             "#BC3C29FF"],
    'jama': ["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF", "#80796BFF", "#374E55FF",
             "#DF8F44FF"],
    'npg': ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF",
            "#7E6148FF", "#B09C85FF"]}


CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
CB91_Grad_BP = ["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF",
             "#BC3C29FF"]

# [
#     '#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8',
#     '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
#     '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
#     '#568ae6', '#5986e4', '#5c81e2', '#607de0',
#     '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
#     '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
#     '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
#     '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
#     '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
#     '#a924b7', '#ac20b5', '#b01bb3', '#b317b1',
# ]



# -------饼图--------
def pie_graph(df_input, column_name, top_num, title, path, font_size=11, startangle=90, palette_style='nejm'):
    """
    绘制分类排序饼图
    df_input: Dataframe 处理数据
    column_name: str 绘图列名
    top_num：int 图表中展示的前几位
    title： str 标题
    font_size : int 字体大小 默认=11
    startangle：int 初始角度 默认=90
    返回值：
    plot_name_list
    """
    plt.clf()
    df_temp = df_input[[column_name]].dropna()
    total = pd.DataFrame({'total': df_temp.groupby(column_name).size()})
    total = total.sort_values(['total'], ascending=False)
    total.reset_index(inplace=True)
    total_head = total.head(top_num)
    total_num = total.shape[0]
    if (top_num > total_num):
        top_num = total_num
    if (top_num < total_num):
        other = {column_name: '其它', 'total': len(df_temp) - sum(total_head['total'])}
        total_head = total_head.append(other, ignore_index=True)
    total_head['percent'] = total_head['total'] / sum(total_head['total'])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.figure(figsize=(12, 6.5))
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['bottom'].set_color('none')
    # top_num_rest=total_num-top_num
    if (top_num < total_num):
        explode = top_num * [0.1] + [0]
    elif (top_num == total_num):
        explode = top_num * [0.05]
    # colors = ['turquoise', 'aquamarine', 'springgreen', 'yellow', 'chartreuse', 'lightcoral',
    #           'red', 'orangered', 'blueviolet', 'fuchsia', 'hotpink', 'lightskyblue', 'orangered',
    #           'lime', 'limegreen', 'aquamarine', 'lightseagreen']
    colors = palette_dict[palette_style]
    patches, l_text, p_text = plt.pie(total_head['total'], labels=total_head[column_name],
                                      colors=colors,
                                      radius=3.5,
                                      center=(4, 4),
                                      labeldistance=1.05,
                                      explode=explode,
                                      wedgeprops={'linewidth': 1},
                                      pctdistance=0.8, frame=1,
                                      textprops={'fontsize': 14, 'color': 'black'},
                                      counterclock=False, autopct="%0.2f%%", startangle=startangle)
    for t in p_text:
        t.set_size(font_size)
    for t in l_text:
        t.set_size(font_size)
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc='center right', fontsize=font_size)
    plt.title(title, fontsize=font_size + 2)
    plt.axis('equal')
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "pie_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def comparison_plot(df_input, features, path, group=None, isstack=None, name_y=None, hue_=None, kind='box',
                    concat_way=None, row_size=None, col_size=None, palette_style='nejm'):
    '''
    绘制比较图
    df_input: dataframe 输入数据
    features: list 选择的特征名列表
    path：str 路径
    group : str 分组变量,分组变量必须小于等于10
    isstack：bool 是否堆积，默认False
    name_y： str 仅concat_way=='samedia'时可用，图片中Y轴的显示名
    hue_: str 默认为None，仅concat_way=='free'时可用，第二个分组类型
    kind: str 默认为box，作图方式，可选kind='bar','box','violin'
    concat_way: str 默认为None,图片连接方式，可选concat_way='None','free','samedia'
    row_size,col_size： int 默认为None,concat_way=='free'时可用，子图的行数和列数
    返回值：
    plot_name_list
    '''
    plot_name_list = []
    plt.clf()
    sns.set_palette(palette= palette_dict[palette_style])
    def _choose_plot(plot_method, group, feature, hue, data, ax=None):
        if plot_method == 'box':
            sns.boxplot(x=group, y=feature, hue=hue, data=data, ax=ax)
        elif plot_method == 'bar':
            if isstack == 'True' and hue != None:
                sns.barplot(x=group, y=feature, hue=hue, data=data, ax=ax, dodge=False)
            else:
                sns.barplot(x=group, y=feature, hue=hue, data=data, ax=ax)
        elif plot_method == 'violin':
            sns.violinplot(x=group, y=feature, hue=hue, data=data, ax=ax)
        return ax

    if hue_ == None or len(df_input[hue_].unique()) <= 10:
        if len(features) == 1:
            rel = _choose_plot(kind, group, features[0], hue_, df_input)
            # leg = rel._legend
            # leg.set_bbox_to_anchor([1, 0.7])
            plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
            plt.xticks(size='small', rotation=90, fontsize=13)
        elif concat_way == None and len(features) != 1:
            print("多特征必须设置组合方式")
            return {'error': '多特征必须设置组合方式'}
        elif concat_way == 'free':
            if col_size == None and row_size == None:
                col_size = math.floor(math.sqrt(len(features)))
                row_size = math.ceil(len(features) / col_size)
            fig, axes = plt.subplots(row_size, col_size, figsize=(col_size * 5, row_size * 5))
            for i, ax in enumerate(axes.flat):
                if i < len(features):
                    _choose_plot(kind, group, features[i], hue_, df_input, ax=ax)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
                    ax.set_xticklabels(ax.get_xticklabels(), size='small', rotation=90, fontsize=13)
                    if hue_ != None:
                        ax.legend(loc='center left', bbox_to_anchor=(0.01, 1.2), ncol=5, title=hue_)
                else:
                    ax.set_visible(False)
            gs1 = gridspec.GridSpec(row_size, col_size)
            gs1.tight_layout(fig)
        elif concat_way == 'samedia':
            if col_size != None or row_size != None:
                print('同一个图中不能控制子图大小')
                return {'error': '同一个图中不能控制子图大小'}
            else:
                df_new = df_input[[features[0], group]]
                df_new['type'] = features[0]
                for i in range(1, len(features)):
                    df_i = df_input[[features[i], group]]
                    df_i['type'] = features[i]
                    df_i = df_i.rename(columns={features[i]: features[0]})
                    df_new = pd.concat([df_new, df_i])
                plt.figure(figsize=(7, 6))
                _choose_plot(kind, 'type', features[0], group, df_new)

                plt.legend(bbox_to_anchor=(1, 0.7))
                plt.xticks(size='small', rotation=90, fontsize=13)
                plt.xlabel('')
                plt.ylabel(name_y)

        time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
            datetime.datetime.now().second)
        random_number = random.randint(1, 100)
        plot_name = "comparison_plot_" + time_ + str(random_number) + ".jpeg"
        plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
        plt.close()
        plot_name_list.append(plot_name)
    else:
        return {'error': 'hue分组不能超过10'}
    return plot_name_list


def point_line_plot(df_input, features, group, path, hue_=None, kind='scatter', row_size=None, col_size=None, palette_style='nejm'):
    '''
    df_input : dataframe 输入数据
    features : list 选择的特征名列表
    group : str 分组变量 当kind='strip'时，group必须为分类变量,当kind='rel'时，group可为连续变量
    path : str 路径
    hue_ : str 默认为None，第二个分组类型
    kind : str 默认为scatter，作图方式，可选kind='strip','rel','scatter'
    row_size,col_size：int 默认为None,子图的行数和列数
    返回值
    plot_name_list
    '''
    plot_name_list = []
    plt.clf()
    sns.set_palette(palette=palette_dict[palette_style])
    def _choose_plot(df_input, plot_method, group, features, feature_num, hue, ax=None):
        if plot_method == 'strip':
            sns.stripplot(x=group, y=features[feature_num], jitter=True, data=df_input, ax=ax)
        elif plot_method == 'rel':
            if hue == None:
                sns.regplot(x=group, y=features[feature_num], scatter_kws={'s': 25, 'alpha': 0.5}, data=df_input)
            else:
                sns.lmplot(x=group, y=features[feature_num], hue=hue, scatter_kws={'s': 25, 'alpha': 0.5}, data=df_input)
                groups = list(df_input[hue].unique())
                groups.sort()
                text = []
                for k in groups:
                    df_input_k = df_input[df_input[hue] == k]
                    corr_k = df_input_k[group].corr(df_input_k[features[feature_num]])
                    text_k = 'Fitted line and 95%CI of {} group, r={:.4f}'.format(k, corr_k)
                    text.append(text_k)
                print(text)
                plt.legend(text, loc='lower center', bbox_to_anchor=(0.5, 1.03))
            corr = df_input[group].corr(df_input[features[feature_num]])
            textz = ' Pearson correlation of Total Data, r={:.4f}'.format(corr)
            plt.title(textz,verticalalignment='top')

        elif plot_method == 'scatter':
            groupss = len(df_input[hue].unique())
            sns.scatterplot(x=group, y=features[feature_num], hue=hue,
                            data=df_input, ax=ax,palette=palette_dict[palette_style][:groupss])


    def features_choose_plot(df_input, kind, group, features, hue, axes):
        for i, ax in enumerate(axes.flat):
            if i < len(features):
                if kind == 'strip' or kind == 'scatter':
                    _choose_plot(df_input, kind, group, features, i, hue, ax)
                elif kind == 'rel':
                    sns.regplot(x=group, y=features[i], scatter_kws={'s': 25, 'alpha': 0.5}, data=df_input, ax=ax)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
                if kind == 'scatter' and hue != None:
                    ax.legend(loc='center left', bbox_to_anchor=(-0.1, 1.13), ncol=5)
            else:
                ax.set_visible(False)
    if hue_ is not None:
        df_input=df_input.dropna(subset=[hue_])
    if hue_ == None or len(df_input[hue_].unique()) <= 10:
        if kind=="strip":
            if len(df_input[group].unique()) > 10:
                return {'error': '分类变量分类项过多，N不能超过10'}
        if len(features) == 1:
            _choose_plot(df_input, kind, group, features, 0, hue_)
            if kind == 'scatter' and hue_ != None:
                plt.legend(loc='center left', bbox_to_anchor=(0, 1.1), ncol=5)
        else:
            if col_size == None and row_size == None:
                col_size = math.floor(math.sqrt(len(features)))
                row_size = math.ceil(len(features) / col_size)
            fig, axes = plt.subplots(row_size, col_size, figsize=(col_size * 4, row_size * 4))
            features_choose_plot(df_input, kind, group, features, hue_, axes)
            gs1 = gridspec.GridSpec(row_size, col_size)
            gs1.tight_layout(fig)
        time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
            datetime.datetime.now().second)
        plot_name = "point_line_plot_" + time_ + ".jpeg"
        plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
        plt.close()
        plot_name_list.append(plot_name)
    else:
        return {'error': 'hue分组不能超过10'}
    return plot_name_list

#Echarts堆叠图
def pile_plot(df_input, x_, y_):
    #data_num 按数值获取分组列数据
    data_num = pd.crosstab(df_input[x_], df_input[y_])
    #data_rate 0 按百分比按行/当列总和获取分组列数据
    data_rate = data_num.div(data_num.sum(0))
    #获取y_name列表
    yListName = list(map(lambda x: str(y_) + "_" + str(x), list(data_num.columns)))
    #格式化y轴值名称
    xListName = list(map(lambda x: str(x_) + "_" + str(x), list(data_num.index)))
    #数值型数据
    data = []
    #百分比数据
    rate=[]
    for i in range(len(xListName)):
        data.append(list(data_num.iloc[i]))
        rate.append(list(map(lambda x: round(x *100,2), list(data_rate.iloc[i]))))
    return {"xListName":xListName,"yListName":yListName,"data":data,"rate":rate}

def stackbar_plot(df_input, x_, y_, path, direction='horizon', palette_style='nejm'):
    """
    绘制堆叠柱状图
    df_input: Dataframe 处理数据
    x_: str 离散型变量
    y_: str 离散型变量
    path ：str 路径
    direction: str 默认为horizon 水平，vertical 垂直
    返回值：
    plot_name_list
    """
    plt.clf()
    sns.set_palette(palette=palette_dict[palette_style])
    data_raw = pd.crosstab(df_input[x_], df_input[y_])
    data = data_raw.div(data_raw.sum(1), axis=0)  # 交叉表转换成比率，为得到标准化堆积柱状图
    list_x = []
    for x in data_raw.index:
        list_x.append(str(x))
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    if direction == 'horizon':
        local_var1 = pd.Series(0, index=data_raw.index)
        for i in range(0, data_raw.shape[1]):
            if i == 0:
                axs[0].barh(list_x, data_raw.iloc[:, i], height=0.4, label=y_ + str(data.columns[i]))
            else:
                axs[0].barh(list_x, data_raw.iloc[:, i], height=0.4, left=local_var1, label=y_ + str(data.columns[i]))
            local_var1 = data_raw.iloc[:, i] + local_var1

        axs[0].set_xlabel(y_ + '计数')
        axs[0].set_ylabel(x_)
        axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.95), ncol=1)
        axs[0].set_title('计数堆叠柱状图')
        local_var2 = pd.Series(0, index=data.index)
        for i in range(0, data.shape[1]):
            if i == 0:
                axs[1].barh(list_x, data.iloc[:, i], height=0.4, label=y_ + str(i))
            else:
                axs[1].barh(list_x, data.iloc[:, i], height=0.4, left=local_var2, label=y_ + str(i))
            local_var2 = data.iloc[:, i] + local_var2
        axs[1].set_xlabel(y_ + '百分比')
        axs[1].set_ylabel(x_)
        axs[1].set_xlim(0, 1)
        axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.95), ncol=1)
        axs[1].set_title('百分比堆叠柱状图')
    if direction == 'vertical':
        local_var1 = pd.Series(0, index=data_raw.index)
        for i in range(0, data_raw.shape[1]):
            if i == 0:
                axs[0].bar(list_x, data_raw.iloc[:, i], width=0.4, label=y_ + str(data.columns[i]))
            else:
                axs[0].bar(list_x, data_raw.iloc[:, i], width=0.4, bottom=local_var1, label=y_ + str(data.columns[i]))
            local_var1 = data_raw.iloc[:, i] + local_var1

        axs[0].set_xlabel(x_)
        axs[0].set_ylabel(y_ + '计数')
        axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.95), ncol=1)
        axs[0].set_title('计数堆叠柱状图')
        local_var2 = pd.Series(0, index=data.index)
        for i in range(0, data.shape[1]):
            if i == 0:
                axs[1].bar(list_x, data.iloc[:, i], width=0.4, label=y_ + str(i))
            else:
                axs[1].bar(list_x, data.iloc[:, i], width=0.4, bottom=local_var2, label=y_ + str(i))
            local_var2 = data.iloc[:, i] + local_var2
        axs[1].set_xlabel(x_)
        axs[1].set_ylabel(y_ + '百分比')
        axs[1].set_ylim(0, 1)
        axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.95), ncol=1)
        axs[1].set_title('百分比堆叠柱状图')

    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plt.tight_layout()
    plot_name = "stackbar_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def dist_plot(df_input, features, title, path, col_size=None, row_size=None):
    """
      绘制频率分布直方图
      df_input: Dataframe 处理数据Dataframe
      features : list 选择的特征名列表
      title: str 标题
      path： str 路径
      col_size，row_size：int 指定行和列
      返回值：
      plot_name_list
      """
    plt.clf()
    def features_choose_plot(df_input, features, axes):
        if len(features)>1:
            for i, ax in enumerate(axes.flat):
                if i < len(features):
                    sns.distplot(df_input[features[i]].dropna(), rug=True,
                                 rug_kws={'color': 'k', 'lw': 1.5},
                                 hist_kws={'color': 'r'}, ax=ax)
        else:
            sns.distplot(df_input[features].dropna(), rug=True,
                         rug_kws={'color': 'k', 'lw': 1.5},
                         hist_kws={'color': 'r'}, ax=axes)
    if (col_size == None and row_size == None) or(col_size*row_size<len(features)) :
        col_size = math.floor(math.sqrt(len(features)))
        row_size = math.ceil(len(features) / col_size)
    fig, axes = plt.subplots(row_size, col_size, figsize=(col_size * 4, row_size * 4))

    features_choose_plot(df_input, features, axes)
    plt.suptitle(title)
    gs1 = gridspec.GridSpec(row_size, col_size)
    gs1.tight_layout(fig)
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    plot_name = "dist_plot_" + time_ +str(random_number)+ ".jpeg"
    plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


# ----------------
# ----暂不展示---------------------
# ----------------
# -------水平柱状图----------------------------
def horizontal_bar_plot(df_input, name, value, title, path):
    """
    绘制水平柱状图
    df_input:处理数据Dataframe
    name：名称列 str
    value:数据列 str
    title：表名
    返回值：
    plt
    """
    plt.clf()
    plot_name_list = []
    n = len(df_input[name])

    # lengths = list(df_input[value])
    # name_lgs = [len(s) for s in df_input[name]]
    # x_range = abs(lengths[0] - lengths[-1]) + int(max(name_lgs) - min(name_lgs))/6.0 + 0.5)

    plt.figure(figsize=[6, max(n, 10)/2])
    ax = sns.barplot(x=value, y=name, data=df_input, order=list(df_input[name])[::-1], palette='Blues_d', saturation=1.5)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.title(title, loc='center', fontsize='10', fontweight='bold')
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "horizontal_bar_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()

    plot_name_list.append(plot_name)
    return plot_name_list


# ---------------森林图----------------
def forest_plot(df_input, name, value, err, direct, title, path, rotation_=0, ylim=[0.0, 1.0],
                fig_size=[4, 4], color_mark='yes', mean_mark='square', axvline_tick=1):
    """
    绘制水平森林图，用于不同模型评分
    df_input:处理数据Dataframe
    name：名称列 str
    value:数据列 str
    err：误差列 str 或list
    direct：str 方向（horizontal，vertical）
    title：str 表名
    rotation_:int X轴标旋转角度 默认45
    ylim：list 默认[0,1]
    fig_size：list [10,10]
    colormark：str 是否标记保护因素和危险因素 默认'yes'是‘no’否
    mean_mark:str 中间实体采用的形状标记 默认'square'方形  ‘circle’圆形
    返回值：
    plt
    """
    plt.clf()
    n = len(df_input[name])
    try:
        df_input[value] = df_input[value].astype(float)
        df_input[err] = df_input[err].astype(float)
    except:
        return '数据错误'
    plt.figure(figsize=fig_size)
    if (direct == 'horizontal'):
        if isinstance(err, list) & (len(err) == 2):
            err_l = df_input[value] - df_input[err[0]]
            err_h = df_input[err[1]] - df_input[value]
            for i in range(n):
                plt.errorbar(i + 1, df_input.loc[i, value], yerr=[[err_l[i]], [err_h[i]]], fmt='-o',
                             elinewidth=1, capsize=4, markersize=8, color=CB91_Grad_BP[i], ecolor=CB91_Grad_BP[i],
                             label=df_input.loc[i, name])
        elif isinstance(err, str):
            err_ = df_input[err]
            for i in range(n):
                plt.errorbar(i + 1, df_input.loc[i, value], yerr=err_[i], fmt='-o',
                             elinewidth=1, capsize=4, markersize=8, color=CB91_Grad_BP[i], ecolor=CB91_Grad_BP[i],
                             label=df_input.loc[i, name] + '=' + str(round(df_input.loc[i, value], 3)) + '(' + str(
                                 round(err_[i], 3)) + ')')
        else:
            return 'err参数错误'
        plt.xticks(range(1, n + 1), df_input[name], rotation=rotation_)
        plt.ylim(ylim)
        plt.legend(
            loc='lower right',
            numpoints=1,
            fancybox=True,
            fontsize='xx-small',
        )
        plt.title(title)
    if (direct == 'vertical'):
        ylim = list(reversed(range(1, n + 1)))
        if mean_mark == 'square':
            mark = 's'
        elif mean_mark == 'circle':
            mark = 'o'
        if color_mark == 'yes':
            values = df_input[value].astype(float)
            if isinstance(err, list) & (len(err) == 2):
                err_ = np.array([df_input[value] - df_input[err[0]], df_input[err[1]] - df_input[value]])
                for i in range(0, n):
                    if values[i] < axvline_tick:
                        ax = plt.errorbar(x=values[i], y=ylim[i], xerr=err_[:, i:i + 1], fmt=mark,
                                          color="blue", ecolor='black', elinewidth=1, capsize=4, markersize=8)
                    else:
                        ax = plt.errorbar(x=values[i], y=ylim[i], xerr=err_[:, i:i + 1], fmt=mark,
                                          color="brown", ecolor='black', elinewidth=1, capsize=4, markersize=8)
            elif isinstance(err, str):
                err_ = df_input[err]
                for i in range(0, n):
                    try:
                        if values[i] < 1:
                            ax = plt.errorbar(x=values[i], y=ylim[i], xerr=err_[i], fmt=mark,
                                              color="blue", ecolor='black', elinewidth=1, capsize=4, markersize=8)
                        else:
                            ax = plt.errorbar(x=values[i], y=ylim[i], xerr=err_[i], fmt=mark,
                                              color="brown", ecolor='black', elinewidth=1, capsize=4, markersize=8)
                    except:
                        continue
            else:
                return 'err参数错误'
        elif color_mark == 'no':
            if isinstance(err, list) & (len(err) == 2):
                err_ = [df_input[value] - df_input[err[0]], df_input[err[1]] - df_input[value]]
            elif isinstance(err, str):
                err_ = df_input[err]
            else:
                return 'err参数错误'
            ax = plt.errorbar(x=df_input[value], y=ylim, xerr=err_, fmt=mark,
                              color="limegreen", ecolor='black', elinewidth=2, capsize=4, markersize=15)
        plt.axvline(x=axvline_tick, ls="--", c="black")  # 添加垂直直线
        plt.yticks(ylim, df_input[name])  # ,fontsize=15
        plt.xticks(rotation=rotation_)
        plt.ylim(0, n + 1)
        # plt.xlim(0, 1)
        plt.title(title)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['left'].set_visible(False)  # 左边框
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    plot_name = "forest_plot_" + time_ + str(random_number) +".jpeg"
    plt.savefig(path + plot_name, dpi=600,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def box_plot(df_input, features, title, path, group=None, hue_=None):
    """
    绘制箱型图
    df_input: Dataframe 处理数据
    features： list 名称列
    title： str 标题
    path： str 路径
    group: str 第一个分组变量 默认为None
    hue: str 第二个分许变量 默认为None
    返回值：
    plot_name_list
    """
    plt.clf()
    if (isinstance(features, list) & (group is None) & (hue_ is None)):
        df_input[features].boxplot(
            patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans=True,  # 以点的形式显示均值
            boxprops={'color': 'black', 'facecolor': 'steelblue'},  # 设置箱体属性，如边框色和填充色
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6},
            # 设置均值点的属性，如点的形状、填充色和点的大小
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4},
            # 设置中位数线的属性，如线的类型和颜色
            medianprops={'linestyle': '--', 'color': 'black'}
        )
    else:
        col_size = math.floor(math.sqrt(len(features)))
        row_size = math.ceil(len(features) / col_size)
        fig, axes = plt.subplots(row_size, col_size, figsize=(col_size * 5, row_size * 5))
        for i, ax in enumerate(axes.flat):
            if i < len(features):
                sns.boxplot(x=group, y=features[i], hue=hue_, data=df_input, ax=ax)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
                ax.set_xticklabels(ax.get_xticklabels(), size='small', rotation=90, fontsize=13)
                if hue_ != None:
                    ax.legend(loc='center left', bbox_to_anchor=(-0.02, 1.13), ncol=5, title=hue_)
                else:
                    ax.set_visible(False)
            fig.tight_layout()

        # for i in range(len(features)):
        #     plt.figure()
        #     sns.boxplot(x=group,y=features[i],hue=hue_,data=df_input)
        #     plt.tight_layout()

    plt.title(title)
    plot_name_list = []
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    plot_name = "box_plot_" + time_ + str(random_number) + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()
    plot_name_list.append(plot_name)
    return plot_name_list


def scatter_plot(df_input, x_, y_, title, path, hue_=None, alpha='auto'):
    """
    绘制散点图
    df_input: Dataframe 处理数据
    x_: str 连续性变量
    y_: str 连续性变量
    title : str 标题
    path ：str 路径
    hue : str 分组变量 默认为 None
    alpha:int 设置透明度 默认 auto
    返回值：
    plot_name_list
    """
    plt.clf()
    sns.scatterplot(x=x_, y=y_, hue=hue_, data=df_input, alpha=alpha)
    plt.title(title)
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "scatter_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def violin_plot(df_input, x_, y_, title, path, hue_=None, orient=None, scale='area'):
    """
    绘制小提琴图
    df_input: Dataframe 处理数据
    x_: str 离散型变量
    y_: str 连续性变量
    title ：str 标题
    path： str 路径
    hue_: str 离散型变量
    orient: str 调整方向，默认为垂直，'h'为水平
    scale: str 默认area ， area-面积相同,count-按照样本数量决定宽度,width-宽度一样
    返回值：
    plot_name_list
    """
    plt.clf()
    if orient == None:
        sns.violinplot(x=x_, y=y_, hue=hue_, data=df_input,
                       split=True, linewidth=2,  # 线宽
                       width=0.8,  # 箱之间的间隔比例
                       palette='muted',  # 设置调色板
                       orient=orient, scale=scale,
                       gridsize=50  # 设置小提琴图的平滑度，越高越平滑
                       )
        plt.grid(axis='y', linestyle='--', linewidth=1, color='gray', alpha=0.4)

    if orient == 'h':
        sns.violinplot(x=y_, y=x_, hue=hue_, data=df_input,
                       split=True, linewidth=2,  # 线宽
                       width=0.8,  # 箱之间的间隔比例
                       palette='muted',  # 设置调色板
                       orient=orient, scale=scale,
                       gridsize=50  # 设置小提琴图的平滑度，越高越平滑
                       )

        plt.grid(axis='x', linestyle='--', linewidth=1, color='gray', alpha=0.4)
    plt.title(title)
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "violin_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def rel_plot(df_input, x_, y_, title, path, hue_=None, col_=None, kind_='scatter'):
    """
    绘制点，线混合图
    df_input: Dataframe 处理数据
    x_:str 连续性变量
    y_:str 连续性变量
    title: str 标题
    path：str 路径
    hue_:str 离散型变量 默认：None
    col_:str 离散型变量 默认：None
    kind_:str 绘制散点图 默认：'scatter'
    返回值：
    plot_name_list
    """
    plt.clf()
    rel = sns.relplot(x=x_, y=y_, hue=hue_, col=col_, data=df_input, kind=kind_)
    rel.fig.suptitle(title, x=0.53, y=0.95)
    plt.subplots_adjust(top=0.85)  # 修改标题位置
    leg = rel._legend
    leg.set_bbox_to_anchor([1, 0.7])  # change the values here to move the legend box
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "rel_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list


def strip_plot(df_input, path, x_, y_, hue_=None, dodge_=False):
    """
    绘制分类散点图
    df_input: Dataframe 处理数据
    path: str 路径
    x_: str 离散变量
    y_: str 连续性变量
    hue_: str 离散变量 进行内部分类
    dodge：bool 是否将hue的变量分开  默认False，打开为True
    返回值：
    plot_name_list
    """
    plt.clf()
    sns.set(style='whitegrid', color_codes=True)
    sns.stripplot(x=x_, y=y_, data=df_input, jitter=True, hue=hue_, dodge=dodge_)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    plot_name = "strip_plot_" + time_ + ".jpeg"
    plt.savefig(path + plot_name,bbox_inches = 'tight')
    plt.close()
    plot_name_list = []
    plot_name_list.append(plot_name)
    return plot_name_list




"""
        Plotting functions for ML model part
"""

def plot_learning_curve(
        estimator,
        X, y,
        cv=None,  # 交叉验证
        n_jobs=None,  # 设定所要使用的线程
        scoring=None,
        path=None,
    ):
    plt.clf()
    fig = plt.figure(figsize=(6, 4), dpi=600)

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
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.tick_params(bottom=False,labelbottom=True)

    ax.set_title(estimator.__class__.__name__ + " Learning Curve")
    ax.set_ylim([0.0, 1.005])
    ax.set_xlabel("Training Samples")
    ax.set_ylabel(scoring)
    ax.grid(which='major', axis='y', linestyle='-.')

    ax.plot(train_sizes, np.mean(train_scores,axis=1), 'd--', color=CB91_Grad_BP[0], label="Training Set")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'd--', color=CB91_Grad_BP[1], label="Validation Set")
    
    ax.legend(loc="lower right")
    plt.gca()
    plt.close()
    plot_name_list = []
    time_ = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
    datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    plot_name = "learning_curve_" + time_ + str(random_number)
    if path is not None:
        plot_name_list.append(save_fig(path, plot_name, '.jpeg', fig))
    return plot_name_list
