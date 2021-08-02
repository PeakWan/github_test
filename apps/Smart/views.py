import base64
import json
import os
import pickle
import re
import matplotlib.pyplot as plt
import uuid
import pandas as pd
import time
import threading
# from django.conf import settings
import eventlet

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Max
from django.shortcuts import render
from numpy.matlib import randn

from AnalysisFunction.X_1_DataGovernance import miss_data_delete, miss_data_filling, get_var_low_colinear
from AnalysisFunction.X_2_DataSmartStatistics import data_describe, levene_test, get_var_correlation, one_way_anova, \
    chi_square_test, fisher_test, median_difference, get_var_vif, multi_anova, comprehensive_smart_analysis, multi_comp, \
    nonparametric_test_continuous_feature, nonparametric_test_categorical_feature
from AnalysisFunction.X_3_DataSeniorStatistics import multi_models_regression, stratification_regression, multivariate_analysis, \
    smooth_curve_fitting_analysis, trend_regression, smart_goup_analysis
from AnalysisFunction.X_2_R_DataSmartStatistics import R_multi_comp
from apps.index_qt.models import File_old, Browsing_process, Member, MemberType

from django.http import HttpResponse, JsonResponse
from django import http
from django.views import View
from django_redis import get_redis_connection
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication
from libs.get import get_s, loop_add, write, read, filtering_view

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# class Advan(View):
#     SET = True


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.np.integer):
            return int(obj)
        elif isinstance(obj, pd.np.floating):
            return float(obj)
        elif isinstance(obj, pd.np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Zhfx(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        综合智能统计分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "choice"
        context["begin"] = "begin"
        context['name'] = 'zhfx'
        context['group'] = 'single'
        return render(request, 'index/zhfx.html', context=context)

    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 分组变量str
        group = json_data.get("group")
        # 分类变量名称list
        categorical_features = json_data.get("categorical_features")
        # 连续变量名称list
        continuous_features = json_data.get("continuous_features")
        # 分组标签list
        group_labels = json_data.get("group_labels")
        # bool False
        relate = json_data.get("relate")
        # 是否配对 Ture or False
        show_method = json_data.get("show_method")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        # print(relate)
        # print(type(relate))
        if not decimal_num:
            decimal_num = 1
        if relate == "False":
            relate = False
        elif relate == "True":
            relate = True
        if show_method == "False":
            show_method = False
        elif show_method == "True":
            show_method = True
        # print(relate)
        # print(type(relate))
        # print(type(False))
        num = json_data.get("number")

        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)
        # relate = False
        # group_labels = None

        if not continuous_features:
            continuous_features = None
        if not categorical_features:
            categorical_features = None
        if not group_labels:
            group_labels = None

        if not all([group]):
            return http.JsonResponse({'code': 1001, 'error': '请填写变量'})
        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]
            print('智能统计分析')
            print(filter_data)
            if filter_data:
                df = filtering_view(df,json_data)
                # print(df)
                # print(4444444444444444444)                

            # 调用综合智能统计分析方法
            try:
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = comprehensive_smart_analysis(df_input=df, group=group,
                                                      categorical_features=categorical_features,
                                                      continuous_features=continuous_features,
                                                      group_labels=group_labels, show_method=show_method, relate=relate,
                                                      decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                # mylock.release()
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})

            try:
                if re["error"]:
                    return HttpResponse(json.dumps({'code': '201', 'error': re["error"]}))
            except Exception as e:
                print(e)
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df  和str 数据
            df_result = re[0]
            str_s = re[1]
            # print(str_s, "-===================")
            str_s = str_s.replace("\n", "<br/>")
            a = df_result.fillna(' ')
            # print(df_result)
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    # break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        # break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/综合智能统计分析' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1003, 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据

            df_file = string2 + '/综合智能统计分析' + id + '/' + 'df_result.pkl'

            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '综合智能统计分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                # print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                'str': str_s,
                "text": "tableDescription",
                "form": form,
            }

        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 1006, 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Sjms(APIView):

    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        数据描述
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "much"
        context["value"] = "much"
        context["begin"] = "begin"
        context["name"] = "sjms"
        return render(request, 'index/sjms.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        # features=name
        name = json_data.get("name")

        num = json_data.get("number")

        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})
            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]
            
            print('数据描述')
            if filter_data:
                df = filtering_view(df,json_data)
            if not name:
                name = None
            # 调用数据描述方法
            try:

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):

                    re = data_describe(df, features=name, decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
                # print(re)
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析文件或填写的数据不正确,分析失败'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 分类数据描述
            df_result = re[0]
            # 计量数据描述
            df_result_2 = re[1]
            # print(1111111111111111)
            # print(re)
            # list02 = ['sum', 'sorting_item', 'Highest_frequency_term', 'frequency']
            result_o_desc = {}
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "分类数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # print(name, "name-=-=-------------------------11-1=-1--1==-=1-")
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for num in range(len(d[name[0]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            dict_x[index] = str(d.iloc[num][y])
                            y += 1
                        result_o_desc['info'].append(dict_x)
            # print(11111111111111111111111111111111)
            # 计量数据描述
            a = df_result_2
            before = {}
            before["name"] = "计量数据描述"

            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'unique', 'top', 'freq']
                # 循环列名
                for i in range(len(name)):
                    # print(len(name), name, "-=-=-=-=-=")
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # 服务器代码
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/数据描述' + id)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1005, 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            # 分类数据描述
            df_file = string2 + '/数据描述' + id + '/' + 'df_result.pkl'
            # 计量数据描述
            df_file_2 = string2 + '/数据描述' + id + '/' + 'df_result_2.pkl'
            print(df_file, "-=-=-=")
            # 分类数据描述
            write(df_result, df_file)
            # 计量数据描述
            write(df_result_2, df_file_2)

            browsing_process = {}
            browsing_process['name'] = '数据描述'
            browsing_process['str_result'] = "无分析描述"
            browsing_process['df_result_2'] = df_file
            browsing_process['df_result_2_2'] = df_file_2

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                # print(process)
                if process:
                    order = int(process.order) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            if result_o_desc:
                form = [result_o_desc, before]
            else:
                form = [before]
            print(result_o_desc)
            print('----------------------------------------------')
            print(before)
            context = {
                'project': project,
                "form": form,
                "text": "table",

            }

        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        return http.JsonResponse({'code': '200', 'error': '分析成功', 'context': context})


class Fxqx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        方差齐性
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "All"
        context["begin"] = "begin"
        context["name"] = "fc"
        context['group'] = 'single'
        return render(request, 'index/fxqx2.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        # print(json_data)
        # df_input, features, group,group_labels=None,method='mean'

        features = json_data.get("features")  # 获取多选列名  list
        group = json_data.get("group")  # 获取单选选列名
        group_labels = json_data.get("group_labels")  # 获取用户传的laber
        method = json_data.get("method")
        # 原始还是处理后的数据
        num = json_data.get("number")

        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        # 判断传过来的是否是空列表
        if not group_labels:
            group_labels = None

        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        if not all([features, group]):
            return http.JsonResponse({'code': 1001, 'error': '请填写变量'})
        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:

            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})
            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            print('方差齐性')
            if filter_data:
                df = filtering_view(df,json_data)
            # df_input, features, group,group_labels=None,method='mean'
            # 方差齐性  levene_test(df_input, group, continuous_features,group_labels=None, method='median',decimal_num=3
            print(features, group, group_labels, method, num, decimal_num, "=====================")

            try:
                re = levene_test(df, continuous_features=features, group=group, group_labels=group_labels,
                                 method=method, decimal_num=int(decimal_num))
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析文件或填写的数据不正确,分析失败'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df
            df_result = re[0]
            # 提取str
            str_result = re[1]
            str_result = str_result.replace("\n", "<br/>")
            # 把数据里面的NaN 转换为None
            df_result = df_result.fillna('')
            # 计量数据描述
            a = df_result
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'unique', 'top', 'freq']
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/方差齐性' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1005, 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/方差齐性' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)
            browsing_process = {'name': '方差齐性', 'df_result_2': df_file, "str_result": str_result}

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                "str": str_result,
                'project': project,
                "form": form,
                "text": "tableDescription",
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Xgx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        相关性分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        # context["value"] = "much"
        context["begin"] = "begin"
        context["name"] = "xgx"
        return render(request, 'index/xgxjy.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        #  json 获取参数
        json_data = json.loads(request.body.decode())
        cor_method = json_data.get("cor_method")
        plot_method = json_data.get("plot_method")
        features = json_data.get("arr")
        hue = json_data.get('hue')
        num = json_data.get("number")
        # 获取色带
        cmap_style = json_data.get('cmap_style')
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")

        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1003, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            print('相关性分析')
            if filter_data:
                df = filtering_view(df,json_data)
            # # 服务器上面由于路径原因要用这一段代码
            # # 把对象写入前端
            # plts.plot(randn(50).cumsum(), 'k--')
            # 写入数据库的路径
            end01 = "/static/"
            end = "/"

            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # strint3路径 为用户创的 项目下面
            print(string3)
            # string_tp 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "相关性分析" + id

            # print(type(f_list), type(cor_method),"=====", type(plot_method), "======", type(hue),"========")
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            if not features:
                # 如果用户没传数据默认所以列名
                features = list(df)
            try:
                # # 调用相关性分析方法
                savePath = file_s + "/"
                print(features, cor_method, "=====", plot_method, "======", savePath)

                features = list(map(str,
                                    features))  # df_input, features=None, cor_method='pearson', plot_method='heatmap', hue_=None,decimal_num=3

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = get_var_correlation(df, features=features, cor_method=cor_method, plot_method=plot_method,
                                             hue_=hue, savepath=savePath, decimal_num=int(decimal_num),cmap_style=cmap_style)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

                # print(2131241231412312)
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df
            df_result = re[0]
            plts = re[1]
            # print(plts, "2222222222222")
            # 返回df数据给前端

            a = df_result.fillna('')
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # print(before)
            # 把得到的图片保存
            # 返回前端 图片的静态路径
            # plt_file = string_tp + "/" + "相关性分析" + id + "/" + "plt.png"
            # plt_file02 = file_s + "/" + "plt.png"
            # plt.savefig(plt_file02, bbox_inches="tight")
            # plt.close()

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)

            # plts = re[2]
            pltlist = loop_add(plts, "相关性分析", string_tp, id)
            # print(pltlist, "1111111111111111")
            # 下面是共同的
            browsing_process = {'name': '相关性分析', "df_result_2": df_file, "str_result": " "}
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            # print(pltlist, "222222222222222222")
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                "img": pltlist,
                'project': project,
                "text": "tableImg",
                "form": form,
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Gxx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        共线性分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "much"
        context["begin"] = "begin"
        return render(request, 'index/gxx.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        #  json 获取参数
        json_data = json.loads(request.body.decode())

        features = json_data.get("arr")
        num = json_data.get("number")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")

        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")

        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1003, 'error': '会员查询失败，请重试'})
            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            # print('正太性验证')
            if filter_data:
                df = filtering_view(df,json_data)
            # # 调用共线性分析方法

            if not features:
                features = None

            try:
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = get_var_vif(df, features=features, decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df
            df_result = re
            # print(df_result, "============")
            # 返回df数据给前端
            a = df_result.fillna('')
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # print(before)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/共线性分析' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1005, 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/共线性分析' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)

            # 下面是共同的
            browsing_process = {'name': '共线性分析', "df_result_2": df_file, "str_result":"无分析描述"}
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                "text": "table",
                "form": form,
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))

#样本量分析
class Sample(APIView):
    authentication_classes = [MyBaseAuthentication, ]

    def get(self, request, project_id, data_id):
        context = {
            'begin': 'sample',
            'project_id': project_id
        }
        return render(request, 'index/sample.html', context=context)

    def post(self, request, project_id, data_id):
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功'}, cls=NpEncoder))

class Fcsjy(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        非参数检验
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "begin"
        context["name"] = "zwscy"
        return render(request, 'index/statistic_u.html', context=context)


    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 获取传过来的第一个列名
        y_column = json_data.get("y_column")
        # 获取传过来的连续数字
        y_labels = json_data.get("y_labels")
        # 判断用户是原始数据还是分析后的数据
        num = json_data.get("number")
        # 获取传过来的多个列名
        continuous_list = json_data.get("arr")
        # 获取传过来识别那个非参数检验函数的 识别
        commentVal = json_data.get("commentVal")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        if not y_labels:
            y_labels = None
        else:
            # y_labels 里面数字字符换为int形式
            try:
                y_labels = list(map(int, y_labels))
            except Exception as e:
                print(e)
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        if not all([y_column, continuous_list]):
            return http.JsonResponse({'code': '201', 'error': '请填写变量'})

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})
            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用非参数检验方法
            try:
                print(y_column, continuous_list, commentVal, y_labels)
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    # re = get_var_vif(df, features=features,decimal_num=int(decimal_num))
                    success = 1

                    # df_input,group,continuous_features,method,group_labels=None,show_method=False,decimal_num=3
                    # 临时定义一个show_method
                    show_method = False
                    # statistic_u,statistic_w, d_k_w, d_f  ['Mannwhitney-U','Wilcoxon','Kruskal-Wallis','Friedman']  cochrans_q
                    if commentVal in ['Mannwhitney-U', 'Wilcoxon', 'Kruskal-Wallis', 'Friedman']:
                        re = nonparametric_test_continuous_feature(df_input=df, group=y_column,
                                                                   continuous_features=continuous_list,
                                                                   method=commentVal, group_labels=y_labels,
                                                                   show_method=show_method,
                                                                   decimal_num=int(decimal_num))

                    elif commentVal in ['cochrans_q', 'McNemar']:
                        re = nonparametric_test_categorical_feature(df_input=df, group=y_column,
                                                                    categorical_features=continuous_list,
                                                                    method=commentVal, group_labels=y_labels,
                                                                    show_method=show_method,
                                                                    decimal_num=int(decimal_num))

                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
            # print(re)

            try:
                if re["error"]:
                    return HttpResponse(json.dumps({'code': '201', 'error': re["error"]}))
            except Exception as e:
                print(e)

            # 提取df  和str 数据
            df_result_2 = re[0]
            # 替换 使前面自动换行
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            a = df_result_2.fillna('')
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}  # 'name': a.iloc[n].name
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)
            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            # 判断调用的那个方法 文件夹名称
            str_dir = "非参数检验"
            if commentVal == "Mannwhitney-U":  # ['Mannwhitney-U','Wilcoxon','Kruskal-Wallis','Friedman'] ['cochrans_q','McNemar']
                str_dir = "U检验"
            elif commentVal == "Wilcoxon":
                str_dir == "Wilcoxon检验"
            elif commentVal == "Kruskal-Wallis":
                str_dir == "Kruskal-Wallis检验"
            elif commentVal == "Friedman":
                str_dir == "Friedman检验"
            elif commentVal == "cochrans_q":
                str_dir == "cochrans_q检验"
            elif commentVal == "McNemar":
                str_dir == "McNemar检验"

            # print("str_dir文件夹名称是", str_dir)

            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/' + str_dir + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/' + str_dir + id + '/' + 'df_result.pkl'
            write(df_result_2, df_file)
            if not str_s:
                str_s = "无分析描述"
            browsing_process = {}
            browsing_process['name'] = str_dir
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                'str': str_s,
                "text": "tableDescription",
                "form": form,
            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Kfjy(APIView):

    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        卡方检验
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "All"
        context["begin"] = "begin"
        context["name"] = "kfjy"
        context['group'] = 'single'
        return render(request, 'index/kfjy.html', context=context)

    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 获取传过来的第一个列名
        y_column = json_data.get("y_column")
        # 获取传过来的连续数字
        y_labels = json_data.get("y_labels")
        # 判断用户是原始数据还是分析后的数据
        num = json_data.get("number")
        # 获取传过来的多个列名
        continuous_list = json_data.get("arr")

        # 获取小数位
        decimal_num = json_data.get("num_select")

        show_method = json_data.get("relate")
        # 数据筛选
        filter_data = json_data.get("filter")

        if show_method == "False":
            show_method = False
        elif show_method == "True":
            show_method = True

        if not y_labels:
            y_labels = None
        else:
            try:
                y_labels = list(map(int, y_labels))
            except Exception as e:
                print(e)
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:

            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用卡方检验方法
            try:
                print(y_column, continuous_list, y_labels, "=========================")
                # (df_input, group, categorical_features,yates_correction=False,group_labels=None,show_method=False,decimal_num=3)

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = chi_square_test(df, group=y_column, categorical_features=continuous_list,
                                         group_labels=y_labels, show_method=show_method, decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df  和str 数据
            # print(re)
            df_result = re[0]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            a = df_result.fillna('')
            before = {}
            print("========11111========")
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)
            print("========22222========")
            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/卡方检验' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            print("========33333========")
            df_file = string2 + '/卡方检验' + id + '/' + 'df_result.pkl'
            print(df_result)
            write(df_result, df_file)
            print("========444444========")
            browsing_process = {}
            browsing_process['name'] = '卡方检验'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                'str': str_s,
                "text": "tableDescription",
                "form": form,
            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Zws(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        中位数差异分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "All"
        context["begin"] = "begin"
        context["name"] = "zwscy"
        context['group'] = 'single'
        return render(request, 'index/zws.html', context=context)

    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 获取传过来的第一个列名
        y_column = json_data.get("y_column")
        # 获取传过来的连续数字
        y_labels = json_data.get("y_labels")
        # 判断用户是原始数据还是分析后的数据
        num = json_data.get("number")
        # 重采样次数
        iter_n = json_data.get("iter_n")
        # 获取传过来的多个列名
        continuous_list = json_data.get("arr")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        if not y_labels:
            y_labels = None
        else:
            try:
                y_labels = list(map(int, y_labels))
            except Exception as e:
                print(e)
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        if not all([y_column, continuous_list]):
            return http.JsonResponse({'code': '201', 'error': '请填写变量'})

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa
        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用中位数差异分析方法
            try:
                print(y_column, continuous_list, y_labels, iter_n)
                # (df_input,group,continuous_features,group_labels=None,iter_n=100,decimal_num=3)

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = median_difference(df, group=y_column, continuous_features=continuous_list,
                                           group_labels=y_labels, iter_n=int(iter_n), decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
            try:
                if re["error"]:
                    return HttpResponse(json.dumps({'code': '201', 'error': re["error"]}))
            except Exception as e:
                print(e)
            # 提取df  和str 数据
            df_result = re[0]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            a = df_result.fillna('')
            # b = one_way_anova(df, "住院次数", ['性别', '年龄'], y_labels=[1, 2])
            before = {}
            # a = a[0]
            # before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # print(name, "-=-=-=")
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/中位数差异分析' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/中位数差异分析' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '中位数差异分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {

                'project': project,
                'str': str_s,
                'form': form,
                "text": "tableDescription",
            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class F_jy(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        Fisher检验
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "All"
        context["begin"] = "begin"
        context['group'] = 'single'
        return render(request, 'index/f_jy.html', context=context)

    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 获取传过来的第一个列名
        y_column = json_data.get("y_column")
        # 获取传过来的连续数字
        y_labels = json_data.get("y_labels")
        # 判断用户是原始数据还是分析后的数据
        num = json_data.get("number")
        # 获取传过来的多个列名
        continuous_list = json_data.get("arr")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        if not y_labels:
            y_labels = None
        else:
            try:
                y_labels = list(map(int, y_labels))
            except Exception as e:
                print(e)
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)
        # 获取传的列名
        if int(num) == 0:

            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用Fisher检验方法
            try:  # (df_input, group, categorical_features,group_labels=None,show_method=False, decimal_num=3)

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = fisher_test(df, group=y_column, categorical_features=continuous_list, group_labels=y_labels,
                                     decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})


            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df  和str 数据
            df_result = re[0]
            a = df_result.fillna('')
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    # if i == 8:
                    #     break
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for i in name:
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[i] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[i] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # print(before)
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/Fisher检验' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1003, 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/Fisher检验' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = 'Fisher检验'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                'str': str_s,
                "form": form,
                "text": "tableDescription",

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 3001, 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class D_fc(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        多组独立单因素方差分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "All"
        context["begin"] = "begin"
        context["name"] = "fc"
        context['group'] = 'single'
        return render(request, 'index/d_fc.html', context=context)

    def post(self, request, project_id, data_id):
        # 定义数据 因为前端未写好先自己定义数据
        json_data = json.loads(request.body.decode())
        # 获取传过来的第一个列名
        y_column = json_data.get("y_column")
        # 获取传过来的连续数字
        y_labels = json_data.get("y_labels")
        # 判断用户是原始数据还是分析后的数据
        num = json_data.get("number")
        # 获取传过来的多个列名
        continuous_list = json_data.get("arr")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")
        if not y_labels:
            y_labels = None
        else:
            try:
                y_labels = list(map(int, y_labels))
            except Exception as e:
                print(e)
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        if not all([y_column, continuous_list]):
            return http.JsonResponse({'code': '201', 'error': '请填写变量'})

        # 获取传的列名
        if int(num) == 0:

            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa
        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            #  多组独立单因素方差分析  (df_input,group,continuous_features,group_labels=None,show_method=False,decimal_num=3)
            try:
                print(y_column, continuous_list, y_labels)
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = one_way_anova(df, group=y_column, continuous_features=continuous_list, group_labels=y_labels,
                                       show_method=False, decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取df  和str 数据
            df_result = re[0]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 提取数据
            a = df_result.fillna('')
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])

                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)

                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/多组独立单因素方差分析' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/多组独立单因素方差分析' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)
            print(str_s, "-=-=-=str_s=-==")
            if not str_s:
                str_s = "无分析描述"
            print(str_s, "-=-=-=str_s=-==")
            browsing_process = {}
            browsing_process['name'] = '多组独立单因素方差分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                'str': str_s,
                "form": form,
                "text": "tableDescription",
            }
        except Exception as e:
            print(e)

            return HttpResponse(json.dumps({'code': '201', 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class D_fc2(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        多因素方差分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "much"
        context["begin"] = "begin"
        return render(request, 'index/d_fc2.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        #  json 获取参数
        json_data = json.loads(request.body.decode())
        # 统计方法{"F", "Chisq", "Cp"}
        # anova_test = json_data.get("anova_test")
        # 展示格式{1,2,3}
        # anova_typ = json_data.get("anova_typ")
        # 因变量
        group = json_data.get("group")
        # 1：常规 2： 计算相互作用
        # formula_type = json_data.get("formula_type")
        # 定义协变量 list
        features = json_data.get("arr")

        num = json_data.get("number")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # hue= json_data.get("hue")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")

        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            try:
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = multi_anova(df, group=group, features=features, decimal_num=int(decimal_num))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # print(re)
            # 提取df
            df_result = re[0]
            # str_s = re[1]
            # 返回df数据给前端
            a = df_result.fillna('')
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])

                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for i in name:
                        i = str(i)

                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[i] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[i] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            # print(before)

            # 服务器上面
            end = "/"
            # 本地代码上面
            # end = "/"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/多因素方差分析' + id)
                # print(string2, "123123123")
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            # 写入数据
            df_file = string2 + '/多因素方差分析' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)

            # 下面是共同的
            browsing_process = {'name': '多因素方差分析', "df_result_2": df_file, "str_result":"无分析描述"}
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            form = [before]
            context = {
                'project': project,
                "text": "tableDescription",
                "form": form,
                "str": str_s,
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Dcbj(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        多重比较
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "secondSingle"
        context["begin"] = "begin"
        return render(request, 'index/dcbj.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        #  json 获取参数
        json_data = json.loads(request.body.decode())
        # group,feature,method
        group = json_data.get("group")
        feature = json_data.get("feature")
        # method = json_data.get("method")
        num = json_data.get("number")
        # 数据筛选
        filter_data = json_data.get("filter")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")

        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1003, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]

            if filter_data:
                df = filtering_view(df,json_data)
            end01 = "/static/"
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # string2 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "多重比较" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savepath = file_s + '/'
            try:
                # multi_comp(df_input, group, continuous_feature, method='tukeyhsd',plot_type=1):
                # # 调用多重比较方法
                print(group, feature)
                print(type(group), type(feature))
                # group = "性别_二分组"

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = multi_comp(df_input=df, group=group, continuous_feature=feature, path=savepath)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            plts = re[2]
            str_result = re[0]
            # 判断结果是不是dataframe数据类型
            if isinstance(re[1], pd.DataFrame):
                # 提取df
                df_result = re[1]
                try:
                    for u in df_result.columns:
                        if df_result[u].dtype == bool:
                            df_result[u] = df_result[u].astype('int')
                except Exception as e:
                    print(e)
                # 返回df数据给前端

                a = df_result.fillna('')

                before = {}
                if list(a):
                    # 保存第一列的列名
                    before['Listing'] = []
                    name = list(a)
                    # 循环列名
                    for i in range(len(list(a))):
                        before['Listing'].append(name[i])
                        # if i == 8:
                        #     break
                    # 添加数据
                    before['info'] = []
                    for n in range(len(a[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            # 最多添加八个数据
                            # if y == 8:
                            #     break
                            try:
                                if dict_x[i]:
                                    t = str(uuid.uuid1())
                                    i = str(i) + t
                                    dict_x[index] = str(a.iloc[n][y])
                            except Exception as e:
                                dict_x[index] = str(a.iloc[n][y])
                            y += 1
                        before['info'].append(dict_x)

                # 把得到的图片保存
                # 返回前端 图片的静态路径
                pltlist = loop_add(plts, "多重比较", string_tp, id)
                # 写入数据
                df_file = file_s + '/' + 'df_result.pkl'
                write(df_result, df_file)

                # 下面是共同的
                browsing_process = {'name': '多重比较', "df_result_2": df_file, "str_result":str_result}
                for i in range(len(pltlist)):
                    if i == 0:
                        name = 'plt'
                        browsing_process[name] = pltlist[i]
                    else:
                        name = 'plt' + str(i + 1)
                        browsing_process[name] = pltlist[i]
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    print(process)
                    if process:
                        order = int(process.order) + 1
                        print(order)
                        browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                                   user_id=request.user.id,
                                                                   order=order,
                                                                   file_old_id=project_id)
                        project = browsing.id

                except Exception as e:
                    print(e)
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,
                                                               file_old_id=project_id)
                    project = browsing.id

                form = [before]
                context = {
                    'str': str_result,
                    "img": pltlist,
                    "project": project,
                    "text": "tableImglist",
                    "form": form,
                }

            else:
                pltlist = loop_add(plts, "多重比较", string_tp, id)
                # 下面是共同的
                browsing_process = {'name': '多重比较', "str_result":str_result}
                for i in range(len(pltlist)):
                    if i == 0:
                        name = 'plt'
                        browsing_process[name] = pltlist[i]
                    else:
                        name = 'plt' + str(i + 1)
                        browsing_process[name] = pltlist[i]
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    print(process)
                    if process:
                        order = int(process.order) + 1
                        print(order)
                        browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                                   user_id=request.user.id,
                                                                   order=order,
                                                                   file_old_id=project_id)
                        project = browsing.id
                except Exception as e:
                    print(e)
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,
                                                               file_old_id=project_id)
                    project = browsing.id
                
                context = {
                    'str': str_result,
                    "img": pltlist,
                    "project": project,
                    "text": "tableImglist",
                }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})

        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Znfx(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        百分位数智能分组分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)

        # 单选加多选
        # context["value"] = "All"
        context["value"] = "dobuleSingle"
        context["begin"] = "begin"
        return render(request, 'index/znfx.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        #  json 获取参数
        json_data = json.loads(request.body.decode())
        # print(json_data, "=-----------===========")

        x = json_data.get("x")
        y_clo = json_data.get("y")
        group = json_data.get("group")
        group_num = json_data.get("group_num")
        type_int = json_data.get("type_int")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        num = json_data.get("number")
        # 获取图片风格
        palette_style = json_data.get('palette_style')
        # 数据筛选
        filter_data = json_data.get("filter")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)
        if not y_clo:
            y_clo = None
        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa
        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")

        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]


            if filter_data:
                df = filtering_view(df,json_data)
            # # 百分位数智能分组分析
            print(x, "=====", y_clo, "=====", group, "======", group_num, "========")
            # 写入数据库的路径

            end01 = "/static/"
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # strint3路径 为用户创的 项目下面

            # string2 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "百分位数智能分组分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            try:  # (df_input, x_feature, y_feature, group_=None, group_num=5, group_cut=None, type=1,decimal_num=3)
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    # print(11111111111)
                    # print(group)
                    # print(df)
                    re = smart_goup_analysis(df_input=df, x_feature=x, y_feature=group, group_=y_clo,
                                             group_num=int(group_num), group_cut=None, type=int(type_int),
                                             decimal_num=int(decimal_num), path=savepath,palette_style=palette_style)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

                # re = smart_goup_analysis(df,'年龄','时间','性别_中文',type=2,group_num=8,decimal_num=1)

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1008, 'error': re['error']})
            except Exception as e:
                print(e)
                print('正常无error')
            # 提取df
            print(re[0])
            df_result = re[0]

            # print("===================", df_result)
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            plts = re[2]
            # 返回的字典把名字个dataframe数据分类
            # 表名称
            d_name = []
            # dataframe表
            d_dafr = []
            for x, y in df_result.items():
                d_name.append(x)
                d_dafr.append(y)

            # a = df_result.replace(6, 888)
            # 取第一个dataframe数据
            a = d_dafr[0].fillna('')

            before = {}
            before["name"] = d_name[0]
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                    before['info'] = []
                    for n in range(len(a[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            try:
                                if dict_x[i]:
                                    t = str(uuid.uuid1())
                                    i = str(i) + t
                                    dict_x[index] = str(a.iloc[n][y])
                            except Exception as e:
                                dict_x[index] = str(a.iloc[n][y])
                            y += 1
                        before['info'].append(dict_x)

            result_o_desc = {}
            result_o_desc["name"] = d_name[1]
            # 判断有没有表
            # 分类数据描述
            d = d_dafr[1].fillna('')
            # 计量数据描述
            if list(d):
                # 保存第一列的列名
                result_o_desc['Listing'] = []
                # 所有列名的列表
                name = list(d)
                # 循环列名
                for i in range(len(list(d))):
                    result_o_desc['Listing'].append(name[i])
                # 添加数据
                result_o_desc['info'] = []
                for n in range(len(d[name[1]])):
                    dict_x = {}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        i = str(i)
                        # 最多添加八个数据
                        # if y == 8:
                        #     break
                        try:
                            if dict_x[i]:
                                t = str(uuid.uuid1())
                                i = str(i) + t
                                dict_x[index] = str(d.iloc[n][y])
                        except Exception as e:
                            dict_x[index] = str(d.iloc[n][y])
                        y += 1
                    result_o_desc['info'].append(dict_x)
            print(33333)
            print(y_clo)
            if y_clo is not None:
                print(55555)
                third = {}
                third["name"] = d_name[2]
                # 判断有没有表
                # 分类数据描述
                t = d_dafr[2].fillna('')
                # 计量数据描述
                if list(t):
                    # 保存第一列的列名
                    third['Listing'] = []
                    # 所有列名的列表
                    name = list(t)
                    # 循环列名
                    for i in range(len(list(t))):
                        third['Listing'].append(name[i])
                    # 添加数据
                    third['info'] = []
                    for n in range(len(t[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            # 最多添加八个数据
                            # if y == 8:
                            #     break
                            try:
                                if dict_x[i]:
                                    o = str(uuid.uuid1())
                                    i = str(i) + o
                                    dict_x[index] = str(t.iloc[n][y])
                            except Exception as e:
                                dict_x[index] = str(t.iloc[n][y])
                            y += 1
                        third['info'].append(dict_x)
            print(44444)
            # # 服务器上面由于路径原因要用这一段代码
            # # 把对象写入前端
            # plts.plot(randn(50).cumsum(), 'k--')
            plts = re[2]

            pltlist = loop_add(plts, "百分位数智能分组分析", string_tp, id)
            # 把得到的图片保存
            # 返回前端 图片的静态路径

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(d_dafr[0], df_file)

            df_file_2 = file_s + '/' + 'df_result2.pkl'
            write(d_dafr[1], df_file_2)

            if y_clo is not None:
                df_file_3 = file_s + '/' + 'df_result3.xlsx'
                d_dafr[2].to_excel(df_file_3)
            # 下面是共同的
            if y_clo is None:
                browsing_process = {'name': '百分位数智能分组分析', "df_result_2": df_file, "df_result_2_2": df_file_2,
                                    "str_result": str_s}
            else:
                browsing_process = {'name': '百分位数智能分组分析', "df_result_2": df_file, "df_result_2_2": df_file_2,
                                    "df_result_2_2_2": df_file_3, "str_result": str_s}

            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            print(pltlist)

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            if y_clo is not None:
                form = [before, result_o_desc, third]
            else:
                form = [before, result_o_desc]
            context = {
                "img": pltlist,
                'project': project,
                "text": "tableDescriptionImg",
                "form": form,
                "str": str_s,
            }
        except Exception as e:
            print(e, )
            return http.JsonResponse({'code': 1006, 'error': '分析失败，请填写正确的变量'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class MultiView(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        多模型回归分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "much"
        context["value"] = "muchSingleCheckbox"
        context["begin"] = "begin"
        context["name"] = "muchCheckbox"
        return render(request, 'index/dmxhg.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        # 获取暴露变量
        exposure_variable = json_data.get('exposure')
        # 获取因变量
        dependent_variables = json_data.get('dependent')
        # # 获取分层变量
        stratification_variable = json_data.get('stratification_feature')
        # 获取回归模型名称
        model_name = json_data.get('model_name')
        # 获取调整变量1
        adjust_features_model_1 = json_data.get('adjust1')
        # 获取调整变量2
        adjust_features_model_2 = json_data.get('adjust2')

        time_variable = json_data.get('time_variable')
        # time_variable= None
        # 获取小数位
        decimal_num = json_data.get("num_select")
        num = json_data.get("number")
        style = json_data.get("style")
        # 数据筛选
        filter_data = json_data.get("filter")
        # stratification_feature = json_data.get('stratification_feature')
        # df_input, exposure_variable, dependent_variables, model_name='logit',adjust_features_model_1=None, adjust_features_model_2=None, stratification_variable=None, decimal_num=3
        if adjust_features_model_1 == []:
            adjust_features_model_1 = None
        if adjust_features_model_2 == []:
            adjust_features_model_2 = None
        if stratification_variable == "":
            stratification_variable = None
        if time_variable == []:
            time_variable = None
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
                # 服务器代码
            if filter_data:
                df = filtering_view(df,json_data)
            end = "/"
            end01 = "/static/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # string_tp 为图片的静态路径
            string_tp = string2[string2.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string2 + "/" + "多模型回归分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用多模型回归分析方法
            try:
                re = multi_models_regression(df, exposure_variable=exposure_variable,
                                             dependent_variables=dependent_variables, model_name=model_name,
                                             adjust_features_model_1=adjust_features_model_1,
                                             adjust_features_model_2=adjust_features_model_2,
                                             stratification_variable=stratification_variable,
                                             time_variable=time_variable, decimal_num=int(decimal_num),
                                             savePath=savepath,style=int(style))

                # re = multi_models_regression(df, '年龄', ['事件', '性别_中文'], model_name='logit', adjust_features_model_1=['时间'], stratification_variable='配对多分组')
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 分类数据描述
            df_result = re[0]
            # 保存字符串
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 获取图片
            plts = re[2]
            # print(plts, "--=-=-=11111=-=-=")
            result_o_desc = {}
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "分类数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for n in range(len(d[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            # 最多添加八个数据
                            # if y == 8:
                            #     break
                            try:
                                if dict_x[i]:
                                    t = str(uuid.uuid1())
                                    i = str(i) + t
                                    dict_x[index] = str(d.iloc[n][y])
                            except Exception as e:
                                dict_x[index] = str(d.iloc[n][y])
                            # print("============================",y,"=================================")
                            y += 1
                        result_o_desc['info'].append(dict_x)
            pltlist = loop_add(plts, "多模型回归分析", string_tp, id)
            # print(pltlist, "--=-=-=222222222=-=-=")
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)
            # print(33333333333333333333)
            browsing_process = {}
            browsing_process['name'] = '多模型回归分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                # print(process)
                if process:
                    order = int(process.order) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            form = [result_o_desc]
            # print(pltlist, "--=-=-=3333333=-=-=")
            context = {
                'project': project,
                "form": form,
                "img": pltlist,
                "text": "tableDescriptionImg",
                'str': str_s

            }

        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Dys(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        单因素\多因素分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "much"
        context["value"] = "single_factor"
        context["begin"] = "begin"
        context["name"] = "single"

        return render(request, 'index/dys.html', context=context)
 
    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        # 获取应变量
        dependent_variable = json_data.get('dependent_variable')
        # 获取自变量（定量）
        continue_exposure_variables = json_data.get('continue_exposure_variables')
        # # 获取自变量（定类）
        categorical_exposure_variables = json_data.get('categorical_exposure_variables')
        # 获取回归模型名称
        model_name = json_data.get('model_name')
        # 获取调整变量
        adjust_variables = json_data.get('adjust_variables')
        # 获取时间变量
        time_variable = json_data.get('time_variable')
        # time_variable= None
        # 获取小数位
        decimal_num = json_data.get("num_select")
        num = json_data.get("number")
        style = json_data.get("style")
        # 获取ref
        categorical_exposure_variable_ref = json_data.get('categorical_exposure_variables_ref')
        # 数据筛选
        filter_data = json_data.get("filter")
        if adjust_variables == []:
            adjust_variables = None
        # if adjust_features_model_2 == []:
        #     adjust_features_model_2 = None
        # 校验传值
        if time_variable == "":
            time_variable = None
        
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)
        print(categorical_exposure_variables)
        print(categorical_exposure_variable_ref)
        if len(categorical_exposure_variables) > len(categorical_exposure_variable_ref):
            none_num = len(categorical_exposure_variables) - len(categorical_exposure_variable_ref)
            print(none_num)
            for i in range(none_num):
                categorical_exposure_variable_ref.append(None)
        if len(categorical_exposure_variables) < len(categorical_exposure_variable_ref):
            return http.JsonResponse({'code': 1015, 'error': 'ref参数数量大于定类多选框'})

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
                # 服务器代码
            if filter_data:
                df = filtering_view(df,json_data)
            end = "/"
            end01 = "/static/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # string_tp 为图片的静态路径
            string_tp = string2[string2.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string2 + "/" + "单因素多因素分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用多模型回归分析方法
            try:

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = multivariate_analysis(df_input=df, continue_exposure_variables=continue_exposure_variables,
                                                 categorical_exposure_variables=categorical_exposure_variables, model_name=model_name,
                                                 dependent_variable=dependent_variable,
                                                #  adjust_features_model_2=adjust_features_model_2,
                                                 time_variable=time_variable, decimal_num=int(decimal_num),
                                                 savePath=savepath,adjust_variables=adjust_variables,categorical_exposure_variable_ref=categorical_exposure_variable_ref,style=int(style))
                    # print(222222222222222)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

                # re = multi_models_regression(df, '年龄', ['事件', '性别_中文'], model_name='logit', adjust_features_model_1=['时间'], stratification_variable='配对多分组')
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print(e)
                print('正常无error')
            # 分类数据描述
            df_result = re[0]
            # 保存字符串
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 获取图片
            plts = re[2]
            # print(plts, "--=-=-=11111=-=-=")
            result_o_desc = {}
            df_result = df_result.fillna('')
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "分类数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for n in range(len(d[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            # 最多添加八个数据
                            # if y == 8:
                            #     break
                            try:
                                if dict_x[i]:
                                    t = str(uuid.uuid1())
                                    i = str(i) + t
                                    dict_x[index] = str(d.iloc[n][y])
                            except Exception as e:
                                dict_x[index] = str(d.iloc[n][y])
                            # print("============================",y,"=================================")
                            y += 1
                        result_o_desc['info'].append(dict_x)
            pltlist = loop_add(plts, "单因素多因素分析", string_tp, id)
            # print(pltlist, "--=-=-=222222222=-=-=")
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)
            # print(33333333333333333333)
            browsing_process = {}
            browsing_process['name'] = '单因素多因素分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s

            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                # print(process)
                if process:
                    order = int(process.order) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            form = [result_o_desc]
            # print(pltlist, "--=-=-=3333333=-=-=")
            context = {
                'project': project,
                "form": form,
                "img": pltlist,
                "text": "tableDescriptionImg",
                'str': str_s
            }

        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Startification(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        分层分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "much"
        context["value"] = "Ret"
        context["begin"] = "begin"
        context["name"] = "fcfx"
        return render(request, 'index/fcfx.html', context=context)

    def post(self, request, project_id, data_id):

        # print(Advan.SET, "-0-0=0-90=0=-0")
        # if Advan.SET == True:
        #     Advan.SET = None
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        # 获取暴露变量
        exposure_feature = json_data.get('exposure')
        # 获取因变量
        dependent_features = json_data.get('dependent')
        # 获取分层变量
        stratification_features = json_data.get('adjust1')

        time_variable = json_data.get('time_variable')
        # # 获取小数位
        # formater = json_data.get('fomater')
        # 获取回归模型名称
        model_name = json_data.get('model_name')
        # 获取调整变量1
        adjust_features = json_data.get('stratification')
        # 获取小数位
        decimal_num = json_data.get("num_select")
        style = json_data.get("style")
        # 数据筛选
        filter_data = json_data.get("filter")
        # (df_input, exposure_variable, dependent_variable, stratification_variables=None, model_name='logit', adjust_variables=None, decimal_num=3)
        """
        df_input:Dataframe 
        exposure_feature: str 暴露变量(定量)
        dependent_feature： str 应变量
        stratification_features： list 分层变量(定类)
        model_name: str 回归模型名称
        {'logit':'逻辑回归','ols'：'线性回归','poisson':'泊松回归'}
        adjust_features：调整变量 list
        decimal_num：int
        """

        if adjust_features == []:
            adjust_features = None
        if time_variable == []:
            time_variable = None
        # if not all([exposure_feature,dependent_features,formater, model_name,stratification_features]):
        #     return HttpResponse(json.dumps({'code': 1001, 'error': '缺少参数'}))
        num = json_data.get("number")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

            if filter_data:
                df = filtering_view(df,json_data)
            # # 写入数据
            # 服务器代码
            end01 = "/static/"
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # strint3路径 为用户创的 项目下面

            # string2 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "分层分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savepath = file_s + '/'
            # 调用分层分析方法
            try:
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = stratification_regression(df, exposure_variable=exposure_feature,
                                                   dependent_variable=dependent_features,
                                                   stratification_variables=stratification_features,
                                                   model_name=model_name, adjust_variables=adjust_features,
                                                   time_variable=time_variable, decimal_num=int(decimal_num),
                                                   savePath=savepath,style=int(style))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

                # re = stratification_regression(df,'年龄','事件',['性别_中文'],adjust_variables=['时间'])
            except Exception as e:
                print(e)
                # Advan.SET = True
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            # 分类数据描述
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            df_result = re[0]
            df_result = df_result.fillna('')
            # 保存字符串
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            plts = re[2]

            result_o_desc = {}
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for num in range(len(d[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            dict_x[index] = d.iloc[num][y]
                            y += 1
                        result_o_desc['info'].append(dict_x)

            # 把得到的图片保存
            # 返回前端 图片的静态路径

            pltlist = loop_add(plts, "分层分析", string_tp, id)
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '分层分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id
            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            form = [result_o_desc]
            context = {
                'project': project,
                "form": form,
                'img': pltlist,
                "text": "tableDescriptionImg",
                'str': str_s
            }
        except Exception as e:
            print(e)
            # Advan.SET = True
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        # Advan.SET = True
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Smooth(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    # SET = True
    def get(self, request, project_id, data_id):
        """
        平滑曲线拟合分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "much"
        context["value"] = "different"
        context["begin"] = "begin"
        context["name"] = "phqx"
        return render(request, 'index/phqx.html', context=context)

    def post(self, request, project_id, data_id):
        # print(request.user.username, "=-=-=-=-21=3-1=2-3=1-2=3-")
        # if Smooth.SET == True:
        #     Smooth.SET = False
        """
        接收传的列名
        :param request:
        :return:
        """

        # print(self, "-=-=-235378168734683458")
        # print(request.user, "-=-=-235378168734683458-=======")
        json_data = json.loads(request.body.decode())
        # 获取暴露变量
        exposure_feature = json_data.get('exposure')
        # 获取因变量
        dependent_features = json_data.get('dependent')
        # 获取折点list变量
        inflection_list = json_data.get('inflection_list')
        # 获取小数位
        # formater = json_data.get('fomater')
        # 获取调整变量1
        adjust_features = json_data.get('adjust1')
        # 获取小数位
        decimal_num = json_data.get("num_select")
        style = json_data.get("style")
        # 数据筛选
        filter_data = json_data.get("filter")
        if not adjust_features:
            adjust_features = None

        list01 = []
        for i in inflection_list:
            if i:
                list01.append(int(i))
        inflection_list = list01

        if not inflection_list:
            inflection_list = None
        # if not all([exposure_feature,dependent_features,formater]):
        #     return HttpResponse(json.dumps({'code': 1001, 'error': '缺少参数'}))
        num = json_data.get("number")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存最新数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用平滑曲线拟合分析方法
            # 服务器代码
            end = "/"
            end01 = "/static/"
            # 本地代码上面
            # end = "\\"

            if int(num) == 0:
                string2 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            # print(string2)
            # string2 为图片的静态路径
            string_tp = string2[string2.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string2 + "/" + "平滑曲线拟合分析" + id
            try:
                # 创建一个唯一文件夹
                os.mkdir(string2 + '/平滑曲线拟合分析' + id)

            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': 1005, 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            try:
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = smooth_curve_fitting_analysis(df, exposure_variable=exposure_feature,
                                                       dependent_variable=dependent_features,
                                                       inflection_list=inflection_list, adjust_variable=adjust_features,
                                                       decimal_num=int(decimal_num), savePath=savepath,style=int(style))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            # 分类数据描述
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 保存字符串
            str_s = re[2]
            str_s = str_s.replace("\n", "<br/>")
            # 图片
            plts = re[0]

            df_result = re[1]
            df_result = df_result.fillna('')
            result_o_desc = {}
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for num in range(len(d[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            dict_x[index] = d.iloc[num][y]
                            y += 1
                        result_o_desc['info'].append(dict_x)

            # 写入数据
            pltlist = loop_add(plts, "平滑曲线拟合分析", string_tp, id)

            # 分类数据描述
            df_file = string2 + '/平滑曲线拟合分析' + id + '/' + 'df_result.pkl'
            # 分类数据描述
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '平滑曲线拟合分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            # 判别要保存的图片有几张

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                print(process)
                if process:
                    order = int(process.order) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            form = [result_o_desc]

            context = {
                'project': project,
                "form": form,
                "text": "tableDescriptionImg",
                'str': str_s,
                'img': pltlist
            }


        except Exception as e:
            # Smooth.SET = True
            print(e)
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        # Smooth.SET = True
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))
        # else:
        #     return http.JsonResponse({'code': 200, 'error': '上一次还没有运行结束'})


class Qsfx(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        趋势回归分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)

        context["value"] = "muchCheckboxs"
        context["begin"] = "begin"
        context["name"] = "muchCheckbox"
        return render(request, 'index/qsfx.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """
        json_data = json.loads(request.body.decode())
        # 获取暴露变量
        exposure_feature_list = json_data.get('exposure_feature_list')
        # 获取因变量
        dependent_feature = json_data.get('dependent_feature')
        # # 获取小数位
        # formater = json_data.get('formater')
        # categorical_independent_variables_ref = json_data.get('categorical_independent_variables_ref')
        # 获取回归模型名称
        categorical_independent_variables_ref = None
        model_name = json_data.get('model_name')
        # 获取调整变量1
        adjust_features_1 = json_data.get('adjust_features_1')
        # 获取调整变量2
        adjust_features_2 = json_data.get('adjust_features_2')

        exposure_feature_ref_label_list = json_data.get('exposure_feature_ref_label_list')
        # 获取小数位
        decimal_num = json_data.get("num_select")
        style = json_data.get("style")
        time_variable = json_data.get('time_variable')
        # 数据筛选
        filter_data = json_data.get("filter")
        # exposure_feature_ref_label_list=[2,'b']

        # if adjust_features_1 == []:
        # adjust_features_1 = None
        if time_variable == "":
            time_variable = None
        if not categorical_independent_variables_ref:
            # for x in range(len(exposure_feature_list)):
            #     categorical_independent_variables_ref.append(None)
            print(categorical_independent_variables_ref)
        else:
            ref = []
            for x in categorical_independent_variables_ref:
                try:
                    y = int(x)
                    ref.append(y)
                except Exception as e:
                    ref.append(x)
            categorical_independent_variables_ref = ref

        num = json_data.get("number")
        # 创建一个uuid唯一值
        id = uuid.uuid1()
        id = str(id)

        # 获取传的列名
        if int(num) == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif int(num) == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 打开文件
            # 查询该用户的用户等级
            try:
                grade = Member.objects.get(user_id=request.user.id)
                # 查询当前用户的等级
                member = MemberType.objects.get(id=grade.member_type_id)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'})

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

            if filter_data:
                df = filtering_view(df,json_data)
            # 调用趋势回归分析方法
            # 写入数据库的路径
            end01 = "/static/"
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # strint3路径 为用户创的 项目下面

            # string2 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "趋势回归分析" + id
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            try:
                print(exposure_feature_list, "categorical_exposure_variables", adjust_features_1,
                      "continuous_exposure_variables", time_variable, "time_variable", dependent_feature,
                      "dependent_variable", adjust_features_2, "adjust_variable", categorical_independent_variables_ref,
                      "categorical_independent_variables_ref", savepath, "savePath", decimal_num, "decimal_num")

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = trend_regression(df, categorical_exposure_variables=adjust_features_1,
                                          continuous_exposure_variables=adjust_features_2,
                                          dependent_variable=dependent_feature, model_name=model_name,
                                          adjust_variable=exposure_feature_list,
                                          categorical_independent_variables_ref=categorical_independent_variables_ref,
                                          time_variable=time_variable, decimal_num=int(decimal_num), savePath=savepath,style=int(style))
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请填写正确的变量'})
            # print("=-----------=-=-=-+_++++++++++++++++++")
            # print(re)
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 分类数据描述
            df_result = re[0]
            # 保存字符串
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            plts = re[2]
            result_o_desc = {}
            # 判断有没有表
            if list(df_result):
                result_o_desc["name"] = "分类数据描述"
                # 分类数据描述
                d = df_result
                # 计量数据描述
                if list(d):
                    # 保存第一列的列名
                    result_o_desc['Listing'] = []
                    # 所有列名的列表
                    name = list(d)
                    # 循环列名
                    for i in range(len(list(d))):
                        result_o_desc['Listing'].append(name[i])
                    # 添加数据
                    result_o_desc['info'] = []
                    for num in range(len(d[name[1]])):
                        dict_x = {}
                        y = 0
                        # 制定第一行 所有列的所有info数据
                        for index,i in enumerate(name):
                            i = str(i)
                            dict_x[index] = d.iloc[num][y]
                            y += 1
                        result_o_desc['info'].append(dict_x)

            pltlist = loop_add(plts, "趋势回归分析", string_tp, id)

            # 写入数据
            # 分类数据描述
            df_file = string3 + '/趋势回归分析' + id + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '趋势回归分析'
            browsing_process['df_result_2'] = df_file
            browsing_process['str_result'] = str_s
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            # print(pltlist)
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                    'order')
                # print(process)
                if process:
                    order = int(process.order) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=order,
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=1,
                                                           file_old_id=project_id)
                project = browsing.id
            # 把返回结果数据封装到列表里面
            form = [result_o_desc]
            context = {
                'project': project,
                "form": form,
                'str': str_s,
                'img': pltlist,
                "text": "tableDescriptionImg",
            }

        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': 1008, 'error': '请在试一次'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))



















