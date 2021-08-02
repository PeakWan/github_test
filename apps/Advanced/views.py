import base64
import json
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import uuid
import pandas as pd
import eventlet
import time
import threading

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Max
from django.shortcuts import render
from numpy.matlib import randn
from AnalysisFunction.X_3_R_DataSeniorStatistic import R_logistic_regression, R_liner_regression, R_cox_regression, R_NRI_ana
from AnalysisFunction.X_5_R_SmartPlot import R_froest_plot

from AnalysisFunction.X_3_DataSeniorStatistics import two_groups_roc, multi_models_regression, cox_model
from AnalysisFunction.X_4_MachineLearningMethod import ML_Classfication, features_importance, \
    model_score_with_features_add, ML_Regression, ML_Clustering, two_groups_classfication_multimodels

from apps.index_qt.models import File_old, Browsing_process, Member, MemberType

from django.http import HttpResponse, JsonResponse
from django import http
from django.views import View
from django_redis import get_redis_connection

from libs.get import get_s, loop_add, write, read, filtering_view,get_table

from AnalysisFunction.X_1_DataGovernance import _feature_get_n_b, data_standardization
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication
# ---------logistics回归分析------------------
# from AnalysisFunction.X_3_DataSeniorStatistics import _model_selection

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

    
    
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


class Loghg(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        logistics回归分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "loghg"
        context["begin"] = "end"
        context["name"] = "logistic"
        return render(request, 'index/loghg.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """
        # 接收json数据
        json_data = json.loads(request.body.decode())
        dependent_variable = json_data.get("group")
        continuous_independent_variables = json_data.get("features")
        categorical_independent_variables = json_data.get("categorical_independent_variables")
        categorical_independent_variables_ref = json_data.get("log_ref")
        interaction_effects_variables = json_data.get("interaction_effects_variables")
        model_name = json_data.get("optione_wireBack")
        step_method = json_data.get("option_radio")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 获取是否保存结果
        log_save = json_data.get('save')
        
        style = json_data.get('style')
        # 原始还是处理后的数据
        num = json_data.get("number")
        # 数据筛选
        filter_data = json_data.get("filter")
        id = uuid.uuid1()
        id = str(id)

        if len(categorical_independent_variables) > len(categorical_independent_variables_ref):
            none_num = len(categorical_independent_variables) - len(categorical_independent_variables_ref)
            print(none_num)
            for i in range(none_num):
                categorical_independent_variables_ref.append(None)
        if len(categorical_independent_variables) < len(categorical_independent_variables_ref):
            return http.JsonResponse({'code': 1015, 'error': 'ref参数数量大于定类多选框'})

        # 获取传的列名
        if num == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif num == 1:
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
            # # 写入数据

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
            file_s = string3 + "/" + "logistics回归分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savepath = file_s + '/'
            try:
                re = R_logistic_regression(df_input=df, dependent_variable=dependent_variable,
                                           continuous_independent_variables=continuous_independent_variables,
                                           categorical_independent_variables=categorical_independent_variables,
                                           categorical_independent_variables_ref=categorical_independent_variables_ref,
                                           interaction_effects_variables=interaction_effects_variables,
                                           savePath=savepath, model_name=model_name,
                                           step_method=step_method, decimal_num=int(decimal_num),style=int(style))

            except Exception as e:
                print(str(e))
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请选择正确变量'})
            # print(re)
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print(e)
                print('正常')
            # 提取df
            save_data = re[3]
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 计量数据描述
            a = df_result
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                name[0] = '&emsp;'
                # 循环列名s
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {'&emsp;': a.iloc[n].name}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for index,i in enumerate(name):
                        dict_x[index] = str(a.iloc[n][y])
                        y += 1
                    # print(dict_x)
                    before['info'].append(dict_x)
            # 把得到的图片保存
            # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "logistics回归分析", string_tp, id)

            df_file = file_s + '/' + 'dfresult222.pkl'
            write(df_result, df_file)

            save_files = file_s + '/' + 'df_result.pkl'
            write(save_data, save_files)

            browsing_process = {'name': 'logistic回归', 'df_result_2': df_file, 'str_result': str_s,
                                'df_result': save_files}
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            if int(log_save) == 1:
                # 找到之前的处理后的数据替换掉
                try:
                    # 找到这个项目的最新处理的数据
                    latest_data = Browsing_process.objects.get(file_old=project_id, is_latest=1)
                    if latest_data:
                        # 修改数据结果
                        latest_data.is_latest = 0
                        latest_data.save()
                except Exception as e:
                    print(e)
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
                        browsing.is_latest = 1
                        browsing.save()


                except Exception as e:
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,

                                                               file_old_id=project_id)
                    browsing.is_latest = 1
                    browsing.save()
                    project = browsing.id

            else:
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
            plt_list = pltlist
            context = {
                'img': plt_list,
                'project': project,
                "form": form,
                "text": "tableDescriptionIImages1",
                'str': str_s
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '网络延迟，请稍后再试！！！'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Xxhg(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        线性回归
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "loghg"
        context["begin"] = "end"
        context["name"] = "xxhg"
        return render(request, 'index/xxhg.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        # 接收json数据
        json_data = json.loads(request.body.decode())
        dependent_variable = json_data.get("group")
        continuous_independent_variables = json_data.get("features")
        categorical_independent_variables = json_data.get("categorical_independent_variables")
        categorical_independent_variables_ref = json_data.get("xxhg_ref")
        # categorical_independent_variables_ref = ["", ""]
        interaction_effects_variables = json_data.get("interaction_effects_variables")
        model_name = json_data.get("optione_wireBack")
        step_method = json_data.get("optione_wireBack")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        num = json_data.get("number")
        style = json_data.get('style')
        # 数据筛选
        filter_data = json_data.get("filter")
        if not categorical_independent_variables_ref:
            for x in range(len(categorical_independent_variables)):
                categorical_independent_variables_ref.append(None)
        else:
            ref = []
            for x in categorical_independent_variables_ref:
                try:
                    y = int(x)
                    ref.append(y)
                except Exception as e:
                    ref.append(x)
            categorical_independent_variables_ref = ref

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
            file_s = string3 + "/" + "线性回归" + id
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用线性回归方法

            try:
                print(dependent_variable, continuous_independent_variables, categorical_independent_variables,
                      categorical_independent_variables_ref, interaction_effects_variables, model_name, step_method)
                re = R_liner_regression(df_input=df, dependent_variable=dependent_variable,
                                        continuous_independent_variables=continuous_independent_variables,
                                        categorical_independent_variables=categorical_independent_variables,
                                        categorical_independent_variables_ref=categorical_independent_variables_ref,
                                        interaction_effects_variables=interaction_effects_variables,
                                        savePath=savepath, model_name=model_name, step_method=step_method,
                                        decimal_num=int(decimal_num),style=int(style))


            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
                
            # 提取plt  和str 数据
            # print(re)
            df_result = re[0]
            plts = re[2]
            print(re[1])
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 获取返还给页面的df数据
            a = df_result
            before = {}
            before["name"] = "结果数据描述"
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
                        # 最多添加八个数据
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
            pltlist = loop_add(plts, "线性回归", string_tp, id)

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '线性回归'
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
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                # print(process)
                if process:
                    order = int(process['order__max']) + 1
                    # print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before]
            context = {
                'project': project,
                'img': pltlist,
                "form": form,
                "text": "tableDescriptionIImages1",
                "str": str_s

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Roc(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        二组独立样本ROC分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "roc"
        return render(request, 'index/roc.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """
        # 接收json数据
        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        title = json_data.get("title")
        num = json_data.get("number")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 获取图片风格
        palette_style = json_data.get('palette_style')
        # 数据筛选
        filter_data = json_data.get("filter")
        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存过最新数据，所有没有最新数据噢'}))
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
            file_s = string3 + "/" + "二组独立样本ROC分析" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用二组独立样本ROC分析方法
            try:
                if decimal_num:
                    decimal_num = int(decimal_num)
                else:
                    decimal_num = 3
                re = two_groups_roc(df, features=features, group=group, title=title, savePath=savepath,
                                    decimal_num=decimal_num,palette_style=palette_style)
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            a = df_result
            before = {}
            before["name"] = "结果数据描述"
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
                # 循环列名
                for i in range(len(list(a))):
                    before['Listing'].append(str(name[i]))
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

            # print("=============---------======222")
            # 把对象写入前端
            # plts.plot(randn(50).cumsum(), 'k--')

            # 把得到的图片保存
            pltlist = loop_add(plts, "二组独立样本ROC分析", string_tp, id)
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)

            browsing_process = {}
            browsing_process['name'] = '二组独立样本ROC分析'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before]
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Coxfx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        COX回归分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "Ret"
        context["begin"] = "end"
        context["name"] = "coxfx"
        return render(request, 'index/coxfx.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())

        sta_variable = json_data.get("sta_variable")
        tim_variable = json_data.get("tim_variable")
        continuous_independent_variables = json_data.get("continuous_independent_variables")

        categorical_independent_variables = json_data.get("categorical_independent_variables")
        decimal_num = json_data.get("decimal_num")
        step_method = json_data.get("step_method")
        interaction_effects_variables = json_data.get("interaction_effects_variables")

        categorical_independent_variables_ref = json_data.get("categorical_independent_variables_ref")

        timequant = json_data.get("timequant")
        timeroc = json_data.get('timeroc')
        style = json_data.get('style')
        # 是否展示校准曲线
        calibrate = json_data.get('calibrate')
        u = json_data.get('u')
        # 数据筛选
        filter_data = json_data.get("filter")
        # 获取时间变量
        timepreinc = json_data.get("timepreinc")
        # 获取时间列表
        timeprelist = json_data.get('timeprelist')

        # 修改参数并且解析
        if len(categorical_independent_variables) < len(categorical_independent_variables_ref):
            return http.JsonResponse({'code': 1015, 'error': 'ref参数数量大于定类多选框'})
        list01 = []
        try:
            for i in timequant[0].split(','):
                if i:
                    list01.append(float(i))
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1016, 'error': '时间分位点输入不正确，请输入大于0小于1的小数并用逗号隔开'})
        timequant = list01
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
            df_r = read(file_path)
            df = df_r.iloc[0:int(member.number)]
            
            # 数据筛选
            if filter_data:
                df = filtering_view(df,json_data)
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
            file_s = string3 + "/" + "COX回归分析" + id
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savePath = file_s + "/"

            # 调用 COX回归分析方法
            try:
                # print(time_column, event_column, groups, features, num, "================") # (df_input, time_column, event_column,groups=None, features=None)
                ref = []
                if not categorical_independent_variables_ref:
                    for x in range(len(categorical_independent_variables)):
                        ref.append(None)
                else:
                    for x, y in categorical_independent_variables_ref.items():
                        # 查找数据类型
                        print(df[x].dtypes, '--------------------------------------------')
                        try:
                            if df[x].dtypes == 'float64':
                                a = float(y)
                                ref.append(a)
                            elif df[x].dtypes == 'object':
                                a = str(y)
                                ref.append(a)
                            else:
                                a = int(y)
                                ref.append(a)
                        except Exception as e:
                            ref = []
                            print('转换错误')
                            print(e)
                if len(categorical_independent_variables) > len(ref):
                    none_num = len(categorical_independent_variables) - len(ref)
                    for i in range(none_num - 1):
                        ref.append(None)
                if u:
                    u = float(u)
                else:
                    u = None
                if timeprelist == 'None':
                    timeprelist = None
                re = R_cox_regression(df_input=df, sta_variable=sta_variable, tim_variable=tim_variable,
                                      continuous_independent_variables=continuous_independent_variables,
                                      categorical_independent_variables=categorical_independent_variables,
                                      categorical_independent_variables_ref=ref,
                                      interaction_effects_variables=interaction_effects_variables,
                                      savePath=savePath,
                                      step_method=step_method, decimal_num=int(decimal_num), timequant=timequant,timeroc=timeroc, style=int(style),calibrate=calibrate,u=u,timepreinc=timepreinc,timeprelist=timeprelist)

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result = re[0]

            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            a = df_result
            before = {}
            before["name"] = "结果数据描述1"
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
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
            # # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "COX回归分析", string_tp, id)

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result, df_file)
            browsing_process = {}
            browsing_process['name'] = 'COX回归分析'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before]
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"
            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Zydpx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        重要度排序
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "zydpx"
        return render(request, 'index/zydpx.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")

        top_features = int(json_data.get("top_features"))
        model_type = int(json_data.get("model_type"))
        standardization = json_data.get('standardization')
        searching = json_data.get('searching')
        num = json_data.get("number")
        # 数据筛选
        filter_data = json_data.get("filter")


        if standardization == 'false':
            standardization = False
        else:
            standardization = True
        if searching == 'false':
            searching = False
        else:
            searching = True
            
        if not features:
            features = None
        # 使用要分析的列名

        # =========================
        # 方法更新了 前端没写好 暂时用一个临时定义的列名代替
        # electedText = ["年龄", "门诊次数", "出院状态", "观测时间内住院次数"]

        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存过最新数据，所有没有最新数据噢'}))
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
            file_s = string3 + "/" + "重要度排序" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用重要度排序方法
            try:
                print(group, features, model_type, top_features, num, "================")
                # df_input,group,features,top_features = 10,model_type = 1,):
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = features_importance(df, group=group, features=features, top_features=top_features,
                                             model_type=model_type, savePath=savepath,standardization=standardization, searching=searching)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 获取返还给页面的df数据
            key_list = list(df_result.keys())
            a = df_result[key_list[0]]
            before = {}
            before["name"] = key_list[0]
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

            # 写入数据
            pltlist = loop_add(plts, "重要度排序", string_tp, id)
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            # df_result[key_list[0]].to_excel(df_file)
            write(df_result[key_list[0]], df_file)

            browsing_process = {}
            browsing_process['name'] = '重要度排序'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before]
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Blmx(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        变量模型评分分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "blmx"
        return render(request, 'index/blmx.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        # v_round
        v_round = int(json_data.get("v_round"))
        scoring = json_data.get("scoring")
        model_type = int(json_data.get("model_type"))
        importance_first = json_data.get('importance_first')
        searching = json_data.get('searching')
        num = json_data.get("number")
        # 数据筛选
        filter_data = json_data.get("filter")

        if importance_first == 'true':
            importance_first = True
        else:
            importance_first = False
        
        if not features:
            features = None

        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存最新的数据，所有没有最新数据噢'}))
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
            file_s = string3 + "/" + "变量模型评分分析" + id
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用变量模型评分分析方法
            try:
                print(group, features, v_round, scoring, model_type, num, "================")

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = model_score_with_features_add(df, group=group, features=features, v_round=v_round,
                                                       scoring=scoring, model_type=model_type, savePath=savepath,importance_first=importance_first)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})

            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # print(re)
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            key_list = list(df_result.keys())
            a = df_result[key_list[0]]
            before = {}
            before["name"] = key_list[0]
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
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
                        # print(str(a.iloc[n][y]), type(a.iloc[n][y]),"1111111111111------------111111111111")
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
            pltlist = loop_add(plts, "变量模型评分分析", string_tp, id)
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            write(df_result[key_list[0]], df_file)

            browsing_process = {}
            browsing_process['name'] = '变量模型评分分析'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before]

            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Jqxxfl(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        机器学习分类
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "jqxxfl"
        return render(request, 'index/jqxxfl.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        # v_round
        validation_ratio = float(json_data.get("validation_ratio"))
        score_evalute = json_data.get("score_evalute")
        method = json_data.get("method")
        n_splits = int(json_data.get("n_splits"))
        # searching = int(json_data.get("searching"))
        searching = json_data.get("searching")
        decimal_num = int(json_data.get("num_select"))
        num = json_data.get("number")
        style = json_data.get('style')
        # 数据筛选
        filter_data = json_data.get("filter")

        if not features:
            features = None
        if searching == "False":
            searching = False
        elif searching == "True":
            searching = True
        # 使用要分析的列名

        # =========================
        # 方法更新了 前端没写好 暂时用一个临时定义的列名代替
        # electedText = ["年龄", "门诊次数", "出院状态", "观测时间内住院次数"]

        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存最新的数据，所有没有最新数据噢'}))
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
            file_s = string3 + "/" + "机器学习分类" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用机器学习分类方法
            try:
                print(group, features, validation_ratio, score_evalute, method, n_splits, num, "================")

                eventlet.monkey_patch(thread=False)
                time_limit = 400  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = ML_Classfication(df, group=group, features=features, decimal_num=int(decimal_num),
                                          validation_ratio=validation_ratio, scoring=score_evalute, method=method,
                                          n_splits=n_splits, searching=searching, savePath=savepath)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result_dict = re[0]
            key_list = list(df_result_dict.keys())
            df_result = re[0][key_list[0]]
            df_result_2 = re[0][key_list[1]]
            df_result_3 = re[0][key_list[2]]

            plts = re[2]

            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            before=get_table(df_result, key_list[0])
            second=get_table(df_result_2, key_list[1])
            three=get_table(df_result_3, key_list[2])
            form = [before, second, three]
            # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "机器学习分类", string_tp, id)

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            df_file2 = file_s + '/' + 'df_result2.pkl'
            df_file3 = file_s + '/' + 'df_result3.pkl'

            write(df_result, df_file)
            write(df_result_2, df_file2)
            write(df_result_3, df_file3)

            browsing_process = {}
            browsing_process['name'] = '机器学习分类'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            browsing_process['df_result_2_2'] = df_file2
            browsing_process['df_result_2_2_2'] = df_file3
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"

            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Jqxxhg(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        机器学习回归
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "jqxxhg"
        return render(request, 'index/jqxxhg.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        decimal_num = json_data.get("num_select")
        # v_round
        validation_ratio = float(json_data.get("validation_ratio"))
        score_evalute = json_data.get("score_evalute")
        method = json_data.get("method")
        n_splits = int(json_data.get("n_splits"))
        searching = json_data.get("searching")
        # searching = False
        num = json_data.get("number")
        # 数据筛选
        filter_data = json_data.get("filter")

        if not features:
            features = None

        if searching == "False":
            searching = False
        elif searching == "True":
            searching = True
        # 使用要分析的列名

        # =========================
        # 方法更新了 前端没写好 暂时用一个临时定义的列名代替
        # electedText = ["年龄", "门诊次数", "出院状态", "观测时间内住院次数"]

        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存最新的数据，所有没有最新数据噢'}))
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
            file_s = string3 + "/" + "机器学习回归" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
            savepath = file_s + '/'
            # 调用机器学习回归方法
            try:
                print(group, features, validation_ratio, score_evalute, method, n_splits, num, "================")

                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    re = ML_Regression(df, group=group, features=features, validation_ratio=validation_ratio,
                                       scoring=score_evalute, method=method, n_splits=n_splits, searching=searching,
                                       decimal_num=int(decimal_num), savePath=savepath)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result_dict = re[0]
            key_list = list(df_result_dict.keys())
            df_result = re[0][key_list[0]]
            df_result_2 = re[0][key_list[1]]

            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            before = get_table(df_result, key_list[0])
            second = get_table(df_result_2, key_list[1])
            form = [before, second]

            pltlist = loop_add(plts, "机器学习回归", string_tp, id)

            # 写入数据
            df_file = file_s + '/' + 'df_result.xlsx'
            df_file2 = file_s + '/' + 'df_result2.xlsx'

            write(df_result, df_file)
            write(df_result_2, df_file2)

            browsing_process = {}
            browsing_process['name'] = '机器学习回归'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            browsing_process['df_result_2_2'] = df_file2
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]

            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            form = [before, second]
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"

            }
        except Exception as e:
            print(e)
            mylock.release()
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Jqxxjl(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        机器学习聚类
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "jqxxjl"
        return render(request, 'index/jqxxjl.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        decimal_num = json_data.get("num_select")
        method = json_data.get("method")
        searching = json_data.get("searching")
        num = json_data.get("number")
        # 获取是否保存结果
        log_save = json_data.get('save')
        # 数据筛选
        filter_data = json_data.get("filter")

        if not log_save:
            log_save = 0
        else:
            log_save = int(log_save)

        if not features:
            features = None
        if not group:
            group = None
        # 使用要分析的列名
        if searching == "False":
            searching = False
        elif searching == "True":
            searching = True
        # =========================
        # 方法更新了 前端没写好 暂时用一个临时定义的列名代替

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存最新的数据，所有没有最新数据噢'}))
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
            # 写入数据库的路径
            # end01 = "/static/"
            end = "/"
            # 本地代码上面
            # end = "\\"
            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]

            file_s = string3 + "/" + "机器学习聚类" + id
            savepath = file_s + '/'
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))
                # 调用机器学习聚类方法
            try:
                print(features, group, method, num, "================")
                eventlet.monkey_patch(thread=False)
                time_limit = 30  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):
                    print(features, group, method, searching, decimal_num, savepath)
                    re = ML_Clustering(df, features=features, group=group, method=method, searching=searching,
                                       decimal_num=int(decimal_num), savePath=savepath)
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result_dict = re[0]
            key_list = list(df_result_dict.keys())
            df_result = re[0][key_list[0]]
            df_result_2 = re[0][key_list[1]]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")

            # 获取返还给页面的df数据
            a = df_result
            before = {}
            before["name"] = key_list[0]
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
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
                            if dict_x[str(i)]:
                                i = str(i) + "2"
                                dict_x[str(index)] = str(a.iloc[n][y])
                        except Exception as e:
                            dict_x[str(index)] = str(a.iloc[n][y])
                        y += 1
                    before['info'].append(dict_x)

            ## 2020/12/31 Owen Yang: 不显示聚类标注后的完整数据表
            # b = df_result_2
            # second = {}
            # second["name"] = key_list[1]
            # if list(b):
            #     # 保存第一列的列名
            #     second['Listing'] = []
            #     name = list(b)
            #     # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
            #     # 循环列名
            #     for i in range(len(list(b))):
            #         second['Listing'].append(name[i])
            #         # if i == 8:
            #         #     break
            #     # 添加数据
            #     second['info'] = []
            #     for n in range(len(b[name[1]])):
            #         dict_x = {}
            #         y = 0
            #         # 制定第一行 所有列的所有info数据
            #         for i in name:
            #             i = str(i)
            #             # 最多添加八个数据
            #             # if y == 8:
            #             #     break
            #             try:                            
            #                 if dict_x[str(i)]:
            #                     i = str(i) + "2"
            #                     dict_x[str(i)] = str(b.iloc[n][y])
            #             except Exception as e:
            #                 dict_x[str(i)] = str(b.iloc[n][y])
            #             y += 1
            #         second['info'].append(dict_x)
            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            df_file2 = file_s + '/' + 'df_result2.pkl'

            write(df_result, df_file)
            write(df_result_2, df_file2)
            # df_result_2.to_excel(df_file2)

            browsing_process = {}
            browsing_process['name'] = '机器学习聚类'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            browsing_process['df_result'] = df_file2
            if int(log_save) == 1:
                # 找到之前的处理后的数据替换掉
                try:
                    # 找到这个项目的最新处理的数据
                    # print('===========')
                    # print(project_id)
                    latest_data = Browsing_process.objects.get(file_old=project_id, is_latest=1)
                    if latest_data:
                        # 修改数据结果
                        latest_data.is_latest = 0
                        latest_data.save()
                except Exception as e:
                    print(e)
                    # print(333333333333)
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
                        browsing.is_latest = 1
                        browsing.save()


                except Exception as e:
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,
                                                               file_old_id=project_id)
                    browsing.is_latest = 1
                    browsing.save()
                    project = browsing.id

            else:
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
            # form = [before, second] ## 2020/12/31 Owen Yang: 不显示聚类标注后的完整数据表
            form = [before, ]

            context = {
                'project': str(project),
                'str': str_s,
                "form": form,
                "text": "tableDescription"
            }
            print(context)
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return http.JsonResponse({'code': '200', 'error': '分析成功', 'context': context})
        # return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class D_roc(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        分类多模型综合分析
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        context["value"] = "All"
        context["begin"] = "end"
        context["name"] = "d_roc"
        return render(request, 'index/d_roc.html', context=context)

    def post(self, request, project_id, data_id):
        """
        接收传的列名
        :param request:
        :return:
        """

        json_data = json.loads(request.body.decode())
        group = json_data.get("group")
        features = json_data.get("features")
        methods = json_data.get("method")

        testsize = float(json_data.get("testsize"))
        boostrap = int(json_data.get("boostrap"))
        decimal_num = json_data.get("num_select")
        # 数据筛选
        filter_data = json_data.get("filter")

        searching = json_data.get("searching")
        # searching = False
        num = json_data.get("number")
        
        if searching == 'true':
            searching = True
        else:
            searching = False
        # 使用要分析的列名

        # =========================
        # 方法更新了 前端没写好 暂时用一个临时定义的列名代替
        # electedText = ["年龄", "门诊次数", "出院状态", "观测时间内住院次数"]

        if not all([group]):
            return HttpResponse(json.dumps({'code': '201', 'error': '请填写变量'}))

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
                return HttpResponse(json.dumps({'code': 202, 'error': '你还没有保存的最新的数据，所有没有最新数据噢'}))
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
            # 写入数据库的路径
            end01 = "/static/"
            end = "/"

            if int(num) == 0:
                string3 = file_path[:file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string3 = string[:string.rfind(end)]
            # strint3路径 为用户创的 项目下面 

            # string2 为图片的静态路径
            string_tp = string3[string3.rfind(end01):]

            # file_s 为这方法存图片以及df表的文件夹  绝对路径
            file_s = string3 + "/" + "分类多模型综合分析" + id
            savepath = file_s + '/'
            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

                # 调用多模型分析（ROC评价）方法
            try:

                eventlet.monkey_patch(thread=False)
                time_limit = 300  # set timeout time 3s
                success = None
                with eventlet.Timeout(time_limit, False):

                    # 2020/12/31 Owen Yang: 已经删除 ylim, title 的入参，前端可以直接改
                    # methods = [] # 多选下拉菜单候选项：XGB分类, logistics回归，支持向量机分类，神经网络分类，随机森林，自适应提升树

                    re = two_groups_classfication_multimodels(df, group=group, features=features, methods=methods,
                                                              decimal_num=int(decimal_num), testsize=testsize,
                                                              boostrap=boostrap, searching=searching, savePath=savepath)
                    
                    success = 1
                if success == None:
                    return http.JsonResponse({'code': 1004, 'error': '分析超时请稍后重新分析'})
            except Exception as e:
                print(33333333333)
                print(e)
                return http.JsonResponse({'code': 1004, 'error': '分析失败，请选择正确变量'})
            try:
                if re['error']:
                    return http.JsonResponse({'code': 1005, 'error': re['error']})
            except Exception as e:
                print('正常')
            # 提取plt  和str 数据
            # print(re)
            df_result_dict = re[0]
            key_list = list(df_result_dict.keys())
            df_result = re[0][key_list[0]]
            df_result_2 = re[0][key_list[1]]

            df_result = df_result.fillna('')
            df_result_2 = df_result_2.fillna('')

            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # print("=============---------======111")
            # 获取返还给页面的df数据
            before = get_table(df_result, key_list[0])
            second = get_table(df_result_2, key_list[1])
            form = [before, second]
            # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "分类多模型综合分析", string_tp, id)

            # 写入数据
            df_file = file_s + '/' + 'df_result.pkl'
            df_file2 = file_s + '/' + 'df_result2.pkl'
            write(df_result, df_file)
            write(df_result_2, df_file2)

            browsing_process = {}
            browsing_process['name'] = '分类多模型综合分析'
            browsing_process['str_result'] = str_s
            browsing_process['df_result_2'] = df_file
            browsing_process['df_result_2_2'] = df_file2
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            try:
                process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                    Max('order'))
                print(process)
                if process:
                    order = int(process['order__max']) + 1
                    print(order)
                    browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                               user_id=request.user.id,
                                                               order=str(order),
                                                               file_old_id=project_id)
                    project = browsing.id

            except Exception as e:
                print(e)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order='1',
                                                           file_old_id=project_id)
                project = browsing.id

            # 把返回结果数据封装到列表里面
            context = {
                'project': project,
                'img': pltlist,
                'str': str_s,
                "form": form,
                "text": "tableDescriptionIImages1"
            }
        except Exception as e:
            print(e)
            return HttpResponse(json.dumps({'code': '201', 'error': '网络延迟，请稍后再试！！！'}))
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


# class D_dzb(LoginRequiredMixin, View):
#     def get(self, request, project_id, data_id):
#         """
#         多模型分析(多指标评价)
#         :param request:
#         :return:
#         """
#         context = get_s(request, project_id, data_id)
#         # 单选加多选
#         context["value"] = "All"
#         context["begin"] = "end"
#         context["name"] = "d_dzb"
#         return render(request, 'index/d_dzb.html', context=context)

#     def post(self, request, project_id, data_id):
#         """
#         接收传的列名
#         :param request:
#         :return:
#         """
#         json_data = json.loads(request.body.decode())
#         group = json_data.get("group")
#         features = json_data.get("features")
#         scores = json_data.get("scores")
#         plt_score = json_data.get("plt_score")

#         testsize = float(json_data.get("testsize"))
#         boostrap = int(json_data.get("boostrap"))

#         ylim_min = json_data.get("ylim_min")
#         ylim_max = json_data.get("ylim_max")
#         title = json_data.get("title")

#         num = json_data.get("number")

#         if not features:
#             features = None
#         if not ylim_max:
#             ylim_max = None
#         else:
#             ylim_min = int(ylim_min)

#         if not ylim_min:
#             ylim_min = None
#         else:
#             ylim_max = int(ylim_max)
#         # 使用要分析的列名

#         # =========================
#         # 方法更新了 前端没写好 暂时用一个临时定义的列名代替
#         # electedText = ["年龄", "门诊次数", "出院状态", "观测时间内住院次数"]

#         if not all([group]):
#             return HttpResponse(json.dumps({'code': '201', 'error': '缺少参数'}))

#         # 创建一个uuid唯一值
#         id = uuid.uuid1()
#         id = str(id)
#         # 获取传的列名
#         if int(num) == 0:
#             # 获取数据库的对象
#             user = File_old.objects.get(id=project_id)
#             # 获取原文件的路径
#             file_path = user.path

#         elif int(num) == 1:
#             try:
#                 process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest = 1)
#             except Exception as e:
#                 return HttpResponse(json.dumps({'code': 202, 'error': '你还没有处理过数据，所有没有最新数据噢'}))
#             re_d = eval(process.process_info)
#             file_path = re_d.get("df_result")
#         try:
#             # 打开文件
#             # 查询该用户的用户等级
#             try:
#                 grade = Member.objects.get(user_id=request.user.id)
#                 # 查询当前用户的等级
#                 member = MemberType.objects.get(id=grade.member_type_id)            
#             except Exception as e:
#                 return http.JsonResponse({'code': 1002, 'error': '会员查询失败，请重试'}) 
#             try:
#                 try:
#                     df = pd.read_excel(file_path, nrows=int(member.number))
#                 except Exception as e:
#                     print(e)
#                     df = pd.read_csv(file_path, nrows=int(member.number))
#             except Exception as e:
#                 print(e)
#                 df = pd.read_csv(file_path, nrows=int(member.number),encoding='gbk')
#             except Exception as e:
#                 print(e)
#                 return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

#             # 调用多模型分析(多指标评价)方法
#             try:
#                 print(group, features,scores,plt_score,ylim_min, ylim_max,testsize, boostrap,title, num, "================")
#                 scores=['F1-score','Roc_auc','Accuracy','Recall']
#                 re = two_groups_multiscore_multimodels(df, group, features,scores,plt_score,ylim_min, ylim_max,testsize, boostrap,title)
#             except Exception as e:
#                 print(e)
#                 return http.JsonResponse({'code': 1004, 'error': '分析文件或填写的数据不正确，分析失败'})
#             # 提取plt  和str 数据
#             # print(re)
#             df_result = re[0]
#             plts = re[2]
#             str_s = re[1]
#             str_s = str_s.replace("\n", "<br/>")
#             # print("=============---------======111")
#             # 获取返还给页面的df数据
#             a = df_result
#             before = {}
#             before["name"] = "结果数据描述"
#             if list(a):
#                 # 保存第一列的列名
#                 before['Listing'] = []
#                 name = list(a)
#                 # list01 = ['count', 'skewness', 'kurtosis', 'standarderror', 'statistics', 'p', 'method ']
#                 # 循环列名
#                 for i in range(len(list(a))):
#                     before['Listing'].append(name[i])
#                     # if i == 8:
#                     #     break
#                 # 添加数据
#                 before['info'] = []
#                 for n in range(len(a[name[1]])):
#                     dict_x = {'name': a.iloc[n].name}
#                     y = 0
#                     # 制定第一行 所有列的所有info数据
#                     for i in name:
#                         i = str(i)
#                         # 最多添加八个数据
#                         # if y == 8:
#                         #     break
#                         try:                            
#                             if dict_x[i]:
#                                 t = str(uuid.uuid1())
#                                 i = str(i) + t
#                                 dict_x[i] = str(a.iloc[n][y])
#                         except Exception as e:
#                             dict_x[i] = str(a.iloc[n][y])
#                         y += 1
#                     before['info'].append(dict_x)

#             # print("=============---------======222")
#             # 把对象写入前端
#             # plts.plot(randn(50).cumsum(), 'k--')
#             # 写入数据库的路径
#             end01 = "/static/"
#             end = "/"
#             # 本地代码上面
#             # end = "\\"
#             if int(num) == 0:
#                 string3 = file_path[:file_path.rfind(end)]
#             else:
#                 string = file_path[:file_path.rfind(end)]
#                 string3 = string[:string.rfind(end)]
#             # strint3路径 为用户创的 项目下面 

#             # string2 为图片的静态路径
#             string_tp = string3[string3.rfind(end01):]

#             # file_s 为这方法存图片以及df表的文件夹  绝对路径
#             file_s = string3 + "/" + "多模型分析(多指标评价)" + id

#             print(file_path, "======file_path======")
#             print(string_tp, "======string_tp======")
#             print(file_s, "======file_s======")
#             print(string3, "======string3======")
#             try:
#                 # 创建存图片和df表的文件夹的文件夹
#                 os.makedirs(file_s)
#             except Exception as e:
#                 print(e)
#                 return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))                
#             # 把得到的图片保存
#             # 返回前端 图片的静态路径
#             plt_file = string_tp + "/" + "多模型分析(多指标评价)" + id + "/" + "plt.png"
#             plt_file02 = file_s + "/" + "plt.png"
#             plt.savefig(plt_file02,dpi=600, bbox_inches = 'tight')
#             plt.close()
#             # 写入数据
#             df_file = file_s + '/' + 'df_result.xlsx'
#             df_result.to_excel(df_file)
#             browsing_process = {}
#             browsing_process['name'] = '多模型分析(多指标评价)'
#             browsing_process['plt'] = plt_file
#             browsing_process['str_result'] = str_s
#             browsing_process['df_result_2'] = df_file
#             try:
#                 process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(Max('order'))
#                 print(process)
#                 if process:
#                     order = int(process['order__max']) + 1
#                     print(order)
#                     browsing = Browsing_process.objects.create(process_info=browsing_process,
#                                                               user_id=request.user.id,
#                                                               order=str(order),
#                                                               file_old_id=project_id)
#                     project = browsing.id

#             except Exception as e:
#                 print(e)
#                 browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
#                                                           order='1',
#                                                           file_old_id=project_id)
#                 project = browsing.id

#             # 把返回结果数据封装到列表里面
#             form = [before]

#             context = {
#                 'project': project,
#                 'img': plt_file,
#                 'str': str_s,
#                 "form": form, 
#                 "text": "tableDescriptionImg"

#             }
#         except Exception as e:
#             print(e)
#             return HttpResponse(json.dumps({'code': '201', 'error': '请在试一次'}))
#         return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))


class Wait(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        return render(request, 'index/wait.html')


class Nri(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        """
        净重分类指数
        :param request:
        :return:
        """
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "nri"
        context["begin"] = "end"
        context["name"] = "nri"
        return render(request, 'index/nri.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """
        # 接收json数据
        json_data = json.loads(request.body.decode())
        pstd = json_data.get("pstd")
        pnew = json_data.get("pnew")
        gold = json_data.get("gold")
        cut = json_data.get("cut")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 获取是否保存结果
        log_save = json_data.get('save')
        # 原始还是处理后的数据
        num = json_data.get("number")
        id = uuid.uuid1()
        id = str(id)
        print(cut)
        if cut:
            cut01 = []
            cut02 = cut[0].split(',')
            for i in cut02:
                cut01.append(float(i))
            cut = cut01
        # 数据筛选
        filter_data = json_data.get("filter")

        if not log_save:
            log_save = 0
        else:
            log_save = int(log_save)

        # 获取传的列名
        if num == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif num == 1:
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

            # # 写入数据

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
            file_s = string3 + "/" + "NRI" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savepath = file_s + '/'
            try:
                print(222222222)
                print(pstd)
                print(gold)
                print(cut)
                print(decimal_num)
                re = R_NRI_ana(df_input=df, pstd=pstd, pnew=pnew, gold=gold, cut=cut, path=savepath,
                               decimal_num=int(decimal_num))


            except Exception as e:
                # print(1111111111111)
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请选择正确变量'})
            # 提取df
            print(44444444444444)
            print(re)
            print(55555555555)
            # save_data = re[3]
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 计量数据描述
            a = df_result
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                name[0] = '&emsp;'
                # 循环列名s
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {'&emsp;': a.iloc[n].name}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for i in name:
                        dict_x[i] = str(a.iloc[n][y])
                        y += 1
                    print(dict_x)
                    before['info'].append(dict_x)
            # 把得到的图片保存
            # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "NRI", string_tp, id)

            df_file = file_s + '/' + 'dfresult222.pkl'
            write(df_result, df_file)

            # save_files = file_s + '/' + 'df_result.pkl'
            # write(save_data, save_files)

            browsing_process = {'name': 'NRI', 'df_result': df_file, 'str_result': str_s}
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            if int(log_save) == 1:
                # 找到之前的处理后的数据替换掉
                try:
                    # 找到这个项目的最新处理的数据
                    latest_data = Browsing_process.objects.get(file_old=project_id, is_latest=1)
                    if latest_data:
                        # 修改数据结果
                        latest_data.is_latest = 0
                        latest_data.save()
                except Exception as e:
                    print(e)
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    # print(process)
                    if process:
                        order = int(process.order) + 1
                        print(order)
                        browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                                   user_id=request.user.id,
                                                                   order=order,
                                                                   file_old_id=project_id)
                        project = browsing.id
                        browsing.is_latest = 1
                        browsing.save()


                except Exception as e:
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,

                                                               file_old_id=project_id)
                    browsing.is_latest = 1
                    browsing.save()
                    project = browsing.id

            else:
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    # print(process)
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
            plt_list = pltlist
            context = {
                'img': plt_list,
                'project': project,
                "form": form,
                "text": "tableDescriptionIImages1",
                'str': str_s
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '网络延迟，请稍后再试！！！'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))

#Rcs
class Rcs(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        context = get_s(request, project_id, data_id)
        # 单选加多选
        # context["value"] = "All"
        context["value"] = "rcs"
        context["begin"] = "end"
        context["name"] = "rcs"
        return render(request, 'index/rcs.html', context=context)

    def post(self, request, project_id, data_id):

        """
        接收传的列名
        :param request:
        :return:
        """
        # 接收json数据
        json_data = json.loads(request.body.decode())
        pstd = json_data.get("pstd")
        pnew = json_data.get("pnew")
        gold = json_data.get("gold")
        cut = json_data.get("cut")
        # 获取小数位
        decimal_num = json_data.get("num_select")
        # 获取是否保存结果
        log_save = json_data.get('save')
        # 原始还是处理后的数据
        num = json_data.get("number")
        id = uuid.uuid1()
        id = str(id)
        print(cut)
        if cut:
            cut01 = []
            cut02 = cut[0].split(',')
            for i in cut02:
                cut01.append(float(i))
            cut = cut01
        # 数据筛选
        filter_data = json_data.get("filter")

        if not log_save:
            log_save = 0
        else:
            log_save = int(log_save)

        # 获取传的列名
        if num == 0:
            # 获取数据库的对象
            user = File_old.objects.get(id=project_id)
            # 获取原文件的路径
            file_path = user.path_pa

        elif num == 1:
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

            # # 写入数据

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
            file_s = string3 + "/" + "NRI" + id

            try:
                # 创建存图片和df表的文件夹的文件夹
                os.makedirs(file_s)
            except Exception as e:
                print(e)
                return HttpResponse(json.dumps({'code': '1003', 'error': '创建失败'}, cls=NpEncoder))

            savepath = file_s + '/'
            try:
                print(222222222)
                print(pstd)
                print(gold)
                print(cut)
                print(decimal_num)
                re = R_NRI_ana(df_input=df, pstd=pstd, pnew=pnew, gold=gold, cut=cut, path = savepath, decimal_num=int(decimal_num))


            except Exception as e:
                # print(1111111111111)
                print(e)
                return http.JsonResponse({'code': 1005, 'error': '分析失败，请选择正确变量'})
            # 提取df
            print(44444444444444)
            print(re)
            print(55555555555)
            # save_data = re[3]
            df_result = re[0]
            plts = re[2]
            str_s = re[1]
            str_s = str_s.replace("\n", "<br/>")
            # 计量数据描述
            a = df_result
            before = {}
            if list(a):
                # 保存第一列的列名
                before['Listing'] = []
                name = list(a)
                name[0] = '&emsp;'
                # 循环列名s
                for i in range(len(list(a))):
                    before['Listing'].append(name[i])
                # 添加数据
                before['info'] = []
                for n in range(len(a[name[1]])):
                    dict_x = {'&emsp;': a.iloc[n].name}
                    y = 0
                    # 制定第一行 所有列的所有info数据
                    for i in name:
                        dict_x[i] = str(a.iloc[n][y])
                        y += 1
                    print(dict_x)
                    before['info'].append(dict_x)
            # 把得到的图片保存
            # 返回前端 图片的静态路径
            pltlist = loop_add(plts, "NRI", string_tp, id)

            df_file = file_s + '/' + 'dfresult222.pkl'
            write(df_result, df_file)

            # save_files = file_s + '/' + 'df_result.pkl'
            # write(save_data, save_files)

            browsing_process = {'name': 'NRI', 'df_result': df_file, 'str_result': str_s}
            for i in range(len(pltlist)):
                if i == 0:
                    name = 'plt'
                    browsing_process[name] = pltlist[i]
                else:
                    name = 'plt' + str(i + 1)
                    browsing_process[name] = pltlist[i]
            if int(log_save) == 1:
                # 找到之前的处理后的数据替换掉
                try:
                    # 找到这个项目的最新处理的数据
                    latest_data = Browsing_process.objects.get(file_old=project_id, is_latest=1)
                    if latest_data:
                        # 修改数据结果
                        latest_data.is_latest = 0
                        latest_data.save()
                except Exception as e:
                    print(e)
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    # print(process)
                    if process:
                        order = int(process.order) + 1
                        print(order)
                        browsing = Browsing_process.objects.create(process_info=browsing_process,
                                                                   user_id=request.user.id,
                                                                   order=order,
                                                                   file_old_id=project_id)
                        project = browsing.id
                        browsing.is_latest = 1
                        browsing.save()


                except Exception as e:
                    browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                               order=1,

                                                               file_old_id=project_id)
                    browsing.is_latest = 1
                    browsing.save()
                    project = browsing.id

            else:
                try:
                    process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).latest(
                        'order')
                    # print(process)
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
            plt_list = pltlist
            context = {
                'img': plt_list,
                'project': project,
                "form": form,
                "text": "tableDescriptionIImages1",
                'str': str_s
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1006, 'error': '网络延迟，请稍后再试！！！'})
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': context}, cls=NpEncoder))

class Dca(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id, data_id):
        context = get_s(request, project_id, data_id)
        context["value"] = "dca"
        context["begin"] = "end"
        context["name"] = "dca"
        return render(request, 'index/dca.html', context=context)

    def post(self, request, project_id, data_id):
        json_data = json.loads(request.body.decode())
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': ""}, cls=NpEncoder))












