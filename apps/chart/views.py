import json
import os
import uuid
import time
import threading

import eventlet
import random
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from django.db.models import Max
from numpy.matlib import randn
from django import http
from django.shortcuts import render
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from libs.get import get_s,loop_add,write,read, filtering_view

from AnalysisFunction.X_5_SmartPlot import pie_graph, forest_plot, box_plot, scatter_plot,horizontal_bar_plot, violin_plot, rel_plot, strip_plot, stackbar_plot, comparison_plot, point_line_plot, dist_plot
from apps.index_qt.models import File_old, Browsing_process, Member, MemberType
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication

class PieGraph(APIView):
    """分类排序饼图"""
    authentication_classes = [MyBaseAuthentication, ]

    def get(self, request, project_id):
        context = {
            'value': 'single',
            'project_id': project_id,
            "begin" : "chart"
            
        }
        return render(request, 'index/flbt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取匹配的列表
        feature = json_data.get('feature')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取图表中展示的前几位
        top_num = int(json_data.get('top_num'))
        # 获取字体大小
        font = int(json_data.get('font'))
        # 获取初始角度
        startangle = int(json_data.get('startangle'))
        if not all([feature]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        number = int(number)
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        if number == 0:
            # 查询数据库原始文件
            print(project_id)
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        ##新增echarts图表数据：
        col = feature
        vc = df[col].value_counts(dropna=False)
        data = json.dumps(dict(zip(vc.index, vc)), ensure_ascii=False)
        feature = json.dumps({'feature': col + '分布图'}, ensure_ascii=False)
        ##以下是原代码，全部注释掉
        '''
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
            
        string3 = string2 + '/pie_graph' + str(uuid01)
        
        try:
            os.mkdir(string3)
        except Exception as e:
            print(e)
        print(string3)
        
        savepath = string3+'/'
        print(savepath, "savepath")
        # 分析函数
        try:
            info = pie_graph(df, feature, top_num, font_size=font, startangle=startangle, title="饼图",path=savepath)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})

        # 保存图片
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "pie_graph", route01, str(uuid01))

        # plts = info

        # route02 = route01 + '/pie_graph' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/pie_graph' + str(uuid01) + "/" + "plt.png"
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '分类排序饼图'
        for i in range(len(pltlist)):
            if i == 0:
                name = 'plt'
                browsing_process[name] = pltlist[i]
            else:
                name = 'plt' + str(i + 1)
                browsing_process[name] = pltlist[i]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})
        '''
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'feature':feature,'data':data,'name':'flbt'})
class HorizontalBarPlot(APIView):
    """水平柱状图"""
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'secondSingle',
            'project_id': project_id,
            "begin" : "chart",
            'name':'spzt'
        }
        return render(request, 'index/spzt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        name = json_data.get('name')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取数据列
        value = json_data.get('value')
        
        if not all([name, value]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        number = int(number)
        file_path = None
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        
        string3 = string2 + '/horizontal_bar_plot' + str(uuid01)
        try:
            os.mkdir(string3)
        except Exception as e:
            print(e)
        sapath = string3+'/'
        # 分析函数
        try:
            try:
                file = File_old.objects.get(id=project_id)
                title = str(file.project_name)
            except Exception as e:
                title= '水平柱状图'
                print(e)
                return http.JsonResponse({'code': 2001, 'error': '表名获取失败'})
            info = horizontal_bar_plot(df, name, value, title, path=sapath)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        # 保存图片
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "horizontal_bar_plot", route01, str(uuid01))
        
        # plts = info
        # # 设置图片大小
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/horizontal_bar_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/horizontal_bar_plot' + str(uuid01) + "/" + "plt.png"
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '水平柱状图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0],
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})

class ForestPlot(APIView):
    """森林图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'different',
            'project_id': project_id,
            "begin" : "chart"
        }
        return render(request, 'index/slt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        print(json_data)
        # 获取名称列
        name = json_data.get('name')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取数据列
        value = json_data.get('value')
        # 获取str或list误差列
        err = list(json_data.get('err'))
        # 获取方向
        direct = json_data.get('direct')
        # 获取x轴旋转角度
        rotation = int(json_data.get('rotation'))
        # 获取yilm列表
        yilm01 = list(json_data.get('ylim'))
        # 获取fig_size 列表
        fig_size01 = list(json_data.get('fig_size'))
        # 获取是否标记保护因素和危险因素
        colormark = json_data.get('colormark')
        # 获取中间实体采用的形状标记
        mean_mark = json_data.get('mean_mark')
        # 获取表标题
        title = json_data.get('title')
        yilm = []
        for i in yilm01:
            yilm.append(int(i))
            
        fig_size = []
        for i in fig_size01:
            fig_size.append(int(i))
            
        if not all([name, value,err,direct]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/forest_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/forest_plot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            try:
                file = File_old.objects.get(id=project_id)
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 2001, 'error': '表名获取失败'})
            # info = forest_plot(df,'时间','事件',['多分组','配对分组'],'horizontal','aaaaa',ylim=[0.0,1.0],fig_size=[10,10])
            info = forest_plot(df, name, value, err, direct=direct, title=title, path=savepath,rotation_=rotation, ylim=yilm,fig_size=fig_size,color_mark=colormark,mean_mark=mean_mark)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        # print(info)
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        plts=info
        # 保存图片
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "forest_plot", route01, str(uuid01))
        
        # # 保存图片
        # # 设置图片大小
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/forest_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/forest_plot' + str(uuid01) + "/" + "plt.png"
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '森林图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})

class BoxPlot(APIView):
    """箱图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'different',
            'project_id': project_id,
            "begin" : "chart",
            'name':'xt'
        }
        return render(request, 'index/xt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        features = list(json_data.get('features'))
        # 获取哪种数据
        number = json_data.get('number')
        # 获取分组列
        group = json_data.get('group')
        # 获取组内分组
        hue = json_data.get('hue')

        if not all([features]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})


        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/box_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/box_plot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            if not group:
                group = None
            if not hue:
                hue = None
            info = box_plot(df, features=features, group=group,hue_=hue, title="箱图",path=savepath)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "box_plot", route01, str(uuid01))
        
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/box_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/box_plot' + str(uuid01) + "/" + "plt.png"
        # print(plt_file02, "-=-=-=")
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '箱图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})

class ScatterPlot(APIView):
    """散点线图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'dobuleSingle',
            'project_id': project_id,
            "begin" : "chart",
            "name":'sdxt'
        }
        return render(request, 'index/sdxt.html', context=context)

    def post(self, request, project_id):
        import matplotlib 
        print(matplotlib.matplotlib_fname())
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        y = json_data.get('y')
        # 获取组内分组
        hue = json_data.get('hue')
        # 获取哪种数据
        number = json_data.get('number')

        if not all([x,y]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/scatter_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/scatter_plot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            if not hue:
                hue = None
            info = scatter_plot(df, x_=x, y_=y,hue_=hue,title="散点线图", path=savepath)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "scatter_plot", route01, str(uuid01))
        
        
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/scatter_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/scatter_plot' + str(uuid01) + "/" + "plt.png"
        # plt.savefig(plt_file02)
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号  
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '散点线图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})

class ViolinPlot(APIView):
    """小提琴图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'dobuleSingle',
            'project_id': project_id,
            "begin" : "chart",
            "name":'xtqt'
        }
        return render(request, 'index/xtqt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        y = json_data.get('y')
        # 获取组内分组
        hue = json_data.get('hue')
        # 获取哪种数据
        number = json_data.get('number')

        if not all([x,y]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/violin_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/violin_plot' + str(uuid01))
        except Exception as e:
            print(e)          
        # 分析函数
        try:
            if not hue:
                hue = None
            info = violin_plot(df, x_=x, y_=y,hue_=hue, title="小提琴图", path=string3)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        # plts = info
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        route02 = route01 + info[0]
        # 展示到前端的路径
        plt_file = route02
        
        # # 保存图片
        # plts = info
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/violin_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/violin_plot' + str(uuid01) + "/" + "plt.png"
        # plt.savefig(plt_file02)
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号  
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '小提琴图'
        browsing_process['plt'] = plt_file
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': plt_file
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})
        
class RelPlot(APIView):
    """线混合图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'fourSingle',
            'project_id': project_id,
            "begin" : "chart",
            "name":'xhht'
        }
        return render(request, 'index/xhht.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        y = json_data.get('y')
        # 获取组内分组
        hue = json_data.get('hue')
        # 获取离散型变量
        col = json_data.get('col')
        # 获取绘制散点图还是线图
        kind = json_data.get('kind')
        # 获取哪种数据
        number = json_data.get('number')
        if not all([x,y]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
            
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/rel_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/rel_plot' + str(uuid01))
        except Exception as e:
            print(e)
        # 分析函数
        try:
            if hue == 'None':
                hue = None
            if col == 'None':
                col = None

            info = rel_plot(df, x_=x, y_=y,hue_=hue,col_=col,kind_=kind, title="线混合图", path=string3)
            # info = rel_plot(df, x_=x, y_=y,kind_=kind)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        plts = info
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        route02 = route01 + plts[0]
        plt_file = route02
        
        # plts = info
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/rel_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/rel_plot' + str(uuid01) + "/" + "plt.png"
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '线混合图'
        browsing_process['plt'] = plt_file
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': plt_file
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})
        
class StripPlot(APIView):
    """分类散点图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'secondSingle',
            'project_id': project_id,
            "begin" : "chart",
            "name":'flsd'
        }
        return render(request, 'index/flsd.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        y = json_data.get('y')
        # 获取哪种数据
        number = json_data.get('number')

        if not all([x,y]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
            
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/strip_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/strip_plot' + str(uuid01))
        except Exception as e:
            print(e)
            
        # 分析函数
        try:
            info = strip_plot(df, x_=x, y_=y, title="分类散点图", path=string3)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        # 分析成功保存图片
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        plts = info
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        route02 = route01 + plts[0]
        plt_file = route02
        # route02 = route01 + '/strip_plot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/strip_plot' + str(uuid01) + "/" + "plt.png"
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号  
        # plt.savefig(plt_file02)
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '分类散点图'
        browsing_process['plt'] = plt_file
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': plt_file
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})       

class StackbarPlot(APIView):
    """堆积柱状图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'secondSingle',
            'project_id': project_id,
            "begin" : "chart",
            "name":'djzzt'
        }
        return render(request, 'index/djzzt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        y = json_data.get('y')
        # 获取方向
        direction = json_data.get('direction')
        # 获取图片风格
        palette_style = json_data.get('palette_style')
        # # 水平标签旋转角度，默认水平方向，如标签过长，可设置一定角度，比如设置rotation = 40
        # rotation = json_data.get('rotation')
        # # 获取分类标签的位置
        # location = json_data.get('location')
        # 获取哪种数据
        number = json_data.get('number')
        # 数据筛选
        filter_data = json_data.get("filter")

        if not all([x,y]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)
            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
            
        if filter_data:
            df = filtering_view(df,json_data)
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/stackbar_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/stackbar_plot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            info = stackbar_plot(df_input=df, x_=x, y_=y,  path=savepath,direction=direction,palette_style=palette_style)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 分析成功保存图片
        # 保存图片
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "stackbar_plot", route01, str(uuid01))


        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '堆积柱状图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})       

class DistPlot(APIView):
    """频率分布直方图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'much',
            'project_id': project_id,
            "begin" : "chart",
            "name":'plfbzft'
        }
        return render(request, 'index/plfbzft.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # 获取名称列
        x = json_data.get('x')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取指定行和列col_size，row_size：int 指定行和列
        col_size = json_data.get('col_size')
        row_size = json_data.get('row_size')
        # 数据筛选
        filter_data = json_data.get("filter")

        if not all([x]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)
            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
            
        if filter_data:
            df = filtering_view(df,json_data)
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/dist_plot' + str(uuid01)
        try:
            os.mkdir(string2 + '/dist_plot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            print(x)
            try:
                col_size = int(col_size)
            except Exception as e:
                print(e)
                col_size = None
            try:
                row_size = int(row_size)
            except Exception as e:
                print(e)
                row_size = None
            print(col_size)
            print(row_size)
            info = dist_plot(df_input=df, features=x,  path=savepath,title='',col_size=col_size,row_size=row_size)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 分析成功保存图片
        # 保存图片
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "dist_plot", route01, str(uuid01))

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '绘制频率分布直方图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})       
class ComparisonPlot(APIView):
    """比较图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'All',
            'project_id': project_id,
            "begin" : "chart",
            "name":'bjt'
        }
        return render(request, 'index/bjt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # list选择的特征名列表
        features = json_data.get('features')
        # str分组变量
        group = json_data.get('group')
        # str仅concat_way=='samedia'时可用，图片中Y轴的显示名
        name_y = json_data.get('name_y')
        #str默认为None，仅concat_way=='free'时可用，第二个分组类型
        hue = json_data.get('hue')
        # str默认为box，作图方式，可选kind='bar','box','violin'
        kind = json_data.get('kind')
        # str默认为None,图片连接方式，可选concat_way='None','free','samedia'
        concat_way = json_data.get('concat_way')
        row_size = json_data.get('row_size')
        col_size = json_data.get('col_size')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取图片风格
        palette_style = json_data.get('palette_style')
        # 数据筛选
        filter_data = json_data.get("filter")

        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)
            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        if filter_data:
            df = filtering_view(df,json_data)
        # 分析函数
        try:
            if hue == '':
                hue = None
            if name_y == '':
                name_y = None
            if row_size == '':
                row_size = None
            if col_size == '':
                col_size = None
            concat_way = concat_way.lower()
            if concat_way == 'samedia':
                row_size = None
                col_size = None

            # if kind == 'bar':
            #     isstack = 'True'
            # else:
            #     isstack = None
            # 分析成功保存图片
            end = '/'
            uuid01 = uuid.uuid1()
            if number == 0:
                old_file_path = file_path
                string2 = old_file_path[:old_file_path.rfind(end)]
            else:
                string = file_path[:file_path.rfind(end)]
                string2 = string[:string.rfind(end)]
            string3 = string2 + '/comparison_plot' + str(uuid01)
            try:
                os.mkdir(string2 + '/comparison_plot' + str(uuid01))
            except Exception as e:
                print(e)
            savepath = string3+'/'
            isstack = None
            try:

                info = comparison_plot(df_input=df,features=features,group=group,path=savepath,name_y=name_y,hue_=hue,kind=kind,concat_way=concat_way,row_size=int(row_size),col_size=int(col_size),isstack=isstack,palette_style=palette_style)
            except Exception as e:
                print(e)
                info = comparison_plot(df_input=df,features=features,group=group,path=savepath,name_y=name_y,hue_=hue,kind=kind,concat_way=concat_way,row_size=row_size,col_size=col_size,isstack=isstack,palette_style=palette_style)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        plts = info
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "comparison_plot", route01, str(uuid01))
        # plt_file02 = string2 + '/comparison_plot' + str(uuid01) + "/" + "plt.png"
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签/
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 
        # plt.savefig(plt_file02) 
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '比较图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                           order=order,
                                                           file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                       file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})       
        

class Pointlineplot(APIView):
    """点线图"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request, project_id):
        context = {
            'value': 'All',
            'project_id': project_id,
            "begin" : "chart",
            "name":'dxt'
        }
        return render(request, 'index/dxt.html', context=context)

    def post(self, request, project_id):
        # 接收json数据
        json_data = json.loads(request.body.decode())
        # list选择的特征名列表
        features = json_data.get('features')
        # str分组变量
        group = json_data.get('group')
        #str默认为None，仅concat_way=='free'时可用，第二个分组类型
        hue = json_data.get('hue')
        # str默认为box，作图方式，可选kind='bar','box','violin'
        kind = json_data.get('kind')
        row_size = json_data.get('row_size')
        col_size = json_data.get('col_size')
        # 获取哪种数据
        number = json_data.get('number')
        # 获取图片风格
        palette_style = json_data.get('palette_style')
        # 数据筛选
        filter_data = json_data.get("filter")

        # 判断数据是原始数据还是处理后的数据
        file_path = None
        number = int(number)
        if number == 0:
            # 查询数据库原始文件
            file = File_old.objects.get(id=project_id, user_id=request.user.id)
            file_path = file.path_pa
        elif number == 1:
            try:
                process = Browsing_process.objects.get(user_id=request.user.id, file_old_id=project_id, is_latest=1)
            except Exception as e:
                return http.JsonResponse({'code': 1002, 'error': '你还没有处理过数据，所有没有最新数据噢'})
            re_d = eval(process.process_info)
            file_path = re_d.get("df_result")
        try:
            # 读取文件
            # 查询该用户的用户等级
            grade = Member.objects.get(user_id=request.user.id)
            # 查询当前用户的等级
            member = MemberType.objects.get(id=grade.member_type_id)

            try:
                df_r = read(file_path)
                df = df_r.iloc[0:int(member.number)]
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '上传文件有误请重新上传'})
        if filter_data:
            df = filtering_view(df,json_data)
        # 分析成功保存图片
        end = '/'
        uuid01 = uuid.uuid1()
        if number == 0:
            old_file_path = file_path
            string2 = old_file_path[:old_file_path.rfind(end)]
        else:
            string = file_path[:file_path.rfind(end)]
            string2 = string[:string.rfind(end)]
        string3 = string2 + '/pointlineplot' + str(uuid01)
        try:
            os.mkdir(string2 + '/pointlineplot' + str(uuid01))
        except Exception as e:
            print(e)
        savepath = string3+'/'
        # 分析函数
        try:
            if hue == '':
                hue = None
            if row_size == '':
                row_size = None
            else:
                row_size = int(row_size)
            if col_size == '':
                col_size = None
            else:
                col_size = int(col_size)

            info = point_line_plot(df_input=df,features=features,group=group,path=savepath,hue_=hue,kind=kind,row_size=row_size,col_size=col_size,palette_style=palette_style)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '分析失败，请填写正确的变量'})
        if isinstance(info, str):
            return http.JsonResponse({'code': 1004, 'error': info})
        # 保存图片
        plts = info
        try:
            print(plts)
            if plts['error']:
                # print(3333333333333)
                return http.JsonResponse({'code': 1005, 'error': plts['error']})
        except Exception as e:
            print(e)
            print('正常')
        # # 设置图片大小
        end01 = "/static/"
        route01 = string2[string2.rfind(end01):]
        pltlist = loop_add(plts, "pointlineplot", route01, str(uuid01))
        
        # # 保存图片
        # plts = info
        # end01 = "/static/"
        # route01 = string2[string2.rfind(end01):]
        # route02 = route01 + '/pointlineplot' + str(uuid01)
        # # 展示到前端的路径
        # plt_file = route02 + "/" + "plt.png"
        # plt_file02 = string2 + '/pointlineplot' + str(uuid01) + "/" + "plt.png"
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 
        # plt.savefig(plt_file02) 
        # plt.close()

        # 将数据保存到数据库
        browsing_process = {}
        browsing_process['name'] = '点线图'
        browsing_process['plt'] = pltlist[0]
        # # 查询所有之前的顺序
        try:
            process = Browsing_process.objects.filter(user_id=request.user.id, file_old_id=project_id).aggregate(
                Max('order'))
            print(process)
            if process:
                order = int(process['order__max']) + 1
                print(order)
                browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id,
                                                          order=order,
                                                          file_old_id=project_id)
                project = browsing.id

        except Exception as e:
            print(e)
            browsing = Browsing_process.objects.create(process_info=browsing_process, user_id=request.user.id, order=1,
                                                      file_old_id=project_id)
            project = browsing.id
        data = {
            'project': project,
            'plt': pltlist[0]
        }
        return http.JsonResponse({'code': 200, 'error': '分析成功', 'data': data})       



