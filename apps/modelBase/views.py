import json
import uuid
import time
import os
from django.shortcuts import render
from django.views import View
from django.db.models import Q
from apps.index_qt.models import Member, User, MemberType, Browsing_process, Modelbase
from django import http
from django.core.paginator import Paginator, EmptyPage
from datetime import datetime
from numpy import *
from libs.get import get_path

class ModelIndex(View):
    """给用户展示的模型页面"""
    def get(self, request):
        # 查询所有的模型
        modellist = Modelbase.objects.filter().all()
        # 循环每个模型信息
        info = []
        for i in modellist:
            # 查询每个模型的相关信息
            info.append({'ID':i.id,'username':i.user.username,'model_name':i.model_name,'model_create_time':i.create_time,'model_type':i.model_type,'model_last_time':i.last_time,'model_background':i.model_background,'people':i.number})
        context = {
            'info':info,
        }

        
        return render(request, 'index/model_index.html',context=context)


class MemberList(View):
    """模型列表的显示 """
    def get(self, request):
        """
        模型列表页面
        :param request:
        :return:
        """
        if str(request.user) == 'AnonymousUser':
            return render(request, 'admin/login.html')
        # 查询该人是否是管理员
        user = request.user
        try:
            if not user.super_u == 1:
                return render(request, 'admin/login.html')
        except:
            return render(request, 'admin/login.html')
        return render(request, 'index/model_list.html')

        
        

class ModelUpdate(View):
    """管理员上传模型"""
    def post(self, request):
        # 获取参数
        a = request.FILES.get('model_upfile')
        model_name = request.POST.get('model_name')
        model_background = request.POST.get('background')
        model_outline = request.POST.get('outline')
        model_imgs = request.POST.get('model_imgs')
        model_type = request.POST.get('model_type')
        number = request.POST.get('number')
        versions = request.POST.get('versions')
        url = request.POST.get('url')
        # 校验参数
        if not model_name:
            return http.JsonResponse({'code': 1001, 'error': '请填写模型名称'})
        if not model_background:
            return http.JsonResponse({'code': 1002, 'error': '请填写模型说明'})
        if not model_outline:
            return http.JsonResponse({'code': 1003, 'error': '请填写模型方法'})
        # 查询该人是否是管理员
        user = request.user
        try:
            if not user.super_u == 1:
                return http.JsonResponse({'code': 1004, 'error': '非管理员用户'})
        except:
            return http.JsonResponse({'code': 1004, 'error': '非管理员用户'})
        project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
        # 判断是否上传了文件
        path = ''
        path_img = ''
        # 用户项目文件夹
        x = uuid.uuid1()
        # print(x)
        user_file_path =  str(request.user.id) + '_' + model_name + str(x) + '/'
        os.makedirs(str(project_path)+'/static' + '/model/'  + str(request.user.id) + '_' + model_name + str(x))
        try:
            if a:
                # 获取上传文件的名字
                filename = a.name
                path = str(project_path)+'/static' + '/model/' + user_file_path + filename
                with open(path, 'wb+') as f:
                    for chunk in a.chunks():
                        f.write(chunk)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '保存文件失败'})

        # 将文件名字保存到数据库
        try:
            user = request.user
            last_time = str(datetime.now())
            Modelbase.objects.create(model_name=model_name, user_id=user.id,model_background=model_background, model_outline=model_outline, model_path=path, last_time=last_time,model_info=model_imgs,model_type = model_type,versions = versions,url=url,number=number)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1010, 'error': '保存数据库失败'})
        # 返回页面
        return http.JsonResponse({'code': '200', 'error': '上传模型成功'})

class ModelDelete(View):
    def post(self, request):
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取要获取平均分的模型
        model_id = json_data.get('ID')
        
        try:
            modellist = Modelbase.objects.get(id=model_id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '查找不到该模型'})
        
        img = modellist.model_info.split(',/static/')
        for i in img:
            if i == '':
                continue
            try:
                print(11111111)
                project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
                if 'static' not in i:
                    i =  '/static/'+i
                print(str(project_path)+i)
            except Exception as e:
                print(e)
        print(img)
        
        modellist.delete()

        return http.JsonResponse({'code': 200, 'error': '删除成功'})
        

class ModelViews(View):
    """ 详情页面"""
    def get(self,request):

        return render(request, 'index/model_details.html')
    
    def post(self,request):
        # 获取参数
        json_data = json.loads(request.body.decode())
        id = json_data.get("ID")
        modeldetails = Modelbase.objects.get(id = int(id))
        # 获取信息
        name = modeldetails.model_name
        username = modeldetails.user.username
        user_id = modeldetails.user.id
        background = modeldetails.model_background
        model_outline = modeldetails.model_outline
        model_info = modeldetails.model_info
        model_type = modeldetails.model_type
        versions = modeldetails.versions
        url = modeldetails.url
        context = {
            'name':name,
            'username':username,
            'user_id':user_id,
            'background':background,
            'model_outline':model_outline,
            'model_info':model_info,
            'model_type':model_type,
            'versions':versions,
            'url':url
        }
        return http.JsonResponse({'code': '200', 'error': '页面详情','context':context})
            

class ModelEdit(View):
    def post(self, request):
        # 获取参数
        model_id = request.POST.get('id')
        a = request.FILES.get('model_upfile')
        model_name = request.POST.get('model_name')
        model_background = request.POST.get('background')
        model_outline = request.POST.get('outline')
        model_imgs = request.POST.get('model_imgs')
        model_type = request.POST.get('model_type')
        number = request.POST.get('number')
        versions = request.POST.get('versions')
        url = request.POST.get('url')
        # 校验参数
        if not model_name:
            return http.JsonResponse({'code': 1001, 'error': '请填写模型名称'})
        if not model_background:
            return http.JsonResponse({'code': 1002, 'error': '请填写模型说明'})
        if not model_outline:
            return http.JsonResponse({'code': 1003, 'error': '请填写模型方法'})
        # 查询该人是否是管理员
        user = request.user
        try:
            if not user.super_u == 1:
                return http.JsonResponse({'code': 1004, 'error': '非管理员用户'})
        except:
            return http.JsonResponse({'code': 1004, 'error': '非管理员用户'})
        # 查找模型
        try:
            modellist = Modelbase.objects.get(id=model_id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '查找不到该模型'})
        # 判断是否上传了文件
        if a:
            path = ''
        else:
            path = modellist.model_path
        path_img = ''
        # 用户项目文件夹
        x = uuid.uuid1()
        project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
        # print(x)
        user_file_path =  str(request.user.id) + '_' + model_name + str(x) + '/'
        os.makedirs(str(project_path)+'/static' + '/model/'  + str(request.user.id) + '_' + model_name + str(x))
        try:
            if a:
                # 获取上传文件的名字
                filename = a.name
                path = str(project_path)+'/static' + '/model/' + user_file_path + filename
                with open(path, 'wb+') as f:
                    for chunk in a.chunks():
                        f.write(chunk)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '修改文件失败'})
        try:
            modellist.model_name = model_name
            modellist.model_background = model_background
            modellist.model_outline = model_outline
            modellist.model_type = model_type
            modellist.model_path = path
            modellist.model_info = model_imgs
            modellist.versions = versions
            modellist.url = url
            modellist.number = number
            modellist.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '修改保存失败'})
        return http.JsonResponse({'code': '200', 'error': '修改成功'})
        
        
        
        
class ModelData(View):
    def get(self, request):
                # 获取参数
        page_num = request.GET.get('page')

        # 查询所有的模型
        modellist = Modelbase.objects.filter().all()
        length = len(modellist)
        # 创建分页器：每页N条记录
        paginator = Paginator(modellist, 10)
        # 获取每页会员数据
        try:
            page_books = paginator.page(page_num)
        except EmptyPage:
            # 如果page_num不正确，默认给用户404
            return http.HttpResponseNotFound('empty page')
        # 获取列表页总页数
        total_page = paginator.num_pages
        # 循环每个模型信息
        info = []
        for i in page_books:
            # 查询每个模型的相关信息
            # /static/model/163_ad6a0a64-b400-11eb-93fd-00163e132610/location_1.png,/static/model/163_aeb2b9d4-b400-11eb-a530-00163e132610/location_4.png
            info.append({'ID':i.id,'username':i.user.username,'model_name':i.model_name,'model_create_time':i.create_time,'model_type':i.model_type,'model_last_time':i.last_time,'versions':i.versions,'number':i.number,'model_background':i.model_background,'model_outline':i.model_outline,'img':i.model_info,'file':i.model_path,'model_type':i.model_type,'url':i.url})
        context = {
            'length': length,
            'info':info,
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return http.JsonResponse({'code': 0, 'error': '查询成功','context':context})
            
            
class ModelScoreAverage(View):
    """评价计算评分"""
    def post(self, request):
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取要获取平均分的模型
        model_id = json_data.get('ID')
        
        # 校验参数
        if not model_id:
            return http.JsonResponse({'code': 1010, 'error': '非法访问'})
        
        # 转换评分输入
        try:
            model_id = int(model_id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1020, 'error': '传入非法字符'})
        
        # 查询该模型库的所有评分
        source = Score.objects.filter(modelbase = model_id).all()
        list_average = []
        for i in source:
            list_average.append(i.score)
        
        # 计算平均数
        b = mean(a)
        
        # 将平均分写入数据库中
        
        return http.JsonResponse({'code': 200, 'error': '评价成功'})

class UpdateIMG(View):
    def post(self,request):
        """
        修改用户头像
        :param request:
        :return:
        """
        img = request.FILES.get('img')
        
        project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
        # 判断是否上传了文件
        path = ''
        # 用户项目文件夹
        x = time.time()
        # print(x)
        user_file_path =  str(request.user.id) + '_'  + str(x) + '/'
        os.makedirs(str(project_path)+'/static' + '/model/'  + str(request.user.id) + '_'  + str(x))
        try:
            if img:
                # 获取上传文件的名字
                filename = img.name
                path = str(project_path)+'/static' + '/model/' + user_file_path + filename
                with open(path, 'wb+') as f:
                    for chunk in img.chunks():
                        f.write(chunk)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '保存文件失败'})
        data = '/static' + '/model/' + user_file_path + filename
        
        return http.JsonResponse({'code': 200, 'error': '上传成功','context':data})

class DeleteIMG(View):
    def post(self,request):
        """删除图片"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取要获取平均分的模型
        model_id = json_data.get('path')
        project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
        try:
            path = str(project_path)+model_id
            os.remove(path)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1005, 'error': '删除图片失败'})
        
        return http.JsonResponse({'code': 200, 'error': '删除图片成功'})
        
        
        
            
            
