import json
import os
import re
import datetime
import shutil
from xmlrpc.client import DateTime
from dateutil.relativedelta import relativedelta
import pandas as pd
import datetime
from PIL import Image
from django import http
from django.contrib.auth import authenticate, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import login
from django.db import DatabaseError
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from django import http
from django_redis import get_redis_connection
from apps.index_qt.models import User, File_old, Member,Help,Analysis, LoginHistory
from libs.get import get_s, loop_add, write, read,member_check
from django import http
from django.core.paginator import Paginator, EmptyPage
from django.contrib.auth.hashers import make_password, check_password
from utils.views import LoginRequiredJSONMixin
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication
from rest_framework_jwt.settings import api_settings
from urllib.request import Request
from jsonschema.compat import urlopen
from utils.tools import Tool

class Community(APIView):

    # authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        
        """
        交流区页面
        :param request:
        :return:
        """
        print(1111111)
        print(request.user)
        return render(request, 'index/community.html')

class Index_qtView(APIView): # 用APIView试一下吧 我的全都是APIView。。。好，
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        主页
        :param request:
        :return:
        """
        print('主页打印用') # 数据库字段的类型更改 有些用户的微信名字有符号 
        print(request.user) # 这些都打印了 为什么你还要手动修改路由 你看你现在 扫码成功后 code=061EIKFa1gdyEA0onXFa1GnsdO3EIKFy&state=4e23bab8823811eb8fd000163e132610路由竟然还有这个code。。。
        data = []
        info = File_old.objects.filter(user_id=request.user.id).all()

        # 获取会员图标
        try:
            member_num = Member.objects.get(user=request.user.id)
        except Exception as e:
            print(e)
            # 清理session
            logout(request)
    
            # 退出登录，重定向到登录页
            response = redirect(reverse('index_qt:index_sy'))
    
            # 退出登录时清除cookie中的username
            response.delete_cookie('username')
    
            return response
        # 校验会员时间
        member_check(member_num)
        u_id = member_num.member_type.id
        # 循环所有的项目
        for num in info:
            i = {}
            i['id'] = num.id
            i['project_name'] = num.project_name
            i['background'] = num.background
            i['outline'] = num.outline
            i['file_name'] = num.file_name
            i['user_id'] = num.user
            i['create_time'] = str(num.create_time.year) + '-'+ str(num.create_time.month) + '-'+  str(num.create_time.day)
            i['last_time'] = str(num.last_time.year) + '-'+ str(num.last_time.month) + '-'+  str(num.last_time.day)
            data.append(i)
        context = {
            'data': data,
            'u_id': u_id,
        }

        return render(request, 'index/index.html', context=context)
        
        
class Index_sy(APIView):

    # authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        
        """
        网站首页
        :param request:
        :return:
        """
        print(1111111)
        print(request.user)
        return render(request, 'index/new_index.html')
        

class Video_course(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        
        """
        视频教程
        :param request:
        :return:
        """
        # 获取会员图标
        try:
            member_num = Member.objects.get(user=request.user.id)
        except Exception as e:
            print(e)
            # 清理session
            logout(request)
    
            # 退出登录，重定向到登录页
            response = redirect(reverse('index_qt:index_sy'))
    
            # 退出登录时清除cookie中的username
            response.delete_cookie('username')
    
            return response
        # 校验会员时间
        member_check(member_num)
        u_id = member_num.member_type.id
        context = {
            'u_id': u_id,
        }

        return render(request, 'index/video_course.html',context=context)
    
    def post(self,request):
        """返回视频教程的数据"""
        # 获取参数
        json_data = json.loads(request.body.decode())
        # 获取当前是第几页
        page_num = json_data.get('page')
        # 查询视频返回
        # 查询所有的模型
        modellist = Help.objects.filter().all()
        length = len(modellist)
        # 创建分页器：每页N条记录
        paginator = Paginator(modellist, 8)
        # 获取每页视频数据
        try:
            page_books = paginator.page(page_num)
        except EmptyPage:
            # 如果page_num不正确，默认给用户404
            return http.JsonResponse({'code': 201, 'error': '视频暂未更新'})
        # 获取列表页总页数
        total_page = paginator.num_pages
        # 循环每个模型信息
        info = []
        for i in page_books:
            # 查询每个视频的相关信息
            info.append({'ID':i.id,'video_name':i.video_name,'video_background':i.video_background,'video_info':i.video_info,'video_link':i.video_link})
        context = {
            'length': length,
            'info':info,
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return http.JsonResponse({'code': 0, 'error': '查询成功','context':context})
    
class Index_f(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        主页详情
        :param request:
        :return:
        """
        

        return render(request, 'index/miss_delete.html')


class Login(APIView):
    def get(self, request):
        """
        登录页面
        :param request:
        :return:
        """
        return render(request, 'index/login_0218.html')

    def post(self, request):
        """
        实现登录逻辑
        :param request: 请求对象
        :return: 登录结果
        """
        # 接收参数
        
        json_data = json.loads(request.body.decode())
        username = json_data.get('username')   
        password = json_data.get('password')
        # remembered = request.POST.get('remembered')
        print(username, password, "-=-=-=-==")
        # 校验参数
        # 判断参数是否齐全
        if not all([username, password]):
            return http.JsonResponse({'code': 1002, 'errmsg': '用户名密码不能为空'}) 
            # return render(request, 'index/login_0218.html', {'account_errmsg': '缺少必传参数'})

        # # 判断用户名是否是5-20个字符
        # if not re.match(r'^[a-zA-Z0-9_-]{5,20}$', username):
        #     # return http.JsonResponse({'code': 1003, 'error': '请输入正确的用户名或手机号'}) 
        #     # return http.HttpResponseBadRequest('请输入正确的用户名或手机号')
        #     return render(request, 'index/login_0218.html', {'account_errmsg': '请输入正确的用户名或手机号'})

        # 判断密码是否是8-20个数字
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password):
            # return http.JsonResponse({'code': 1004, 'error': '密码最少8位，最长20位'}) 
            # return http.HttpResponseBadRequest('密码最少8位，最长20位')
            return http.JsonResponse({'code': 504, 'errmsg': '密码最少8位，最长20位'})
            # return render(request, 'index/login_0218.html', {'account_errmsg': '密码最少8位，最长20位'})

        # 认证登录用户
        try:
            try:
                user = authenticate(username=username, password=password)
            except Exception as e:
                print(e)
                user = authenticate(mobile=username, password=password)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 503, 'errmsg': '用户名或密码错误'})
            # return render(request, 'index/login_0218.html', {'account_errmsg': '用户名或密码错误'})
        if user is None :
            # return http.JsonResponse({'code': 1005, 'error': '用户名或密码错误'}) 
            # return http.HttpResponseBadRequest('用户名或密码错误')
            return http.JsonResponse({'code': 502, 'errmsg': '用户名或密码错误'})
            # return render(request, 'index/login_0218.html', {'account_errmsg': '用户名或密码错误'})
        # 查询当前用户是否能使用
        grade = Member.objects.get(user_id=user.id)
        if (int(grade.is_delete) == 1 or grade.user.is_active==False):
            # return http.JsonResponse({'code': 1006, 'error': '当前用户已经不能使用'}) 
            # return http.HttpResponseBadRequest('当前用户已经不能使用')
            return http.JsonResponse({'code': 501, 'errmsg': '当前用户已经不能使用'})
            # return render(request, 'index/login_0218.html', {'account_errmsg': '当前用户已经不能使用'})
        # # 实现状态保持
        time_current_01 = datetime.datetime.now().strftime('%Y-%m-%d')
        time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("===================时间==================")
        print(time_current)
        # print(time_current_01)
        analysis_cout =Analysis.objects.filter(analysis_time__contains=time_current_01).count()
        #当日是否重复登录 >0 代表登录过了
        isRepeatLogin=User.objects.filter(Q(last_login__contains=time_current_01)&Q(id=user.id)).count()
        if analysis_cout == 0:
            sheet = Analysis.objects.create(analysis_time = time_current,login_total=1,real_number=1)
            sheet.save()
        else:
            sheet_count = Analysis.objects.get(analysis_time__contains=time_current_01)
            # sheet_count = sheet_count[0]
            num = sheet_count.login_total
            sheet_count.login_total = num+1
            if isRepeatLogin==0:  #当日未登录过 真实人数可以加1
                sheet_count.real_number+=1
            sheet_count.save()
        login_count =  LoginHistory.objects.create(login_time = time_current,user_id=user.id)
        login_count.save()
        #更新登录时间
        updateUser=User.objects.get(id=user.id)
        updateUser.last_login=time_current
        updateUser.save()
        # 微信扫码登录后 jwt_token的返回 当然 你们没有限制 可以删除掉
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
        payload = jwt_payload_handler(user)
        token = jwt_encode_handler(payload)
        # 响应登录结果
        # return redirect(reverse('index:index'))
        # 响应注册结果
        # response = redirect(reverse('index:index'))
        # 响应登录结果
        next = request.GET.get('next')
        if next:
            response = redirect(next)
        else:
            response = redirect(reverse('index_qt:index_qt'))
            
        

        # 登录时用户名写入到cookie，有效期15天
        name = json.dumps(user.username)
        # print(name)
        response.set_cookie('username', name, max_age=3600 * 24 * 15)
        response.set_cookie('token', token, max_age=3600 * 24 * 15)
        response.set_cookie('uuid', str(user.id), max_age=3600 * 24 * 15)
        response.set_cookie('avatar', str(user.image_tou), max_age=3600 * 24 * 15)
        return response
        # return http.JsonResponse({'code': 200, 'errmsg': '登录成功',})

class RegisterView(APIView):
    def get(self, request):
        """
        注册页面
        :param request:
        :return:
        """

        return render(request, 'index/register.html')

    def post(self, request):
        json_data = json.loads(request.body.decode())
    
        username = json_data.get('username')
        password = json_data.get('password')
        password2 = json_data.get('password2')
        mobile = json_data.get('mobile')
        image_code_client = json_data.get('image_code')
        print(image_code_client)
        uuid = json_data.get('uuid')
        # print(uuid)
        sms_code = json_data.get('sms')
        # end= 'image_codes/'
        # string2 = uuid[uuid.rfind(end):]
        # end2 = '/'
        # string3 = string2[:string2.rfind(end2)]
        # string4 = string3.split('/')
        # uuid = string4[1]
        
        # 2. 校验参数
        # 判断参数是否齐全
        if not all([username, password, password2, mobile]):
            return http.JsonResponse({'code': 1002, 'error': '缺少必传参数'}) 

        # 判断用户名是否是5-20个字符
        if not re.match(r'^[a-zA-Z0-9_]{5,20}$', username):
            return http.JsonResponse({'code': 1003, 'error': '请输入5-20个字符的用户名'}) 

        # 判断密码是否是8-20个数字
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password):
            return http.JsonResponse({'code': 1004, 'error': '请输入8-20位的密码'}) 

        # 判断两次密码是否一致
        if password != password2:
            return http.JsonResponse({'code': 1005, 'error': '两次输入的密码不一致'}) 

        # 判断手机号是否合法
        if not re.match(r'^1[3-9]\d{9}$', mobile):
            return http.JsonResponse({'code': 1006, 'error': '请输入正确的手机号码'}) 

        # 连接redis校验图片验证码
        # 创建连接到redis的对象
        redis_conn = get_redis_connection('code')
        # 提取图形验证码
        image_code_server = redis_conn.get('img_%s' % uuid)
        if image_code_server is None:
            # 图形验证码过期或者不存在
            return http.JsonResponse({'code': 1007, 'error': '图片验证码失效，请刷新后输入'}) 
        # 删除图形验证码，避免恶意测试图形验证码
        try:
            redis_conn.delete('img_%s' % uuid)
        except Exception as e:
            print(e)
        # 对比图形验证码
        image_code_server = image_code_server.decode()  # bytes转字符串
        if image_code_client.lower() != image_code_server.lower():  # 转小写后比较
            return http.JsonResponse({'code': 1008, 'error': '输入图形验证码有误'}) 
            # return render(request, 'index/register.html', {'register_errmsg': '输入图形验证码有误'})

        sms_code_saved = redis_conn.get('sms_%s' % mobile)
        if sms_code_saved is None:
            return http.JsonResponse({'code': 1009, 'error': '无效的短信验证码'}) 
        if sms_code != sms_code_saved.decode():
            return http.JsonResponse({'code': 1010, 'error': '输入短信验证码有误'}) 
        # 3. 创建用户
        # 保存注册数据
        try:
            user = User.objects.create_user(username=username, password=password, mobile=mobile, dft_file='',image_tou='user_pic.png')
            user.save()
            grade = Member.objects.create(user_id=user.id,member_type_id=1)
            grade.save()
        except DatabaseError as e:
            print(e)
            return http.JsonResponse({'code': 1009, 'error': '注册失败'}) 
        
        time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_current_01 = datetime.datetime.now().strftime('%Y-%m-%d')
        analysis_cout =Analysis.objects.filter(analysis_time__contains=time_current_01).count()
        if analysis_cout == 0:
            sheet = Analysis.objects.create(analysis_time = time_current,login_total=1,total_total=1)
            sheet.save()
        else:
            sheet_count = Analysis.objects.get(analysis_time__contains=time_current_01)
            # sheet_count = sheet_count[0]
            num = sheet_count.login_total
            sheet_count.login_total = num+1
            num01 = sheet_count.total_total
            sheet_count.total_total = num01+1
            sheet_count.save()
        # 4. 响应注册结果
        # response = redirect(reverse('index_qt:index_qt'))
        # # 注册时用户名写入到cookie，有效期15天
        # response.set_cookie('username', user.username, max_age=3600 * 24 * 15)

        # return response
        return http.JsonResponse({'code': 200 ,'error': '注册成功'}) 


class Index_user(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        用户中心
        :param request:
        :return:
        """
        # if str(request.user) == 'AnonymousUser':
        #     return render(request, 'index/login.html')
        # 获取当前用户
        user = request.user
        # 取出 当前用户图片路径
        avatar_url = user.image_tou
        # print(name, "=================")
        context = {
            "user_s": user.username,
            "avatar_url": avatar_url
        }
        # 状态保持
        # login(request, user)
        return render(request, 'index/user.html', context=context)

        # return render(request, 'index/user.html')


class LogoutView(APIView):
    def get(self, request):
        """实现退出登录逻辑"""
        # 清理session
        logout(request)

        # 退出登录，重定向到登录页
        response = redirect(reverse('index_qt:index_sy'))

        # 退出登录时清除cookie中的username
        response.delete_cookie('username')
        response.delete_cookie('uuid')
        response.delete_cookie('token')

        return response


class FileView(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        用户中心上传文件
        :param request:
        :return:
        """
        # 获取信息
        list01 = []
        info = File_old.objects.filter(user_id=request.user.id).all()
        for i in info:
            data = {}
            data['name'] = i.file_name
            data['time'] = i.create_time.date()
            list01.append(data)

        context = {
            'info': list01
        }
        return render(request, 'index/1111.html', context=context)


class MobileCountView(APIView):
    """判断手机号是否重复注册"""

    def get(self, request, mobile):
        """
        :param request: 请求对象
        :param mobile: 手机号
        :return: JSON
        """
        count = User.objects.filter(mobile=mobile).count()
        return http.JsonResponse({'code': 200, 'errmsg': 'OK', 'count': count})


class UsernameCountView(APIView):
    """判断用户名是否重复注册"""

    def get(self, request, username):
        """
        :param request: 请求对象
        :param username: 用户名
        :return: JSON
        """
        count = User.objects.filter(username=username).count()
        return http.JsonResponse({'code': 200, 'errmsg': 'OK', 'count': count})


class Project_delete(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request, project_id):
        # 校验参数
        if not all([project_id]):
            return http.JsonResponse({'code': 1001, 'error': '缺少参数'})
        # 查询相关项目
        file = File_old.objects.filter(id=project_id)
        path = file[0].path
        try:
            end = '/'
            path01 = path[:path.rfind(end)]
            shutil.rmtree(path01)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '网络异常，删除失败'})
        file.delete()

        return http.JsonResponse({'code': 200, 'error': '删除成功'})

class User_pic(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        修改用户头像
        :param request:
        :return:
        """
        user = User.objects.get(username=(request.user.username))
        # 状态保持
        # login(request, user)
        avatar_url = user.image_tou
        # print(user, "=================")
        context = {
            "avatar_url": avatar_url
        }
        return render(request, "index/user_pic.html", context=context)
        # return render(request, 'index/user_pic.html')

    def post(self, request):
        f = request.FILES.get("avatar")
        print(f, "=============")
        img = Image.open(f)
        # print(f.name)
        img_name = f.name
        # print(img_name, "===============")

        img.save('/www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent/static/upload/user_images/%s' % img_name)
        # url = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".") + "\\static\\upload\\user_images\\a.jpg"
        url = img_name
        user = request.user
        user.image_tou = url
        user.save()
        # print(user.image_tou)
        return http.JsonResponse({"errno": 200, "errmsg": "ok"})


class User_pass(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        修改用户密码
        :param request:
        :return:
        """
        # 状态保持
        # login(request, user)

        return render(request, 'index/user_pass.html')
        # return render(request, "index/user_pass_info.html", context=context)

    def post(self, request):
        json_dict = json.loads(request.body.decode())
        old_password = json_dict.get('old_password')
        new_password = json_dict.get('new_password')
        print(old_password, new_password)
        if not all([old_password, new_password]):
            return http.JsonResponse({"errno": 1, "errmsg": "缺少参数"})
        # 从cookie中获取当前用户
        # name = request.COOKIES.get("username")
        # 3.检验旧密码是否正确
        if not request.user.check_password(old_password):
            return http.JsonResponse({"errno": 1, "errmsg": "原密码错误"})
        try:
            print(request.user, "==============")
            request.user.set_password(new_password)
            request.user.save()
        except Exception as e:
            return http.JsonResponse({"errno": 0, "errmsg": "失败"})

        return http.JsonResponse({"errno": 0, "errmsg": "ok"})



class Project_edit(APIView):
    
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request, project_id):
        """
        主页编辑显示
        :param request:
        :return:
        """
        # 校验参数
        if not all([project_id]):
            return http.JsonResponse({'code': 1001, 'error': '缺少参数'})
        # 查询相关项目
        file = File_old.objects.get(id=project_id)
        # 查询项目数据表
        data = {}
        # 循环所有的项目
        data['id'] = file.id
        data['name'] = file.project_name
        data['background'] = file.background
        data['outline'] = file.outline
        data['file_name'] = file.file_name
        data['user_id'] = file.user.id
        data['project'] = project_id
        return HttpResponse(json.dumps({'code': '200', 'error': '分析成功', 'context': data}))

class DataEdit(APIView):
    """编辑文件的类"""
    
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request, project_id):
        global Filename
        """
        接收上传的文件
        :param request:
        :return:
        """
        # 获取参数
        a = request.FILES.get('upfile')
        project_name = request.POST.get('name')
        project_background = request.POST.get('background')
        project_outline = request.POST.get('outline')
        # 校验参数
        # 有文件保存到数据库
        file = File_old.objects.get(id=project_id)
        old_file_path = file.path_pa
        old_file_path02 = file.path
        filename = file.file_name
        # # 编辑修改文件
        if a:
            filename = a.name
            m = re.findall(r'\..+', filename)
            # 判断文件名后缀是否被允许打开
            print(m[0])
            if m[0] == '.csv' or m[0] == '.xlsx' or m[0] == '.xls':
                os.remove(old_file_path)
                print(old_file_path02)
                os.remove(old_file_path02)
                print('成功删除文件:', old_file_path)
                print('成功删除文件:', old_file_path02)
                end='/'
                string2 = old_file_path[:old_file_path.rfind(end)]
                # 保存文件
                old_file_path = string2+'/'+filename
                with open(string2+'/'+filename, 'wb+') as f:
                    for chunk in a.chunks():
                        f.write(chunk)
                # 读取文件
                try:
                    try:
                        f = pd.read_excel(old_file_path)
                    except Exception as e:
                        print(e)
                        f = pd.read_csv(old_file_path, encoding='GBK')
                except Exception as e:
                    print(e)
                    f = pd.read_csv(old_file_path)
                a = f.columns
                list01 = []
                try:
                    for i in a:
                        c = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a,])", "", str(i))
                        if c[0:1].isnumeric():
                            c = "_" + c
                        if c not in list01:
                            list01.append(c)
                        else:
                            return JsonResponse({'code': 10040, 'error': '表格数据内有重复列名，请修改后上传'})
                except Exception as e:
                    print(e)
                    return JsonResponse({'code': 10030, 'error': '表格数据有非法字符'})
                c = filename.split('.')
                path_pa = string2+'/' + str(c[0]) + '.pkl'
                write(f, path_pa)
                file.path_pa = path_pa
        print(old_file_path)
        print(filename)
        file.project_name = project_name
        file.background = project_background
        file.outline = project_outline
        file.path = old_file_path
        file.file_name = filename
        file.save()
        return http.JsonResponse({'code': '200', 'error': '修改成功'})

class Security(APIView):
    def get(self, request):
        """
        安全性
        :param request:
        :return:
        """

        return render(request, 'index/security.html')
class Legal_aprovisions(APIView):
    def get(self, request):
        """
        
        :param request:
        :return:
        """

        return render(request, 'index/legal_provisions.html')
class Privacy(APIView):
    def get(self, request):
        """
        隐私
        :param request:
        :return:
        """

        return render(request, 'index/privacy.html')
class About(APIView):
    def get(self, request):
        """
        关于我们
        :param request:
        :return:
        """

        return render(request, 'index/about.html')

# class Forget(APIView):
#     def get(self, request):
#         """
#         忘记密码
#         :param request:
#         :return:
#         """

#         return render(request, 'index/forget.html')
    
#     def post(self, request):
#         """"忘记密码"""
#         print(request.body)
#         json_data = json.loads(request.body.decode())
#         username = json_data.get('username')
#         phone = json_data.get('mobile')
#         sms_code = json_data.get('sms')
#         print(username, phone,sms_code)
#         if not all([username, phone]):
#             return http.JsonResponse({"code": 501, "errmsg": "用户名手机号不能为空"})
#         if not all([sms_code]):
#             return http.JsonResponse({"code": 502, "errmsg": "短信验证码不能为空"})
#         # 从cookie中获取当前用户
#         # 创建连接到redis的对象
#         redis_conn = get_redis_connection('code')
#         sms_code_saved = redis_conn.get('sms_%s' % phone)
#         if sms_code_saved is None:
#             return http.JsonResponse({'code': 1009, 'error': '无效的短信验证码'}) 
#         if sms_code != sms_code_saved.decode():
#             return http.JsonResponse({'code': 1010, 'error': '输入短信验证码有误'}) 
#         try:
#             count = User.objects.filter(username=username,mobile=phone).count()
            
#         except Exception as e:
#             return http.JsonResponse({"code": 0, "errmsg": "用户不存在"})

#         return http.JsonResponse({'code': 200, 'errmsg': 'OK', 'count': count})

class ForgetPassword(APIView):
    """忘记密码修改密码"""
    
    def post(self, request):
        """"忘记密码"""
        print(request.body)
        json_data = json.loads(request.body.decode())
        password = json_data.get('password')
        sms_code = json_data.get('sms')
        phone = json_data.get('mobile')
        # print(username, phone)
        if not all([phone,password,sms_code]):
            return http.JsonResponse({"code": 1, "errmsg": "密码不能输入为空"})
        # 从cookie中获取当前用户
        redis_conn = get_redis_connection('code')
        sms_code_saved = redis_conn.get('sms_%s' % phone)
        if sms_code_saved is None:
            return http.JsonResponse({'code': 1009, 'error': '无效的短信验证码'}) 
        if sms_code != sms_code_saved.decode():
            return http.JsonResponse({'code': 1010, 'error': '输入短信验证码有误'}) 
        # try:
        #     count = User.objects.filter(mobile=phone).count()
            
        # except Exception as e:
        #     return http.JsonResponse({"code": 0, "errmsg": "用户不存在"})
        try:
            user = User.objects.get(mobile=phone)
            
        except Exception as e:
            return http.JsonResponse({"code": 0, "errmsg": "用户不存在"})
        try:
            new_password = make_password(password)
            user.password = new_password
            user.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({"code": 0, "errmsg": "修改失败"})

        return http.JsonResponse({'code': 200, 'errmsg': '修改成功'})
    
def  delFile():
    three_months_ago = (datetime.datetime.now() + relativedelta(months=-3)).strftime('%Y-%m-%d %H:%M:%S')
    sql = "select * from (select id,user_id,path,path_pa from tb_file)as tb_file  join (select users.id from(select id from tb_users where " \
          "(last_login is NULL and date_joined <'" + three_months_ago + "') or last_login<'" + three_months_ago + " ')" \
          "As users JOIN (select user_id from tb_member where member_type_id=1 )As memberv0  on users.id=memberv0.user_id) As ids on tb_file.user_id=ids.id"
    print(sql)
    data = Tool.getData(sql, None)
    # element[0] project_id
    # element[1] user_id
    # element[2] path /www/wwwroot/znfxpt.8dfish.vip/ini/Intelligent/static/exel/user34_大气680644f0-9e49-11eb-ab3f-00163e132610/大气_单次_20201227_2_.xlsx
    # element[3] path_pa /www/wwwroot/znfxpt.8dfish.vip/ini/Intelligent/static/exel/user34_大气680644f0-9e49-11eb-ab3f-00163e132610/大气_单次_20201227_2_.pkl
    for element in data:
        path = element[2]
        # 获取删除目录
        pathDel = path[0:path.rindex("/")]
        print(pathDel)
        try:
            shutil.rmtree(pathDel)
            print(pathDel + "目录已删除")
            # 删除tb_file 表中项目project记录
        except FileNotFoundError as e:
            print(pathDel + "目录未找到,无法删除")
        try:
            print("开始删除"+str(element[0])+"的项目id")
            File_old.objects.filter(id=element[0]).delete()
            print("成功删除"+str(element[0])+"的项目id")
        except FileNotFoundError as e:
            print(e + "项目"+str(element[0])+"删除失败")    
            
            
        