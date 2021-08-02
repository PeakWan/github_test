import json
import re
import datetime
from rest_framework.views import APIView
from django import http
from django.contrib.auth import authenticate, login
from django.contrib.auth.views import logout
from django.db import DatabaseError
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from apps.index_qt.models import Member, User, MemberType, Browsing_process,File_old

class IndexView(View):
    def get(self, request):
        """
        主页
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
        return render(request, 'admin/index.html')


class WeCome(View):
    def get(self, request):
        try:
            if str(request.user) == 'AnonymousUser':
                return render(request, 'admin/login.html')
            user = request.user
            if not user.super_u == 1: # 查询该人是否是管理员
                return render(request, 'admin/login.html')
            memberTotalNum = Member.objects.filter(is_delete=0).all().count()
            addMemberNum = User.objects.filter(date_joined__contains = datetime.datetime.now().strftime('%Y-%m-%d')).all().count()
            projectTotalNum=File_old.objects.all().count()
        except Exception as e:
            print(e)
        context = {
            'memberTotalNum':memberTotalNum,
            'addMemberNum':addMemberNum,
            'projectTotalNum':projectTotalNum
        }
        return render(request, 'admin/welcome.html',context=context)


class Login(View):
    def get(self, request):
        """
        后台登录页面
        登录页面
        :param request:
        :return:
        """

        return render(request, 'admin/login.html')

    def post(self, request):
        """
        实现登录逻辑
        :param request: 请求对象
        :return: 登录结果
        """
        # 接收参数
        username = request.POST.get('name')
        password = request.POST.get('password')
        # remembered = request.POST.get('remembered')

        # 校验参数
        # 判断参数是否齐全
        if not all([username, password]):
            return http.HttpResponseBadRequest('缺少必传参数')

        # 判断用户名是否是5-20个字符
        if not re.match(r'^[a-zA-Z0-9_-]{5,20}$', username):
            # return http.HttpResponseBadRequest('请输入正确的用户名或手机号')
            return render(request, 'admin/login.html', {'account_errmsg': '请输入正确的用户名或手机号'})

        # 判断密码是否是8-20个数字
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password):
            # return http.HttpResponseBadRequest('密码最少8位，最长20位')
            return render(request, 'admin/login.html', {'account_errmsg': '密码最少8位，最长20位'})

        # 认证登录用户
        user = authenticate(username=username, password=password)
        try:
            num = int(user.super_u)
        except Exception as e:
            print(e)
            return render(request, 'admin/login.html', {'account_errmsg': e})
        if user is None and num == 0:
            return render(request, 'admin/login.html', {'account_errmsg': '用户名或密码错误'})
         # 查询当前用户是否能使用

        # 实现状态保持
        login(request, user)
        # 没有记住用户：浏览器会话结束就过期
        request.session.set_expiry(0)
        next = request.GET.get('next')
        if next:
            response = redirect(next)
        else:
            response = redirect(reverse('index:index'))
        
        # 注册时用户名写入到cookie，有效期15天
        response.set_cookie('username', user.username, max_age=3600 * 24 * 15)

        return response


class LogoutView(View):
    def get(self, request):
        """实现退出登录逻辑"""
        # 清理session
        logout(request)
        # 退出登录，重定向到登录页
        response = render(request, 'admin/login.html')

        # 退出登录时清除cookie中的username
        response.delete_cookie('username')
        return response

