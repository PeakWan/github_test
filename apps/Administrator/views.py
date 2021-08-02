from django.shortcuts import render
from django.views import View
from django.core.paginator import Paginator, EmptyPage
from apps.index_qt.models import User
import json
from django import http
from apps.index_qt.models import User
import re

class Administrator_list(View):
    def get(self, request,page_num):
        """
        管理员列表
        :param request:
        :return:
        """
        # 查询所有的管理员会员
        user = User.objects.filter(super_u=1).all()
        length = len(user)
        # 创建分页器：每页N条记录
        paginator = Paginator(user, 2)
        # 获取每页会员数据
        try:
            page_books = paginator.page(page_num)
        except EmptyPage:
            # 如果page_num不正确，默认给用户404
            return http.HttpResponseNotFound('empty page')
        # 获取列表页总页数
        total_page = paginator.num_pages
        # 循环每个会员信息
        info = {}
        for i in page_books:
            # 查询每个人的相关信息

            info[str(i.id)] = {'ID':i.id,'name':i.username,'phone':i.mobile,'time':i.last_login,'state':i.super_u}
        context = {
            'length': length,
            'info':info,
            'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return render(request, 'admin/admin-list.html',context=context)
    
    def post(self, request,page_num):
        """
        查找管理员用户
        """
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        username = json_data.get("username")
        # 校验参数
        if not all([username]):
            return http.JsonResponse({'code': 1001, 'error': '用户名不能为空'})
        # 查询数据库
        try:
            user = User.objects.get(username=username,super_u=1)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '该管理员用户不存在'})
        context = {
            "ID":user.id,
            "name":user.username,
            "phone":user.mobile,
            "time":user.last_login
        }
        return http.JsonResponse({'code': 200, 'error': '查询成功',"context":context})
        
class Administrator_userRoleList(View):
    #返回前端页面
    def get(self, request):
        return render(request, 'admin/admin-userRoleList.html')
    #用户角色配置搜索
    def post(self, request):
        json_data = json.loads(request.body.decode())
        username = json_data.get("username")
        page_num = json_data.get('page')
        user = User.objects.filter(username__contains=username).order_by('id')
        paginator = Paginator(user, 8)
        length = len(user)
        if(length==0):
            return http.JsonResponse({'code': 1008, 'error': "用户不存在,请重新输入"})
        try:
            page_books = paginator.page(page_num)
        except EmptyPage:
            return http.HttpResponseNotFound('empty page')
        total_page = paginator.num_pages
        info = {}
        for i in page_books:
            info[str(i.id)] = {'ID': i.id, 'name': i.username, 'mobile': i.mobile, 'last_login': i.last_login,
                               'super_u': i.super_u, 'is_active': i.is_active}
        context = {
            'length': length,
            'info': info,
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": context})
        
class userRoleEdit(View):
    #用户角色分配回显
    def get(self, request,id):
        try:
            user = User.objects.get(id=id)
        except Exception as e:
            print(e)
        context = {
            'id':user.id,
            'username':user.username,
            'is_active':user.is_active,
            'super_u':user.super_u,
        }
        return render(request, 'admin/list-userRoleEdit.html',context=context)

    # 用户角色分配
    def post(self, request,id):
        json_data = json.loads(request.body.decode())
        id = json_data.get("id")
        is_active = json_data.get("is_active")
        super_u = json_data.get("super_u")
        # 校验参数
        if not all([id, is_active, super_u]):
            return http.JsonResponse({'code': 1001, 'error': '修改参数不能为空'})
        try:
            user = User.objects.get(id=id)
            user.is_active=is_active
            user.super_u=super_u
            user.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code':500, 'error':"修改失败"})
        return http.JsonResponse({'code':200, 'error':"修改成功"})

class Administrator_role(View):
    def get(self, request):
        """
        主页
        :param request:
        :return:
        """

        return render(request, 'admin/admin-role.html')


class Administrator_cate(View):
    def get(self, request):
        """
        主页
        :param request:
        :return:
        """

        return render(request, 'admin/admin-cate.html')


class Administrator_rule(View):
    def get(self, request):
        """
        主页
        :param request:
        :return:
        """

        return render(request, 'admin/admin-rule.html')


class Administrator_add(View):
    def get(self, request):
        """
        管理员添加页面
        :param request:
        :return:
        """

        return render(request, 'admin/admin-add.html')
    
    def post(self, request):
        # 获取参数
        json_data = json.loads(request.body.decode())
        # 获取用户名
        username = json_data.get('username')
        # 获取手机号码
        phone = json_data.get('phone')
        # 获取角色
        role = json_data.get('role')
        # 获取第一次密码
        password_one = json_data.get('password_one')
        # 获取第二次密码
        password_two = json_data.get('password_two')
        
        # 校验参数
        if not all([username, phone, role, password_one, password_two]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
                # 判断用户名是否是5-20个字符
        if not re.match(r'^[a-zA-Z0-9_]{5,20}$', username):
            return http.HttpResponseBadRequest('请输入5-20个字符的用户名')
        # 判断密码是否是8-20个数字
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password_one):
            return http.HttpResponseBadRequest('请输入8-20位的密码')
        # 判断两次密码是否一致
        if password_one != password_two:
            return http.HttpResponseBadRequest('两次输入的密码不一致')
        # 判断手机号是否合法
        if not re.match(r'^1[3-9]\d{9}$', phone):
            return http.HttpResponseBadRequest('请输入正确的手机号码')
        # 创建管理员用户
        try:
            user = User.objects.create_user(username=username, password=password_one, mobile=phone, dft_file='',image_tou='user_pic.png',super_u=1)
            user.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '添加管理员用户失败，请重试'})
        return http.JsonResponse({'code': 200, 'error': '添加管理员用户成功'})
    
    def delete(self, request):
        """删除管理员"""
        # 获取参数
        json_data = json.loads(request.body.decode())
        # 获取用户名
        user_id = json_data.get('id')
        
        # 查询管理员用户
        try:
            user = User.objects.get(id=user_id).delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code':1002,'error':"删除管理员用户失败"})
        
        return http.JsonResponse({'code':200, "error":'删除管理员用户成功'})
            

class Administrator_r_add(View):
    def get(self, request):
        """
        主页
        :param request:
        :return:
        """

        return render(request, 'admin/role-add.html')
