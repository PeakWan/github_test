import json
import uuid
import math
import pymysql
from django.shortcuts import render
from django.views import View
from django.db.models import Q
from dateutil.relativedelta import relativedelta
from apps.index_qt.models import Member, User, MemberType, Browsing_process, Help,MethodDesc
from django import http
from django.core.paginator import Paginator, EmptyPage
from PIL import Image
from utils.tools import Tool
from django.core import serializers
import datetime

class MemberList(View):
    #返回列表页面
    def get(self, request):
        return render(request, 'admin/member-list.html')

    def post(self, request):
        json_data = json.loads(request.body.decode())
        username = json_data.get("username")
        vipRank = json_data.get("vipRank")
        page_num = json_data.get('page')

        try:
            user = User.objects.filter(Q(username__contains=username)|Q(mobile__contains=username))
            memberList = Member.objects.filter(user__in=user)
            if vipRank:
                memberList = Member.objects.filter(Q(user__in=user) & (Q(member_type__id=vipRank)))
            if len(memberList) < 1:
                return http.JsonResponse({'code': 1002, 'error': '您输入的会员不存在，请重新输入'})
            result = {}
            paginator = Paginator(memberList, 8)
            page_books = paginator.page(page_num)
            total_page = paginator.num_pages
            length = len(memberList)
            result = {}
            for i in page_books:
                member_start_time = i.member_initial_time
                member_end_time = i.member_last_time
                userLastLogin = i.user.last_login
                if (i.member_initial_time != None):
                    member_start_time = member_start_time.strftime('%Y-%m-%d %H:%M:%S')
                if (i.member_last_time != None):
                    member_end_time = member_end_time.strftime('%Y-%m-%d %H:%M:%S')
                if (i.user.last_login != None):
                    userLastLogin = userLastLogin.strftime('%Y-%m-%d %H:%M:%S')

                result[i.id] = {'ID': i.id, 'name': i.user.username, 'phone': i.user.mobile,
                                'state': i.member_type.member_name, 'last_login': userLastLogin,
                                'member_start': member_start_time, 'member_end': member_end_time}
            context = {
                'length': length,
                'result': result,
                'total_page': total_page,  # 总页数
                'page_num': page_num,  # 当前页码
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '系统错误'})
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": context})

class MemberQuery(View):
    """会员列表的根据用户名查询用户信息"""
    def post(self, request,page_num):
        json_data = json.loads(request.body.decode())
        username = json_data.get("username")
        vipRank = json_data.get("vipRank")
        # if not all([username]):
        #     return http.JsonResponse({'code': 1001, 'error': '用户名不能为空'})
        try:
            user = User.objects.filter(username__contains=username)
            memberList = Member.objects.filter(user__in=user)
            if vipRank:
                memberList = Member.objects.filter(Q(user__in=user)&(Q(member_type__id=vipRank)))
            if len(memberList) < 1:
                return http.JsonResponse({'code': 1002, 'error': '您输入的会员不存在，请重新输入'})
            result ={}
            for i in memberList:
                member_start_time = i.member_initial_time
                member_end_time = i.member_last_time
                userLastLogin = i.user.last_login
                if (i.member_initial_time != None):
                    member_start_time = member_start_time.strftime('%Y-%m-%d %H:%M:%S')
                if(i.member_last_time != None):
                    member_end_time = member_end_time.strftime('%Y-%m-%d %H:%M:%S')
                if(i.user.last_login !=None):
                    userLastLogin = userLastLogin.strftime('%Y-%m-%d %H:%M:%S')
                result[i.id] = {'ID': i.id, 'name': i.user.username, 'phone': i.user.mobile,
                                'state': i.member_type.member_name,'last_login': userLastLogin,
                                'member_start':member_start_time,'member_end': member_end_time}

                paginator = Paginator(list(result.values()), 2)
                currentPageData = paginator.page(page_num)
                total_page = paginator.num_pages
                length = len(list(result.values()))
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '系统错误'})
        return http.JsonResponse({'code': 200, 'error': '查询成功',"result":list(result.values()),"length":length
                                     ,"total_page":total_page,"page_num":page_num})
    
    def delete(self, request):
        """获取当前用户的id"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        id = json_data.get("id")
        # 查找用户
        try:
            user = Member.objects.get(id=id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '用户不存在'})
        # 逻辑删除用户会员
        user.is_delete = 1
        user.save()
        return http.JsonResponse({'code': 200, 'error': '删除成功'})

class MemberListAdd(View):
    def get(self, request):
        """
        会员列表添加
        :param request:
        :return:
        """

        return render(request, 'admin/member-add.html')

class MemberDel(View):
    def get(self, request, page_num):
        """
        删除会员
        :param request:
        :return:
        """
        # 查询所有的会员
        memberlist = Member.objects.filter(is_delete=1).all()
        length = len(memberlist)
        # 创建分页器：每页N条记录
        paginator = Paginator(memberlist, 10)
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
            info[str(i.id)] = {'ID': i.id, 'name': i.user.username, 'phone': i.user.mobile,
                               'state': i.member_type.member_name}
        context = {
            'length': length,
            'info': info,
            'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return render(request, 'admin/member-del.html', context=context)

    def put(self, request):
        """恢复会员"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        id = json_data.get("id")
        
        # 查询该用户
        try:
            user = Member.objects.get(id=id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '用户不存在'})
        # 恢复该用户
        user.is_delete = 0
        user.save()
        # 返回结果
        return http.JsonResponse({'code': 200, 'error': '恢复成功'})
        
    def post(self, request):
        """查找被逻辑删除的会员"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        username = json_data.get("username")
        # 校验参数
        if not all([username]):
            return http.JsonResponse({'code': 1001, 'error': '用户名不能为空'})
        # 查询数据库
        try:
            user = User.objects.get(username=username)
            member = Member.objects.get(user=user,is_delete=1)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '该会员用户不存在'})
        context = {
            "ID": member.id,
            "name": user.username,
            "phone": user.mobile,
            "state": member.member_type.member_name
        }
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": context})
    
    def delete(self, request):
        """删除会员"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        id = json_data.get("id")
        # 查询相关用户信息删除
        try:
            member = Member.objects.get(id=id)
            user = User.objects.get(id=member.user_id).delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '删除用户失败'})
        # 返回结果
        return http.JsonResponse({'code': 200, 'error': '删除用户成功'})
        
class MemberKissAdd(View):
    def get(self, request):
        """
        会员积分管理添加
        :param request:
        :return:
        """

        return render(request, 'admin/kiss-add.html')

class MemberKissEdit(View):
    def get(self, request,id):
        """
         会员管理页面编辑
        :param request:
        :return:
        """
        # 查询当前会员的类型
        try:
            member =MemberType.objects.get(id=id)
        except Exception as e:
            print(e)
        context = {
            'id':member.id,
            'name':member.member_name,
            'number':member.number,
            'projects':member.projects,
            'price':member.price,
            'flow_number':member.flow_number
        }
        return render(request, 'admin/level-edit.html',context=context)
    def post(self, request, id):
        """编辑会员类型"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        # 获取修改的名字
        name = json_data.get("name")
        # 获取修改的分析条数
        number = json_data.get("number")
        # 获取要修改的项目条数
        project = json_data.get("project")
        price = json_data.get("price")
        flow = json_data.get("flow")

        # 校验参数
        if not all([name, number, project]):
            return http.JsonResponse({'code': 1001, 'error': '修改参数不能为空'})

        # 查询当前类型是否存在
        try:
            member = MemberType.objects.get(id = id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '该用户类型不存在'})

        # 修改会员类型信息
        member.member_name = name
        member.number = number
        member.projects = project
        member.price = price
        member.flow_number = flow
        # 保存修改信息
        member.save()

        return http.JsonResponse({'code':200, 'error':"修改成功"})

        
class MemberListEdit(View):
    def get(self, request,id):
        """
        会员等级编辑
        :param request:
        :return:
        """
        # 查询当前会员的类型

        try:
            # print(123123123123123123)

            member = Member.objects.get(id=id)
            user = User.objects.get(id=member.user_id)
            # print(member.member_type, "-=-=-=-=-=-=")
            # print(member.member_type_id)
            # user =
            member_start_time = member.member_initial_time
            member_end_time = member.member_last_time
            if i.member_initial_time != None:
                member_start_time = member_start_time.strftime('%Y-%m-%d %H:%M:%S')
                member_end_time = member_end_time.strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(e)

        context = {
            'id':user.id,
            'name':user.username,
            'number':user.mobile,
            'projects':member.member_type_id,
            'member_start':member_start_time,
            'member_end':member_end_time
        }

        return render(request, 'admin/list-edit.html',context=context)

    def post(self, request, id):
        """编辑会员类型"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        # 获取修改的名字
        name = json_data.get("name")
        # 获取修改的手机号
        number = json_data.get("number")
        # 获取要修改的会员等级
        project = json_data.get("project")
        # 获取增加到得时间得时间戳
        member_end_time = json_data.get('time')
        # 获取类型
        member_time_type = json_data.get('type')

        # 校验参数
        if not all([name, number, project]):
            return http.JsonResponse({'code': 1001, 'error': '修改参数不能为空'})

        # 查询当前类型是否存在
        try:

            member = Member.objects.get(user_id=id)

            print(member.user_id, "-=-=-")
            user = User.objects.get(id=id)
            print(user.id, "=-=-=-")

        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '该用户类型不存在'})


        if user.username != name:
            count_name = User.objects.filter(username=name).count()
            if int(count_name) == 1:
                return http.JsonResponse({'code':1008, 'error':"用户名重复"})
            else:
                user.username=name
                user.save()
        if user.mobile != number:
            count_mobile = User.objects.filter(mobile=number).count()
            if int(count_mobile) == 1:
                return http.JsonResponse({'code':1008, 'error':"手机号重复"})
            else:
                user.mobile=number
                user.save()
        if member.member_type_id != project:
        # 修改会员类型信息
            member.member_type_id=int(project)
            member.save()


        try:
            # 保存修改信息
            # 保存修改信息
            if member.member_initial_time==None:
                member.member_initial_time=datetime.datetime.now()
            if member.member_last_time==None:
                 member.member_last_time = datetime.datetime.now()
                 
            if member_time_type == 'day':
                # 充值后的时间
                time_last = (member.member_last_time  + datetime.timedelta(days = int(member_end_time))).strftime('%Y-%m-%d %H:%M:%S')
            elif member_time_type == 'months':
                # 充值后的时间
                time_last = (member.member_last_time  + relativedelta(months=+int(member_end_time))).strftime('%Y-%m-%d %H:%M:%S')
            elif member_time_type == 'year':
                # 充值后的时间
                time_last = (member.member_last_time  + relativedelta(years=+int(member_end_time))).strftime('%Y-%m-%d %H:%M:%S')
            else:
                return http.JsonResponse({'code':500, 'error':"非法修改"})

            member.member_last_time = time_last
            member.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code':500, 'error':"添加失败，请及时联系后台人员哦^-^"})
        return http.JsonResponse({'code':200, 'error':"修改成功"})
        
        
class MemberLevelAdd(View):
    def get(self, request):
        """
        会员等级添加
        :param request:
        :return:
        """

        return render(request, 'admin/level-add.html')

    def post(self, request):
        """添加会员等级"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        # 获取修改的名字
        name = json_data.get("name")
        # 获取修改的分析条数
        number = json_data.get("number")
        # 获取要修改的项目条数
        project = json_data.get("project")
        flow=json_data.get("flow")
        price = json_data.get("price")
        # 校验参数
        if not all([name, number, project,flow,price]):
            return http.JsonResponse({'code': 1001, 'error': '参数不能为空'})
        # 添加会员等级
        try:
            member = MemberType.objects.create(member_name=name,price=price, number=number, projects = project,flow_number=flow)
            member.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1002, 'error': '添加会员类型失败'})

        return http.JsonResponse({'code': 200, 'error': "添加成功"})

class MemberLevel(View):
    def get(self, request):
        """
        会员等级
        :param request:
        :return:
        """
        # 查询所有的会员类型
        member_type = MemberType.objects.filter().all()
        length = len(member_type)
        # 保存会员等级的字典
        info = {}
        # 循环所有的会员类型
        for i in member_type:
            info[str(i.id)] = {'ID':i.id,'name':i.member_name,'number':i.number,'price':i.price,'projects':i.projects,'flow_number':i.flow_number}


        context = {
            'info':info,
            'length':length
        }

        return render(request, 'admin/member-level.html',context=context)

class MemberLevelDel(View):
    #会员等级删除
    def post(self, request):
        try:
            json_data = json.loads(request.body.decode())
            arr_box = json_data.get("arr_box")
            #members=MemberType.objects.
            MemberType.objects.filter(id__in=arr_box).delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '删除会员等级失败'})
        return http.JsonResponse({'code': 200, 'error': "删除会员等级删除成功"})

class MemberView(View):
    
    def get(self, request,page_num):
        """
        浏览记录
        :param request:
        :return:
        """
        # 查询所有的浏览记录
        # print(111111111111111111)
        browsing = Browsing_process.objects.filter().all()
        # print(222222222222222222)
        length = len(browsing)
        # 创建分页器：每页N条记录
        paginator = Paginator(browsing, 10)
        # 获取每页会员数据
        try:
            page_books = paginator.page(page_num)
        except EmptyPage:
            # 如果page_num不正确，默认给用户404
            return http.HttpResponseNotFound('empty page')
        # 获取列表页总页数
        total_page = paginator.num_pages

        # 循环添加数据
        info = []
        for i in page_books:
            # 定义一个存放每条数据的字典
            process = {}
            a = eval(i.process_info)
            process[str(i.id)] = {
                'id':i.id,
                'name':i.user.username,
                'is_delete':i.is_delete,
                'time':i.create_time,
                'bro':a
            }
            info.append(process)
        # print(info)
        context = {
            'length': length,
            'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
            'info':info
        }
        return render(request, 'admin/member-view.html',context=context)
        
    def post(self, request,page_num):
        """查找用户相关记录"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取修改的名字
        name = json_data.get("name")
        # 校验参数
        if not all([name]):
            return http.JsonResponse({'code': 1001, 'error': "参数不能为空"})

        # 根据用户名查询该用户的浏览记录
        user = User.objects.get(username=name)
        info = Browsing_process.objects.filter(user_id=user.id).all()
        
        data = []
        # 循环每一条数据
        for i in info:
            a = {}
            process = eval(i.process_info)
            a['id']= i.id
            a['name'] = i.user.username
            a['is_delete'] = i.is_delete
            a['time'] = i.create_time
            a['process'] = process['name']
            data.append(a)
            
        return http.JsonResponse({'code': 200, 'error': "修改成功", 'data':data})
        
    def delete(self, request,page_num):
        """删除浏览记录"""
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        num = json_data.get("id")
        # num = int(num)
        # 查询相关用户信息删除
        try:
            process = Browsing_process.objects.filter(id=num)

            process.delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '删除浏览记录失败'})
        # 返回结果
        return http.JsonResponse({'code': 200, 'error': '删除浏览记录成功'})


class MemberViewData(View):
    def get(self, request):
        """
        浏览记录详情
        :param request:
        :return:
        """

        return render(request, 'admin/admin-view.html')


class Echart(View):
    def get(self, request):
        """
        浏览记录
        :param request:
        :return:
        """

        return render(request, 'admin/echart.html')
        
class Video(View):
    def get(self, request, page_num):
        """
        上传视频
        :param request:
        :return:
        """
        # 查询所有的管理员会员
        video = Help.objects.all()
        length = len(video)
        print(length, "-=-=-12412412412=-=-=-")
        # 创建分页器：每页N条记录
        paginator = Paginator(video, 8)

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
            # 查询每个人的相关信息video_name=username, video_background=phone, video_info=video_file, video_link=phone2
            info[str(i.id)] = {'ID':i.id,'video_name':i.video_name,'video_background':i.video_background,'video_link':i.video_link}
            
        context = {
            'length': length,
            'info':info,
            'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return render(request, 'admin/video.html', context=context)
    
    def post(self, request, page_num):
    #     """
    #     查找管理员用户
    #     """
        # json_data = json.loads(request.body.decode())
        
        # json_data = json.loads(request.body.decode())
        # 获取参数
        # 视频教程标题
        username = request.POST.get("username")
        # 视频教程说明
        phone = request.POST.get("phone")
        # 视频教程链接
        phone2 = request.POST.get("phone2")
        # 上传图像
        f = request.FILES.get("img")
        
        if not username:
            return http.JsonResponse({'code': 1002, 'error': '请上传视频教程标题'})
        if not phone:
            return http.JsonResponse({'code': 1003, 'error': '请上传视频教程说明'})
        if not phone2:
            return http.JsonResponse({'code': 1004, 'error': '请上传视频教程链接'})
        if not f:
            return http.JsonResponse({'code': 1005, 'error': '请上传图像'}) 
        # print(f, "=============")
        img = Image.open(f)
        # print(2222222222222)
        # print(f.name)
        id = uuid.uuid1()
        
        img_name = str(id) + str(f.name) 
        
        print(3333333333333)
        print(img_name, "===============")
        video_file = '/www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent/static/upload/user_images/%s' % img_name
        
        st_file = '/static/upload/user_images/%s' % img_name
        
        img = img.convert('RGB')
        img.save(video_file)
        
        video = Help.objects.create(video_name=username, video_background=phone, video_info=st_file, video_link=phone2)
        video.save()

        return http.JsonResponse({'code': 200, 'error': '上传成功'})
        # return http.JsonResponse({'code': 200, 'error': '查询成功'})


    def delete(self, request, page_num):
        
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        id = json_data.get("id")
        print(id, "-=-=-=-==-id=-=-=12-3=12412")
        try:
            # user = Help.objects.get(id=id)
            user = Help.objects.get(id=id).delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1001, 'error': '用户不存在'})
        return http.JsonResponse({'code': 200, 'error': '删除成功'})



class Video_add(View):
    def get(self, request):
        """
        上传视频
        :param request:
        :return:
        """
        # 查询所有的管理员会员
        # user = User.objects.filter(super_u=1).all()
        # length = len(user)
        # # 创建分页器：每页N条记录
        # paginator = Paginator(user, 2)
        # # 获取每页会员数据
        # try:
        #     page_books = paginator.page(page_num)
        # except EmptyPage:
        #     # 如果page_num不正确，默认给用户404
        #     return http.HttpResponseNotFound('empty page')
        # # 获取列表页总页数
        # total_page = paginator.num_pages
        # # 循环每个会员信息
        # info = {}
        # for i in page_books:
        #     # 查询每个人的相关信息
        #     info[str(i.id)] = {'ID':i.id,'name':i.username,'phone':i.mobile,'time':i.last_login,'state':i.super_u}
        # context = {
        #     'length': length,
        #     'info':info,
        #     'page_skus': page_books,  # 分页后数据
        #     'total_page': total_page,  # 总页数
        #     'page_num': page_num,  # 当前页码
        # }
        return render(request, 'admin/video_add.html')
    
    def post(self, request,page_num):
    #     """
    #     查找管理员用户
    #     """
    #     # 接收参数
    #     json_data = json.loads(request.body.decode())
    # #     # 获取参数
    #     username = json_data.get("username")
    #     phone = json_data.get("phone")
    #     phone2 = json_data.get("phone2")
    #     # 校验参数
    #     if not all([username]):
    #         return http.JsonResponse({'code': 1001, 'error': '用户名不能为空'})
    #     # 查询数据库
    #     try:
    #         user = User.objects.get(username=username,super_u=1)
    #     except Exception as e:
    #         print(e)
    #         return http.JsonResponse({'code': 1003, 'error': '该管理员用户不存在'})
    #     context = {
    #         "ID":user.id,
    #         "name":user.username,
    #         "phone":user.mobile,
    #         "time":user.last_login
    #     }
        return http.JsonResponse({'code': 200, 'error': '查询成功',"context":context})
    
class apinumStatistics(View):
    def  connect_mysql(sql):
        db = pymysql.connect(
            host='47.92.236.49',
            port=3306,
            user='znfxpt_8dfish_vi',
            password='XLihaSLj2Ld7RR8R',
            database='znfxpt_8dfish_vi',
            charset='utf8'
        )
        cur = db.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        db.close()
        return data
    def get(self, request, page_num):

        sql="select apiName,count(*)as apiNum  from(select substring_index(substring(process_info,11),\"'\",1) as apiName from tb_process where 1=1 ) As A where 1=1 group by apiName order by apiNum desc limit 10"
        data=apinumStatistics.connect_mysql(sql)
       # obj = apiNum.objects.extra(select={'tmp':sql,'apiName':'apiName','apiNum':'apiNum'},select_params=())
        length = len(data)
        # 创建分页器：每页N条记录
        paginator = Paginator(data, 10)
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
        k=0;
        for i in page_books:
            # 查询每个人的相关信息video_name=username, video_background=phone, video_info=video_file, video_link=phone2
            info[k] = {'apiName': i[0], 'apiNum': i[1] ,'seq':k+1}
            k=k+1;

        context = {
            'length': length,
            'info': info,
            'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return render(request, 'admin/apinum-statistics.html',context=context)

    def post(self, request,page_num):
        # 接收参数
        json_data = json.loads(request.body.decode())
        # 获取参数
        apiName = json_data.get("apiName")
        duration = json_data.get("duration")
        time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_endByWeekAgo=(datetime.datetime.now()-  datetime.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        time_endByMonthAgo = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        sqlFilter= "and apiName like '%"+apiName+"%'"  if apiName else ''
        timeFilter=""
        if duration=='week':
            timeFilter = "and create_time between '"+time_endByWeekAgo+"' and  '"+time_current+"'"
        elif duration=='month':
            timeFilter = "and create_time between '"+time_endByMonthAgo+"' and  '"+time_current+"'"
        # 校验参数
        sql = "select apiName,count(*)as apiNum  from(select substring_index(substring(process_info,11),\"'\",1) as apiName from tb_process where 1=1 "+timeFilter+" ) As A where 1=1 "+sqlFilter+"group by apiName order by apiNum desc limit 10"
        print(sql)
        data = apinumStatistics.connect_mysql(sql)
        # obj = apiNum.objects.extra(select={'tmp':sql,'apiName':'apiName','apiNum':'apiNum'},select_params=())
        length = len(data)
        # 创建分页器：每页N条记录
        paginator = Paginator(data, 10)
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
        k = 0;
        for i in page_books:
            # 查询每个人的相关信息video_name=username, video_background=phone, video_info=video_file, video_link=phone2
            info[k] = {'apiName': i[0], 'apiNum': i[1], 'seq': k + 1}
            k = k + 1;

        context = {
            'length': length,
            'info': info,
            #'page_skus': page_books,  # 分页后数据
            'total_page': total_page,  # 总页数
            'page_num': page_num,  # 当前页码
        }
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": context})   
        
class usernumStatistics(View):
    #高频用户统计页面初始渲染
    def get(self, request, page_num):
        sql="select count(A.id)as userLoginNum,B.username from (select id,login_time,user_id from tb_login_history where 1=1) As A left join (select id,username from tb_users where super_u <> 1)As B  on A.user_id=B.id  GROUP BY  B.username  desc limit 10"
        data=Tool.getData(sql,None)
        #封装参数 eg[{userLoginNum:3,userName:test},{userLoginNum:2,userName:test2}]
        dataListMap=Tool.tupleToMapList(data,['userLoginNum', 'userName'])
        return render(request, 'admin/usernum-statistics.html',{"info":dataListMap})
    #提交表格查询用户登录统计次数
    def post(self, request,page_num):
        json_data = json.loads(request.body.decode())
        username = json_data.get("username")
        time = json_data.get("time")
        time_endByWeekAgo=(datetime.datetime.now()-  datetime.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        time_endByMonthAgo = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        usernameFilter= "and username like '%"+username+"%'"  if username else ''
        timeFilter = ""
        if time=='week':
            timeFilter = "and login_time>='"+time_endByWeekAgo+"'"
        elif time=='month':
            timeFilter = "and login_time>='" +time_endByMonthAgo+"'"
        # 校验参数
        sql = "select count(A.id)as userLoginNum,B.username from (select id,login_time,user_id from tb_login_history where 1=1 " +timeFilter+") As A  join (select id,username from tb_users where super_u <> 1  "+usernameFilter+")As B  on A.user_id=B.id  GROUP BY  B.username order by userLoginNum desc limit 10"
        data = Tool.getData(sql, None)
        dataListMap = Tool.tupleToMapList(data, ['userLoginNum', 'userName'])
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": dataListMap})
        
#方法详情
class MethodDetail(View):
    #返回方法详情页面
    def get(self, request):
        return render(request, 'admin/method-detail.html')
    # 返回方法详情列表
    def post(self, request):
        json_data = json.loads(request.body.decode())
        method_name = json_data.get("methodName")
        page_num = json_data.get('page')
        try:
            length = MethodDesc.objects.filter(method_name__contains=method_name).count()
            if length < 1:
                return http.JsonResponse({'code': 1002, 'error': '您输入的方法名不存在，请重新输入'})
            total_page=math.ceil(length/5)
            methodList = list(MethodDesc.objects.filter(method_name__contains=method_name).values()[(page_num-1)*5:page_num*5])
            context = {
                'length': length, #总数
                'result': methodList, #结果集
                'total_page': total_page,  # 总页数
                'page_num': page_num,  # 当前页码
            }
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 1003, 'error': '系统错误'})
        return http.JsonResponse({'code': 200, 'error': '查询成功', "context": context})

#方法编辑
class MethodEdit(View):
    #打开方法编辑页面
    def get(self, request,id):
        try:
            method_obj = list(MethodDesc.objects.filter(id=id).values())
            context={}
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': "获取方法信息失败"})
        return render(request, 'admin/method-edit.html',context=method_obj[0])
    #修改方法信息
    def post(self, request, id):
        json_data = json.loads(request.body.decode())
        method_id = json_data.get("method_id")
        method_name = json_data.get("method_name")
        method_desc = json_data.get("method_desc")
        method_url = json_data.get('method_url')
        try:
            method_obj = MethodDesc.objects.get(id=method_id)
            method_obj.method_name=method_name
            method_obj.method_desc = method_desc
            method_obj.method_url = method_url
            method_obj.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '修改方法信息失败'})
        return http.JsonResponse({'code':200, 'error':"修改成功"})

#方法添加
class MethodAdd(View):
    def get(self, request):
        return render(request, 'admin/method-add.html')

    def post(self, request):
        json_data = json.loads(request.body.decode())
        method_name = json_data.get("method_name")
        method_desc = json_data.get("method_desc")
        method_url = json_data.get("method_url")
        # 校验参数
        if not all([method_name, method_desc, method_url]):
            return http.JsonResponse({'code': 500, 'error': '参数不能为空'})
        # 添加方法信息
        try:
            method_obj = MethodDesc.objects.create(method_name=method_name, method_desc=method_desc,
                                                   method_url=method_url)
            method_obj.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '添加方法失败'})
        return http.JsonResponse({'code': 200, 'error': "添加成功"})   

class MethodDel(View):
    #方法删除
    def post(self, request):
        try:
            json_data = json.loads(request.body.decode())
            arr_box = json_data.get("arr_box")
            MethodDesc.objects.filter(id__in=arr_box).delete()
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '删除方法失败'})
        return http.JsonResponse({'code': 200, 'error': "删除方法成功"})        
        
        

