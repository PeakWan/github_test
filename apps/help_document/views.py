from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from apps.index_qt.models import Help
from django import http
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication
from libs.get import get_s, loop_add, write, read,member_check
from apps.index_qt.models import Member, User, MemberType, Browsing_process, Modelbase,Commits_books
from django.core.paginator import Paginator, EmptyPage
import json
# Create your views here.
class HelpIndexView(APIView):
    """视频教程的类"""
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        """
        视频教程的主页面
        :param request:
        :return:
        """
        return render(request, 'index/help_video.html', context=context)
    
    def post(self, request):
        """获取数据"""
        
        # 查询所有的模型
        modellist = Help.objects.filter().all()
        # 循环每个模型信息
        info = []
        for i in modellist:
            # 查询每个模型的相关信息
            info.append({'ID':i.id,'video_name':i.video_name,'video_background':i.video_background,'video_info':i.video_info,'video_link':i.video_link})
        context = {
            'info':info,
        }
        return http.JsonResponse({'code': '200', 'error': '查询成功','context':context})
        

class userComment(APIView):
    """提交评论"""
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        # 获取参数
        json_data = json.loads(request.body.decode())
        
        text = json_data.get('content')
        print(text)
        # refurl = request.META.get('HTTP_REFERER', '/')
        # 校验参数
        if not text:
            return http.JsonResponse({'code': '2001', 'error': '评论内容不能为空'})

        # 保存评论的内容
        try:
            Commits_books.objects.create(user=request.user, content=text)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': '2002', 'error': '发布失败'})
        # return redirect(refurl)
        return http.JsonResponse({'code': 200, 'error': '评论成功'})


class userCommentSon(APIView):
    """对评论的评论的提交"""
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request, comment_id):
        # 获取参数
        json_data = json.loads(request.body.decode())
        
        text = json_data.get('son')
        username = json_data.get('username')
        print(text)

        # 校验参数
        if not text:
            return http.JsonResponse({'code': '2001', 'error': '评论内容不能为空'})
        book = Commits_books.objects.get(id=comment_id)
        # 保存父类的评论
        try:
            if username:
                Commits_books.objects.create(user=request.user, content=text, parent_id=comment_id,commit_username = username)
            else:
                Commits_books.objects.create(user=request.user, content=text, parent_id=comment_id)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': '2002', 'error': '发布失败'})
        # return redirect(refurl)
        return http.JsonResponse({'code': 200, 'error': '评论成功'})
        
class userCommentData(APIView):
    def post(self,request):
        json_data = json.loads(request.body.decode())
        # 获取当前页数
        page = json_data.get('page')
        # 获取一页多少条
        num = json_data.get('num')
        # 获取参数
        # 查询所有的评论
        comment_data = {}
        comment = Commits_books.objects.filter(parent_id=None).order_by('-create_time')

        # 创建分页器：每页N条记录
        paginator = Paginator(comment, int(num))
        # 获取每页会员数据
        try:
            comment_num = paginator.page(page)
        except EmptyPage:
            # 如果page_num不正确，默认给用户404
            return http.HttpResponseNotFound('empty page')
        # 获取列表页总页数
        total_page = paginator.num_pages
        
        
        # 循环获取到该用户的所有的评论,
        for num in comment_num:
            # 将每一条评论添加到字典里面
            comment_data[num.id] = {}
            comment_data[num.id]['username'] = num.user.username
            comment_data[num.id]['user'] = num.user.image_tou
            comment_data[num.id]['text'] = num.content
            comment_data[num.id]['time'] = num.create_time
            comment_data[num.id]['id'] = num.user.id
            comment_data[num.id]['subs'] = {}
            # 获取二级评论
            son = Commits_books.objects.filter(parent_id=num.id).all()
            if son:
                for a in son:
                    comment_data[num.id]['subs'][a.id] = {}
                    comment_data[num.id]['subs'][a.id]['user'] = a.user.image_tou
                    comment_data[num.id]['subs'][a.id]['text'] = a.content
                    comment_data[num.id]['subs'][a.id]['time'] = a.create_time
                    comment_data[num.id]['subs'][a.id]['username'] = a.user.username
                    comment_data[num.id]['subs'][a.id]['sub_username'] = a.commit_username
                    # comment_data[num.id]['subs'][a.id]['grandson'] = {}
                    # # 获取三级评论
                    # grandson = Commits_books.objects.filter(parent_id=a.id).all()
                    # if grandson:
                    #     for b in grandson:
                    #         comment_data[num.id]['subs'][a.id]['grandson'][b.id]= {}
                    #         comment_data[num.id]['subs'][a.id]['grandson'][b.id]['user'] = b.user.image_tou
                    #         comment_data[num.id]['subs'][a.id]['grandson'][b.id]['text'] = b.content
                    #         comment_data[num.id]['subs'][a.id]['grandson'][b.id]['time'] = b.create_time
                    #         comment_data[num.id]['subs'][a.id]['grandson'][b.id]['username'] = b.user.username
                    #         # comment_data[num.id]['subs']['grandson'][b.id]['grandson'] = {}

        context = {
            'info':comment_data,
            'total_page': total_page,  # 总页数
            'page_num': page,  # 当前页码
        }
        
        
        
        return http.JsonResponse({'code': 200, 'error': '查询成功','context':context})
        