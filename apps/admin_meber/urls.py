from django.conf.urls import url

from apps.admin_meber import views

urlpatterns = [

    url(r'^member_list/$', views.MemberList.as_view(), name="list"), # 会员管理页面
    
    url(r'^member_list_edit/(?P<id>\d+)/$', views.MemberListEdit.as_view(), name="member_list_edit"), # 会员管理页面编辑
    
    url(r'^member_list_add/$', views.MemberListAdd.as_view(), name="list_add"), # 会员管理页面
    url(r'^member_query/(?P<page_num>\d+)/$', views.MemberQuery.as_view(), name="query"), # 会员管理页面
    url(r'^member_del/(?P<page_num>\d+)/$', views.MemberDel.as_view(), name="del"), # 会员管理删除
    url(r'^member_level/$', views.MemberLevel.as_view(), name="level"), # 会员等级管理
    url(r'^member_level_del/$', views.MemberLevelDel.as_view(), name="member_level_del"), # 会员等级删除
    url(r'^member_edit/(?P<id>\d+)/$', views.MemberKissEdit.as_view(), name="level_edit"), # 会员等级编辑

    url(r'^member_add/$', views.MemberLevelAdd.as_view(), name="add"), # 会员等级添加
    url(r'^member_kiss_add/$', views.MemberKissAdd.as_view(), name="kiss_add"), # 会员积分添加
    
    url(r'^member_view/(?P<page_num>\d+)/$', views.MemberView.as_view(), name="view"), # 会员浏览
    
    url(r'^member_view_data/$', views.MemberViewData.as_view(), name="view_data"), # 会员浏览详情
    url(r'^member_list/$', views.MemberList.as_view(), name="list"),  # 会员管理页面
    url(r'^member_del/$', views.MemberDel.as_view(), name="del"),  # 会员管理删除
    url(r'^echart/$', views.Echart.as_view(), name="echart"),  # 统计汇总
    
    
    url(r'^video/(?P<page_num>\d+)/$', views.Video.as_view(), name="video"),  # 上传视频
    url(r'^video_add/$', views.Video_add.as_view(), name="video_add"),  # 添加
    url(r'^apinum_statistics/(?P<page_num>\d+)/$', views.apinumStatistics.as_view(), name="apinum_statistics"),  # api使用频次统计
    url(r'^usernum_statistics/(?P<page_num>\d+)/$', views.usernumStatistics.as_view(), name="usernum_statistics"),  # api使用频次统计
]
