from django.conf.urls import url

from apps.Administrator import views

urlpatterns = [
    url(r'^Administrator_list/(?P<page_num>\d+)/$', views.Administrator_list.as_view(), name="A_list"),  # 管理员列表
    url(r'^Administrator_userRoleList/$', views.Administrator_userRoleList.as_view(), name="A_userRoleList"),  # 用户列表
    url(r'^Administrator_userRoleEdit/(?P<id>\d+)/$', views.userRoleEdit.as_view(), name="A_userRoleEdit"), # 用户角色页面编辑
    url(r'^Administrator_role/$', views.Administrator_role.as_view(), name="A_role"),  # 角色管理
    url(r'^Administrator_cate/$', views.Administrator_cate.as_view(), name="A_cate"),  # 权限分类
    url(r'^Administrator_rule/$', views.Administrator_rule.as_view(), name="A_rule"),  # 权限管理
    url(r'^Administrator_add/$', views.Administrator_add.as_view(), name="A_add"),  # 权限管理添加
    url(r'^Administrator_r_add/$', views.Administrator_r_add.as_view(), name="A_r_add"),  # 角色管理添加
]

