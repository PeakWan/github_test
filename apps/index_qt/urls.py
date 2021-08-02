from django.conf.urls import url

from apps.index_qt import views

urlpatterns = [
    url(r'^method_desc/(?P<method_name>\S+)/$', views.MethodDescrible.as_view(), name="method_desc"),
    url(r'^pile_echarts/$', views.PileEcharts.as_view(), name="pile_echarts"),  # echarts图表页面
    url(r'^index/$', views.Index_qtView.as_view(), name="index_qt"),  # 前台主页面
    url(r'^$', views.Index_sy.as_view(), name="index_sy"),  # 首页
    # url(r'^$', views.Index_qtView.as_view(), name="index_qt"),  # 前台主页面
    url(r'^boon/$', views.Index_f.as_view(), name="boon"),  # 按钮页面
    url(r'^user/', views.Index_user.as_view(), name="user"),  # 用户中心页面
    url(r'^login/$', views.Login.as_view(), name="login"),  # 登录页面
    url(r'^register/$', views.RegisterView.as_view(), name="register"),  # 注册页面
    url(r'^logout/$', views.LogoutView.as_view(), name="logout"),  # 退出页面
    url(r'^file/$', views.FileView.as_view(), name="file"),  # 退出页面
    url(r'^mobiles/(?P<mobile>1[3-9]\d{9})/count/$', views.MobileCountView.as_view(), name="mobile_count"),
    url(r'^usernames/(?P<username>[a-zA-Z0-9_]{5,20})/count/$', views.UsernameCountView.as_view(), name="user_count"),
    url(r'^user_pic/', views.User_pic.as_view(), name="user_pic"),  # 用户中心修改头像
    url(r'^user_pass/$', views.User_pass.as_view(), name="user_pass"),  # 用户修改密码
    url(r'^project_delete/(?P<project_id>\d+)/$', views.Project_delete.as_view(), name="user_pass"),  # 项目删除
    url(r'^project_edit/(?P<project_id>\d+)/$', views.Project_edit.as_view(), name="user_edit"),  # 项目编辑
    url(r'^data_edit/(?P<project_id>\d+)/$', views.DataEdit.as_view(), name="data_edit"),  # 项目编辑
    url(r'^video_course/$', views.Video_course.as_view(), name="video_course"),  # 视频教程

    url(r'^securiaty/$', views.Security.as_view(), name="security"),  # 安全性
    url(r'^legal_provisions/$', views.Legal_aprovisions.as_view(), name="legal_provisions"),  # 法律条款
    url(r'^privacy/$', views.Privacy.as_view(), name="privacy"),  # 隐私
    url(r'^about/$', views.About.as_view(), name="about"),  # 关于我们
    # url(r'^forget/$', views.Forget.as_view(), name="forget"),  # 忘记密码
    url(r'^forgetpassword/$', views.ForgetPassword.as_view(), name="forgetpassword"),  # 忘记密码修改密码
    url(r'^community/$', views.Community.as_view(), name="community"),  # 交流区页面
    # url(r'^wx/$', views.WeXinLogin.as_view()),  # 获取微信登录 code值
]

