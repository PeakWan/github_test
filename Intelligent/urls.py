"""Intelligent URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from apscheduler.scheduler import  Scheduler
from apps.index_qt.views import delFile

urlpatterns = [
    url(r'^admin/', include("apps.index.urls", namespace="index")),
    url(r'^admin/member/', include("apps.admin_meber.urls", namespace="member")),  # 会员管理
    url(r'^admin/Administrator/', include("apps.Administrator.urls", namespace="Administrator")),  # 管理员
    url(r'^', include("apps.index_qt.urls", namespace="index_qt")),
    url(r'^', include("apps.upload_exel.urls", namespace="upload")),
    url(r'^', include("apps.verifications.urls", namespace="verifications")),
    url(r'^', include("apps.Smart.urls", namespace="smart")),  # 智能统计
    url(r'^', include("apps.chart.urls", namespace="xsmartplot")),  # 智能绘图
    url(r'^', include("apps.Advanced.urls", namespace="advanced")),  # 高级分析
    url(r'^model/', include("apps.modelBase.urls", namespace="model")),  # 模型库
    url(r'^help/', include("apps.help_document.urls", namespace="help")),  # 视频教程
]


handler500 = "apps.verifications.views.page_error"
handler404 = "apps.verifications.views.page_not_found"
#定时任务
sched = Scheduler()
# @sched.interval_schedule(seconds=60)
@sched.interval_schedule(days=1)
def my_task():
    delFile()
sched.start()