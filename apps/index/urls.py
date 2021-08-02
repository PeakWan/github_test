from django.conf.urls import url

from apps.index import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name="index"),
    url(r'^wecome/$', views.WeCome.as_view(), name="wecome"),  # 欢迎小页面
    url(r'^login/$', views.Login.as_view(), name="login"),  # 登录页面
    url(r'^logout/$', views.LogoutView.as_view(), name="logout"),  # 退出页面

]
