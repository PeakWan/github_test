from django.conf.urls import url
from apps.Smart import views

urlpatterns = [
    
    url(r'^zhfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Zhfx.as_view(), name="zhfx"),  # 综合智能统计分析
    url(r'^sjms/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Sjms.as_view(), name="sjms"),  # 数据描述
    url(r'^fxqx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Fxqx.as_view(), name="fxqx"),  # 方差齐性
    url(r'^xgx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Xgx.as_view(), name="xgx"),  # 相关性分析
    url(r'^gxx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Gxx.as_view(), name="gxx"),  # 共线性分析
    url(r'^sample/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Sample.as_view(), name="sample"),  # 共线性分析
    # 非参数检验
    url(r'^fcsjy/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Fcsjy.as_view(), name="fcsjy"),  # 非参数检验

    url(r'^kfjy/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Kfjy.as_view(), name="kfjy"),  # 卡方检验
    url(r'^f_jy/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.F_jy.as_view(), name="f_jy"),  # Fisher检验
    url(r'^d_fc/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.D_fc.as_view(), name="d_fc"),  # 单因素方差分析
    url(r'^d_fc2/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.D_fc2.as_view(), name="d_fc2"),  # 多因素方差分析
    url(r'^zws/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Zws.as_view(), name="zws"),  # 中位数差异分析
    url(r'^dcbj/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Dcbj.as_view(), name="dcbj"),  # 多重比较
    url(r'^znfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Znfx.as_view(), name="znfx"),  # 智能分组分析
    url(r'^dmxfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.MultiView.as_view(), name="dmxfx"),  # 多模型回归分析
    
    url(r'^dys/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Dys.as_view(), name="dys"),  # 单因素\多因素分析
    
    url(r'^fcfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Startification.as_view(), name="fcfx"),  # 分层分析
    url(r'^phqx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Smooth.as_view(), name="phqx"),  # 平滑曲线拟合分析
    url(r'^qsfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Qsfx.as_view(), name="qsfx"),  # 趋势回归分析
    
]
