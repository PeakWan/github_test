from django.conf.urls import url

from apps.Advanced import views


urlpatterns = [
    
    url(r'^loghg/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Loghg.as_view(), name="loghg"),  # logistic回归
    url(r'^xxhg/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Xxhg.as_view(), name="xxhg"),  # 2.线性回归
    url(r'^roc/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Roc.as_view(), name="roc"),  # 3.二组独立样本ROC分析
    url(r'^coxfx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Coxfx.as_view(), name="coxfx"),  # COX回归分析
    url(r'^zydpx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Zydpx.as_view(), name="zydpx"),  # 重要度排序  影响因子重要度分析
    url(r'^blmx/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Blmx.as_view(), name="blmx"),  # 变量模型评分分析
    url(r'^jqxxfl/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Jqxxfl.as_view(), name="jqxxfl"),  # 机器学习分类
    url(r'^jqxxhg/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Jqxxhg.as_view(), name="jqxxhg"),  # 机器学习回归
    url(r'^jqxxjl/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Jqxxjl.as_view(), name="jqxxjl"),  # 机器学习聚类
    url(r'^d_roc/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.D_roc.as_view(), name="d_roc"),  # 多模型分析（ROC评价）
    url(r'^nri/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Nri.as_view(), name="nri"),
    url(r'^rcs/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Rcs.as_view(), name="rcs"),  # 净重分类指数
    url(r'^dca/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Dca.as_view(), name="dca"),
    # url(r'^d_dzb/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.D_dzb.as_view(), name="d_dzb"),  # 多模型分析（多指标评价）
    url(r'^wait/$', views.Wait.as_view(), name="wait"),  # 等待添加页面
    
]



