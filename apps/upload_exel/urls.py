from django.conf.urls import url
from apps.upload_exel import views

urlpatterns = [
    
    url(r'^upload/$', views.UploadView.as_view(), name="exel"),  # 上传表格
    url(r'^index_home/(?P<project_id>\d+)/$', views.Index_tou.as_view(), name="home"),  # 主页
    url(r'^miss_delete/(?P<project_id>\d+)/$', views.MissDeleteView.as_view(), name="miss_delete"),  # 缺失数据删除函数
    # url(r'^analysis/$', views.AnalysisiView.as_view(), name="analysis"),  # 开始表格
    url(r'^normality/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Normality.as_view(), name="normality"),  # 正态性校验
    url(r'^smart_article/(?P<project_id>\d+)/$', views.SmartView.as_view(), name="article"),  # 智能文章页面
    url(r'^save_data/', views.SaveView.as_view(), name="save"),# 文章保存
    url(r'^statistic/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.StatisticView.as_view(), name="statistic"),  # 两组t检验
    # url(r'^statistic_u/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Statistic_uView.as_view(), name="statistic_u"),  # 两组u检验
    url(r'^miss_data_filling/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.MissDataFilling.as_view(), name="miss_filling"),  # 数据填补
    url(r'^get_var_colinear/(?P<project_id>\d+)/$', views.GetVarView.as_view(), name="colinear"),  # 获取相关性过高变量   
    url(r'^get_var_vif/(?P<project_id>\d+)/$', views.GetVifView.as_view(), name="vif"),  # 获取共线性过高变量
    url(r'^data_balance/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.DataBalanceView.as_view(), name="balance"),  # 样本均衡
    url(r'^psm_matchin/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.PsmMatchinView.as_view(), name="matchin"),  # PSM倾向性匹配
    url(r'^dummies_recoding/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.DummiesView.as_view(), name="recoding"),  # 哑变量重编码
    url(r'^group_recoding/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.GroupRecodingView.as_view(), name="group"),  # 分组重编码
    url(r'^data_standardization/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.DataStandardizationView.as_view(), name="standardization"),  # 数据标准化
    url(r'^non_numerical_value_process/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.NonNumericalView.as_view(), name="numerical"),  # 非数值异常值处理
    url(r'^abnormal_deviation_process/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.AbnormalView.as_view(), name="abnormal"),  # 异常偏离值处理
    url(r'^grouping/(?P<project_id>\d+)/$', views.GroupingView.as_view(), name="grouping"),  # 获取列的数据类型以及数量
    url(r'^listing/$', views.ListingView.as_view()),# 传输列名
    url(r'^button/$', views.ButtonView.as_view()), # 按钮选择
    url(r'^download/(?P<project>\d+)/$', views.Download.as_view(),name="download"),  # 下载处理后的文件
    url(r'^orifinal/(?P<project>\d+)/$', views.Orifinal.as_view(),name="Orifinal"),  # 下载原始的文件
    url(r'^article_download/(?P<project>\d+)/$', views.Article_Download.as_view(),name="Article_Download"),  # 下载文章
    url(r'^survival/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Survival.as_view(),name="survival"),  # 生存分析
    url(r'^ref/(?P<project_id>\d+)/$', views.REFView.as_view(), name="ref"),  # 获取x2中单选列的类型
    url(r'^reflist/(?P<project_id>\d+)/$', views.REFListView.as_view(), name="reflist"),  # 获取x3中多选列的类型
    url(r'^manipulate/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.ManipulateView.as_view(), name="manipulate"),  # 数据操作
    url(r'^data_transform/(?P<project_id>\d+)/(?P<data_id>\d+)/$', views.Transform.as_view(), name="data_transform"),  # 数据转化
]
