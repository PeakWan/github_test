from django.conf.urls import url

from apps.chart import views

urlpatterns = [
    url(r'pie_graph/(?P<project_id>\d+)/$', views.PieGraph.as_view(), name="pie"),  #  1.分类排序饼图
    url(r'horizontal/(?P<project_id>\d+)/$', views.HorizontalBarPlot.as_view(), name="horizontal"),  # 水平柱状图
    url(r'forest_plot/(?P<project_id>\d+)/$', views.ForestPlot.as_view(), name="forest"),  # 森林图
    url(r'box_plot/(?P<project_id>\d+)/$', views.BoxPlot.as_view(), name="box"),  # 箱图
    url(r'scatter_plot/(?P<project_id>\d+)/$', views.ScatterPlot.as_view(), name="scatter"),  # 散点线图
    url(r'violin_plot/(?P<project_id>\d+)/$', views.ViolinPlot.as_view(), name="violin"),  # 小提琴图
    url(r'rel_plot/(?P<project_id>\d+)/$', views.RelPlot.as_view(), name="rel"),  # 线混合图
    url(r'strip_plot/(?P<project_id>\d+)/$', views.StripPlot.as_view(), name="strip"),  # 分类散点图
    url(r'stackbar_plot/(?P<project_id>\d+)/$', views.StackbarPlot.as_view(), name="stackbar"),  # 堆积柱状图
    url(r'pile_plot/(?P<project_id>\d+)/$', views.Pile.as_view(), name="pile"),  # Echarts堆积柱状图Pile
    url(r'comparison/(?P<project_id>\d+)/$', views.ComparisonPlot.as_view(), name="comparison"),  # 比较图
    url(r'pointlineplot/(?P<project_id>\d+)/$', views.Pointlineplot.as_view(), name="plp"),  # 点线图
    url(r'dist_plot/(?P<project_id>\d+)/$', views.DistPlot.as_view(), name="dist"),  # 频率分布直方图
]
# 1.分类排序饼图 2.比较图  3. 点线图 4.分类散点图5.堆积柱状图其他的放后面