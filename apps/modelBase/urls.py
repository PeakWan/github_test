from django.conf.urls import url

from apps.modelBase import views

urlpatterns = [

    url(r'^update/$', views.ModelUpdate.as_view(), name="update"), # 模型上传列表
    url(r'^index/$', views.ModelIndex.as_view(), name="index"), # 用户模型首页
    url(r'^list/$', views.MemberList.as_view(), name="list"), # 模型列表页面
    url(r'^views/$', views.ModelViews.as_view(), name="views"), # 模型详情页面
    url(r'^data/$', views.ModelData.as_view(), name="data"), # 
    url(r'^delete/$', views.ModelDelete.as_view(), name="delete"), # 删除
    url(r'^update_img/$', views.UpdateIMG.as_view(), name="update_img"), # 删除
    url(r'^delete_img/$', views.DeleteIMG.as_view(), name="delete_img"), # 删除 ModelUpdate
    url(r'^edit/$', views.ModelEdit.as_view(), name="edit"), # 删除 ModelUpdate
]