from django.conf.urls import url
from apps.help_document import views

urlpatterns = [
    
    url(r'^index/$', views.HelpIndexView.as_view(), name="index"),  # 视频教程主页
    url(r'^comment/$', views.userComment.as_view(), name="comment"),  # 评论
    url(r'^comment_son/(?P<comment_id>\d+)/$', views.userCommentSon.as_view(), name="comment_son"),  # 评论
    url(r'^comment_data/$', views.userCommentData.as_view(), name="comment_data"),  # 评论

]
