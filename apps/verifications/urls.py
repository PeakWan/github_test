from django.conf.urls import url

from apps.verifications import views
# from apps.verifications.views import WeXinLogin

# from apps.verifications.views import *


urlpatterns = [
    url(r'^image_codes/(?P<uuid>[\w-]+)/$', views.ImageCodeView.as_view()),
    url(r'^wx/$', views.WeXinLogin.as_view()),  # 获取微信登录 code值
    url(r'^weixinlogin/$', views.WexinUrl.as_view()), # 返回登录网址 获取二维码
    url(r'^token/$', views.WexinToken.as_view()), # token
    url(r'^recharge/', views.PaymentView.as_view(), name="recharge"),# 支付宝支付
    url(r'^show_members/', views.showMembersView.as_view(), name="show_members"),# 会员等级以及天数
    url(r'^sms_codes/(?P<mobile>1[3-9]\d{9})/$', views.SMSCodeView.as_view()),  # 短信验证码
    url(r'^wechatpay/$', views.WechatPaymentView.as_view(), name='wechatpay'), # 微信支付的接口
    url(r'^wxpay/$', views.WXPaymentView.as_view(), name='wxpay'), # 微信支付的接口
    url(r'^monitor/$', views.Monitor.as_view(), name='monitor'), # 订单轮询
    url(r'^wx_img/$', views.WXimg.as_view(), name='wximg'), # 删除图片
    url(r'^login_wx/$', views.Login_WX.as_view(), name='login_wx'), # 查找每天登录的用户
    url(r'^regest_mobile/$', views.UsernameMobileView.as_view(), name='regest_mobile'), # 检测当前用户是否绑定手机号
    url(r'^binding_sms/$', views.BDSMS.as_view(), name='bangding'), # 绑定手机号吗
    # url(r'^result', views.result.as_view()), # 微信支付成功之后 回调函数
    # url(r'websocketLink/(?P<out_trade_no>\w+)', views.websocketLink.as_view())  # webSocket 链接
]