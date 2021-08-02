import json
import uuid
import hashlib
import pandas as pd
import datetime
import json
import random
import re
import urllib.request
import os
from django.contrib.auth import authenticate
from dateutil.relativedelta import relativedelta

from django import http
from django.shortcuts import render, redirect
# from django.shortcuts import redirect
from django.urls import reverse
from django.views import View
from django_redis import get_redis_connection
from Celery.Sms.sms import send_sms
# from utils import constants
# from libs.aliyun.sms.sms import sms_app
from libs.captcha.captcha import captcha
from utils import constants
# from utils.emails import Email
from utils.response_code import RETCODE
from dateutil import rrule
from django import http
from django.http import FileResponse
from django.views import View
from django_redis import get_redis_connection
from jsonschema.compat import urlopen
from urllib.request import Request
import requests
from rest_framework_jwt.settings import api_settings
from django.db import connection,transaction
from django.db.models import  Q
from Intelligent.settings import AppID, AppSecret
from apps.index_qt.models import Browsing_process, File_old, User, Analysis, LoginHistory
from libs.captcha.captcha import captcha
from alipay import AliPay
from Intelligent import settings
from utils.views import LoginRequiredJSONMixin
from apps.index_qt.models import Payment, OrderInfo,Member
from django.shortcuts import render
from django.shortcuts import render_to_response
from dateutil.tz import UTC
from django.http import HttpResponseRedirect
from django.contrib.auth.views import login
from django.contrib.auth.hashers import make_password, check_password
from rest_framework.views import APIView
from apps.index_qt.jwt_token import MyBaseAuthentication
from django.http import HttpResponse, JsonResponse
from libs.wx_utils import create_orderId, get_code_url, create_image, trans_xml_to_dict, get_sign, send, wxpay, trans_dict_to_xml
from libs.get import get_s, loop_add, write, read, get_path, filtering_view
def page_not_found(request):
    # return render_to_response('index/404.html')
    return render(request, 'index/new_index.html')

def page_error(request):
    return render_to_response('index/new_index.html')
    

# def page_not_found(request):
#     from django.shortcuts import render_to_response
#     response = render_to_response('index/404.html', {})
#     response.status_code = 404
#     return response
 
# def page_error(request):
#     from django.shortcuts import render_to_response
#     response = render_to_response('index/500.html', {})
#     response.status_code = 500
#     return response
    
class ImageCodeView(View):
    """图形验证码"""

    def get(self, request, uuid):
        """
        :param request: 请求对象
        :param uuid: 唯一标识图形验证码所属于的用户
        :return: image/jpeg
        """
        # 生成图片验证码
        _, text, image = captcha.generate_captcha()

        # 保存图片验证码
        redis_conn = get_redis_connection('code')
        redis_conn.setex('img_%s' % uuid, 120, text)
        # 响应图片验证码
        return http.HttpResponse(image, content_type='image/jpeg')
        

def get_access_token():
    access_token = ''
    try:
        # if cache.has_key('access_token') and cache.get('access_token') != '':
        #     access_token = cache.get('access_token')
        #     logging.critical('cache access_token:'+access_token)
        # else:
            appId = 'wxffa1c117af3308dc'
            appSecret = '15971c954ee5745cd8c0e5410f44baa6'
            postUrl = "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=%s&secret=%s" % (appId, appSecret)
            # print(postUrl)
            urlResp = urllib.request.urlopen(postUrl).read()
            urlResp = json.loads(urlResp)
            # print(urlResp)    
            access_token = urlResp['access_token']
            # print(access_token)
            # cache.set('access_token', access_token, 60 * 100)
            #leftTime = urlResp['expires_in']
    except Exception as e:
        # logging.critical(e.message)
        print(e)
    return access_token

def get_qr_ticket(login_code):
    ticket = ''
    try:
#         if cache.has_key('ticket') and cache.get('ticket') != '':
#             ticket = cache.get('ticket')
#             logging.critical('cache ticket:'+ticket)
#         else:   
        token = get_access_token()
        # logging.critical('get_access_token for ticket:'+token)
        data = {
            'expire_seconds': 604800,
            'action_name'   :'QR_STR_SCENE',
            'action_info'   : {
                'scene'     : {
                    'scene_str' : login_code
                }
        }}
        
        import requests as reqs
        params = json.dumps(data)
        if token != '' :
            ticket_url = "https://api.weixin.qq.com/cgi-bin/qrcode/create?access_token={}".format(token)
            response = reqs.post(url=ticket_url, data=params)
            #response = reqs.urlopen(req).read()
            #get_qr_ticket = urllib.urlopen(ticket_url)
            #urlResp = get_qr_ticket.read().decode("utf-8")
            # logging.critical(response.content)
            js_ticket = json.loads(response.content)
            ticket = js_ticket.get("ticket")
    except Exception as e:
        return ''
    return ticket


class WexinUrl(View):
    def get(self, request):
        STATE = str(uuid.uuid1())
        STATE = STATE.replace('-','')
        myurl = 'https://open.weixin.qq.com/connect/qrconnect?appid=wx7b031c4ac82914c6&redirect_uri=https://www.xsmartanalysis.com/&response_type=code&scope=snsapi_login&state='+STATE
        # myurl = 'https://open.weixin.qq.com/connect/qrconnect?appid=wx7b031c4ac82914c6&redirect_uri=http://znceshi.8dfish.vip/&response_type=code&scope=snsapi_userinfo&state='+STATE+'#wechat_redirect'
        # myurl = 'https://open.weixin.qq.com/connect/oauth2/authorize?appid=wxffa1c117af3308dc&redirect_uri=http://znceshi.8dfish.vip/&response_type=code&scope=snsapi_userinfo&state='+STATE+'#wechat_redirect'
        # 其实你主页上 不需要写 点击微信登录的接口 点击放个a标签 加个点击事件 点击了 就展示二维码 扫码的时候 WeXinLogin()
        # return HttpResponseRedirect('w')
        # a = get_access_token()
        # b = get_qr_ticket(STATE)
        
        # myurl = 'https://mp.weixin.qq.com/cgi-bin/showqrcode?ticket='+str(b)
        return http.JsonResponse({'code': 200, 'weixin_url': myurl})

class WeXinLogin(View):
    def get(self, request):
        wechat_data = request.GET
        signature = wechat_data['signature']
        timestamp = wechat_data['timestamp']
        nonce = wechat_data['nonce']
        echostr = wechat_data['echostr']
        token = 'znfxtoken'
     
        check_list = [token, timestamp, nonce]
        check_list.sort()
        s1 = hashlib.sha1()
        s1.update(''.join(check_list).encode())
        hashcode = s1.hexdigest()
        # print("handle/GET func: hashcode, signature:{0} {1}".format(hashcode, signature))
        if hashcode == signature:
            return HttpResponse(echostr)
        else:
            return HttpResponse("")
    def post(self, request):
        # 公众号登录接口
        # ref = request
        # json_data = json.loads(request.body.decode())
        # # price = json_data.get('price')
        # code = json_data.get('code')  # 获取code值
        # state = json_data.get("state") # 默认就是STATE
        # code = request.GET.get('openid')
        # bc_data = {}
        # try:
        #     token = get_access_token()
        #     # logging.critical('get_access_token for ticket:'+token)
            
        #     import requests as reqs
        #     if token != '' :
        #         ticket_url = "https://api.weixin.qq.com/cgi-bin/user/info?access_token={}&openid={}&lang=zh_CN".format(token,code)
        #         response = reqs.post(url=ticket_url)
        #         # logging.critical('get_wx_userinfo:'+response.content)
        #         bc_data = json.loads(response.content)
        # except Exception as e:
        #     print(e)
        #     return ''
        # # 保存注册数据
        # # print(bc_data)
        # userunionid = bc_data["openid"]
        # usernickname = bc_data["nickname"]
        # userheadimgurl = bc_data["headimgurl"]
        # # userunionid = bc_data["unionid"]
        # count = User.objects.filter(userunionid=userunionid).count()
        # password_user = userunionid[0:10]
        # # print('密码：'+password_user)s
        # new_password = make_password(password_user)
        # # print(count)
        # if count == 0:
        #     try:
        #         user = User.objects.create(username=usernickname, password=new_password, dft_file='',image_tou=userheadimgurl,userunionid=userunionid)
        #         user.save()
        #         grade = Member.objects.create(user_id=user.id,member_type_id=1)
        #         grade.save()
        #     except Exception as e:
        #         print(e)
        #         return http.JsonResponse({'code': 1009, 'error': '登录失败'}) 
        # else:
        #     user = User.objects.filter(userunionid=userunionid)
        #     user = user[0]
        # print('-------------------------11111111111111')
        # print(user)
        # login(request, user)
        # print(request.user)
        # # 微信扫码登录后 jwt_token的返回 当然 你们没有限制 可以删除掉
        # jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        # jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        # jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
        # payload = jwt_payload_handler(user)
        # token = jwt_encode_handler(payload)
        # print(777777777777777777)
        # response = redirect('http://znceshi.8dfish.vip/')
            
        # # print(response)

        # # 登录时用户名写入到cookie，有效期15天
        # name = json.dumps(user.username)
        # print(name)
        # print(token)
        # response.set_cookie('username', name, max_age=3600 * 24 * 15)
        # response.set_cookie('token', token, max_age=3600 * 24 * 15)
        # response.set_cookie('uuid', str(user.id), max_age=3600 * 24 * 15)
        # response.set_cookie('avatar', str(user.image_tou), max_age=3600 * 24 * 15)
        # print(response)
        # return response
        # return HttpResponse("")
        # return http.JsonResponse({'code': 200, 'username': usernickname,"token":token,'uuid':user.id,'avatar':user.image_tou})
    
 
        
        
        json_data = json.loads(request.body.decode())
        # price = json_data.get('price')
        code = json_data.get('code')  # 获取code值
        state = json_data.get("state") # 默认就是STATE
        # 微信登录接口
        # state = request.GET.get("state")
        print(code)
        print(2222222222222222)
        AppID = 'wx7b031c4ac82914c6'    
        AppSecret = '88065e5ffdf0f882f651b8fb8a7ab846'
        myurl = "https://api.weixin.qq.com/sns/oauth2/access_token?appid=" + AppID + "&secret=" + AppSecret + "&code=" + code + "&grant_type=authorization_code"  # 拼接地址 获取access_token值
        # 注意上面的myurl 就是微信给我们的 是哪位用户扫码的access_token 获取到access_token 才能拿到用户信息

        a = Request(myurl)
        html = urlopen(a)
        # 获取数据
        data = html.read()
        strs = json.loads(data.decode())
        mytoken = strs["access_token"]
        myrhtoken = strs["refresh_token"]
        myopenid = strs["openid"]
        myunionid = strs["unionid"]
        # 获取用户信息的api接口
        mytwourl = "https://api.weixin.qq.com/sns/userinfo?access_token=" + mytoken + "&openid=" + myopenid + "&lang=zh_CN"
        # 注意 上述mytwourl 就是获取扫码用户的信息
        b = Request(mytwourl)
        html2 = urlopen(b)
        data2 = html2.read()
        strs2 = json.loads(data2.decode())
        # b = requests.get(mytwourl)
        # strs2 = b.json()
        # d = json.dumps(strs2,ensure_ascii=False)
        # print(d)
        # strs2 = json.loads(data2.decode())
        # 下面是获取扫码者微信帐号完整信息
        print(strs2)
        print('-'*50)
        useropenid = strs2["openid"]
        usernickname = strs2["nickname"]
        userheadimgurl = strs2["headimgurl"]
        userunionid = strs2["unionid"]
        # print(userunionid)
        # print(usernickname)
        # print(userheadimgurl)
        # 保存注册数据
        count = User.objects.filter(userunionid=userunionid).count()
        password_user = userunionid[0:10]
        print('密码：'+password_user)
        new_password = make_password(password_user)
        print(count)
        if count == 0:
            try:
                time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                user = User.objects.create(username=usernickname, password=new_password, dft_file='',image_tou=userheadimgurl,userunionid=userunionid,last_login = time_current)
                user.save()
                grade = Member.objects.create(user_id=user.id,member_type_id=1)
                grade.save()
                time_current_01 = datetime.datetime.now().strftime('%Y-%m-%d')
                # print(time_current_01)
                analysis_cout =Analysis.objects.filter(analysis_time__contains=time_current_01).count()
                if analysis_cout == 0:
                    sheet = Analysis.objects.create(analysis_time = time_current,login_total=1,total_total=1)
                    sheet.save()
                else:
                    sheet_count = Analysis.objects.get(analysis_time__contains=time_current_01)
                    # sheet_count = sheet_count[0]
                    num = sheet_count.login_total
                    sheet_count.login_total = num+1
                    num01 = sheet_count.total_total
                    sheet_count.total_total = num01+1
                    sheet_count.save()
                # 登陆人数
                login_count = LoginHistory.objects.create(login_time = time_current,user_id=user.id)
                login_count.save()
            except Exception as e:
                print(e)
                return http.JsonResponse({'code': 1009, 'error': '登录失败'}) 
        else:
            
            time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            user = User.objects.filter(userunionid=userunionid)
            user = user[0]
            user.last_login =time_current
            user.save()
            time_current_01 = datetime.datetime.now().strftime('%Y-%m-%d')
            print(time_current_01)
            analysis_cout =Analysis.objects.filter(analysis_time__contains=time_current_01).count()
            if analysis_cout == 0:
                sheet = Analysis.objects.create(analysis_time = time_current,login_total=1)
                sheet.save()
            else:
                sheet_count = Analysis.objects.get(analysis_time__contains=time_current_01)
                # sheet_count = sheet_count[0]
                num = sheet_count.login_total
                sheet_count.login_total = num+1
                sheet_count.save()
            # 登陆人数
            login_count = LoginHistory.objects.create(login_time = time_current,user_id=user.id)
            login_count.save()

        # 微信扫码登录后 jwt_token的返回 当然 你们没有限制 可以删除掉
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
        payload = jwt_payload_handler(user)
        token = jwt_encode_handler(payload)
        return http.JsonResponse({'code': 200, 'username': usernickname,"token":token,'uuid':user.id,'avatar':user.image_tou})

    

class WexinToken(APIView):
    def get(self, request):
        wechat_data = request.GET
        signature = wechat_data['signature']
        timestamp = wechat_data['timestamp']
        nonce = wechat_data['nonce']
        echostr = wechat_data['echostr']
        token = 'znfxtoken'
     
        check_list = [token, timestamp, nonce]
        check_list.sort()
        s1 = hashlib.sha1()
        s1.update(''.join(check_list).encode())
        hashcode = s1.hexdigest()
        print("handle/GET func: hashcode, signature:{0} {1}".format(hashcode, signature))
        if hashcode == signature:
            return HttpResponse(echostr)
        else:
            return HttpResponse("")
    
    def post(self, request):
        print(222222)
        pass
    # def check(request):



class PaymentView(APIView):
    """订单支付功能"""
    
    authentication_classes = [MyBaseAuthentication, ]
    def get(self, request):
        # 获取前端传入的请求参数
        query_dict = request.GET
        data = query_dict.dict()
        # print(data)
        # 检查当前用户的等级
        vip_grade = Member.objects.get(user=request.user)
        # print(11111)
        # 校验会员时间
        # member_check(member_num)
        # 当前用户等级
        vip_uer = int(vip_grade.member_type.id)
        if data:
            # 获取并从请求参数中剔除signature
            signature = data.pop('sign')
    
            # 创建支付宝支付对象
            alipay = AliPay(
                appid=settings.ALIPAY_APPID,
                app_notify_url=None,  # 默认回调url
                app_private_key_path=settings.APP_PRIVATE_KEY_PATH,
                alipay_public_key_path=settings.ALIPAY_PUBLIC_KEY_PATH,
                sign_type="RSA2",
                debug=settings.ALIPAY_DEBUG
            )
    
            # 校验这个重定向是否是alipay重定向过来的
            success = alipay.verify(data, signature)
            if success:
                print(data)
                # 读取order_id
                order_id = data.get('out_trade_no')
                # 读取支付宝流水号
                trade_id = data.get('trade_no')
                # print(11111111111111111111)
                print(order_id)
                print(trade_id)
                # print(222222222222222222)

                # 修改订单状态为待评价
                OrderInfo.objects.filter(order_id=order_id, status=OrderInfo.ORDER_STATUS_ENUM['UNCOMMENT']).update(
                    status=OrderInfo.ORDER_STATUS_ENUM["FINISHED"])
                order_info = OrderInfo.objects.get(order_id=order_id)
                # 保存Payment模型类数据
                Payment.objects.create(
                    order_user = request.user,
                    trade_id=trade_id,
                    order_payment = order_info
                )
                # 将用户级别提升
                # 查询用户充值会员类型
                user_member = order_info.member_type_id
                user_member_save = Member.objects.get(user=request.user.id)
                # 要充值的时间（天）（月）（年）
                member_day = order_info.total_amount / order_info.unit_price
                print(member_day)
                # 充值v1会员
                if int(user_member) == 2:
                    user_member_save.member_type_id = 2
                    # 充值当前时间
                    time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 充值后的时间
                    time_last = (datetime.datetime.now() + datetime.timedelta(days = int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
                # 充值v2会员
                if int(user_member) == 3:
                    user_member_save.member_type_id = 3
                    # 充值当前时间
                    time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 充值后的时间
                    time_last = (datetime.datetime.now() + relativedelta(months=+int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
                # 充值v3会员
                if int(user_member) == 4:
                    user_member_save.member_type_id = 4
                    # 充值当前时间
                    time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 充值后的时间
                    time_last = (datetime.datetime.now() + relativedelta(years=+int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
                # 充值年费会员
                
                # 保存到会员表
                user_member_save.member_initial_time = time_current
                user_member_save.member_last_time = time_last
                user_member_save.save()
                
                # 响应trade_id
                context = {
                    'trade_id': trade_id,
                    'vip_grade':vip_uer
                }
                return render(request, 'index/chongzhi.html', context)
            else:
                context ={
                    'vip_grade':vip_uer
                }
                # 订单支付失败，重定向到我的订单
                return render(request, 'index/chongzhi.html',context)
        context = {
            'vip_grade':vip_uer,
            'u_id':vip_uer
        }
        return render(request, 'index/chongzhi.html',context)

    def post(self, request):
        # 查询要支付的订单
        user = request.user
        json_data = json.loads(request.body.decode())
        price = json_data.get('price')
        pay_method = json_data.get('pay_method')
        member_num = json_data.get('member')
        name = json_data.get('name')
        # 获取单价
        single_price = json_data.get('single_price')
        # 获取数量
        count = json_data.get('count')
        # 计算价格
        total_price = float(count) * float(single_price)
        print(total_price != price)
        if total_price != price:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        # 校验是否合法
        if name == 'v1' and total_price < 9.9:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        elif name == 'v2' and total_price < 190:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        elif name == 'v3' and total_price < 1200:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        else:
            pass
        if not price:
            return http.JsonResponse({'code': 1001, 'errmsg': '请选择充值的类型'})
        if int(pay_method) not in [OrderInfo.PAY_METHODS_ENUM['CASH'], OrderInfo.PAY_METHODS_ENUM['ALIPAY']]:
            return http.JsonResponse({'code': 1002, 'errmsg': '支付方式错误'})
        print(type(price))
        price = float(price)
        # 创建支付宝支付对象
        alipay = AliPay(
            appid=settings.ALIPAY_APPID,
            app_notify_url=None,  # 默认回调url
            app_private_key_path=settings.APP_PRIVATE_KEY_PATH,
            alipay_public_key_path=settings.ALIPAY_PUBLIC_KEY_PATH,
            sign_type="RSA2",
            debug=settings.ALIPAY_DEBUG
        )

        # 随机生成一个订单号
        id = request.user.id
        str01 = str(uuid.uuid1())
        
        # 订单号
        order_number = str01.replace('-', '')
        print(order_number)
        # 生成登录支付宝连接
        order_string = alipay.api_alipay_trade_page_pay(
            out_trade_no=order_number,
            total_amount=str(price),
            subject="极智分析会员充值",
            return_url=settings.ALIPAY_RETURN_URL,
        )
        # 检查当前用户的等级
        vip_grade = Member.objects.get(user=request.user)
        print(11111)
        # 当前用户等级
        vip_uer = int(vip_grade.member_type.id)
        # if price == 98 and vip_uer != 1:
        
        # ˉv1会员等级
        # if vip_grade.
            
        # 保存订单
        # 显式的开启一个事务
        with transaction.atomic():
            # 创建事务保存点
            save_id = transaction.savepoint()
            try:
                order = OrderInfo.objects.create(user=request.user,order_id=order_number,total_amount=price,pay_method=pay_method,status=OrderInfo.ORDER_STATUS_ENUM['UNCOMMENT'],unit_price=single_price,member_type_id=int(member_num))
            except Exception as e:
                print(e)
                # 事务回滚  
                transaction.savepoint_rollback(save_id)
                return http.JsonResponse(
                    {'code': 2005, 'error': '订单核对失败，请稍后再试', 'alipay_url': 'https://www.xsmartanalysis.com/recharge/'})
            # 提交订单成功，显式的提交一次事务
            transaction.savepoint_commit(save_id)
            
        # 响应登录支付宝连接
        # 真实环境电脑网站支付网关：https://openapi.alipay.com/gateway.do? + order_string
        # 沙箱环境电脑网站支付网关：https://openapi.alipaydev.com/gateway.do? + order_string
        alipay_url = settings.ALIPAY_URL + "?" + order_string
        return http.JsonResponse({'code': 200, 'error': 'OK', 'alipay_url': alipay_url})


class showMembersView(APIView):
    """查询会员等级以及天数"""
    
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        # 获取当前用户
        user = request.user
        try:
            member_num = Member.objects.get(user=request.user.id)
        except Exception as e:
            print(e)
        # 检查当前用户的等级
        u_id = member_num.member_type.id
        if u_id != 1:
            # 会员的初始时间
            start_time = member_num.member_initial_time
            time01 = datetime.datetime.now(UTC)
            print(type(datetime.datetime.now()))
            print(type(start_time))
            #  查询当前会员的到期时间
            end_time = member_num.member_last_time
            # 计算剩余时间
            try:
                weeks = rrule.rrule(rrule.DAILY, dtstart=time01, until=end_time)
            except Exception as e:
                print(e)
            # 返回剩余时间
            limit_time = weeks.count()-1
        
            context = {
                'grade':u_id,
                'limit_time':limit_time
            }
        else:
             context = {
                'grade':u_id,
                # 'limit_time':limit_time
            }
        return http.JsonResponse({'code': 0, 'error': '查询成功','context':context})

class SMSCodeView(View):
    """短信验证码"""
    
    def get(self, reqeust, mobile):
        """
        :param reqeust: 请求对象
        :param mobile: 手机号
        :return: JSON
        """
        
        # 创建连接到redis的对象
        redis_conn = get_redis_connection('code')
        send_sms_flag = redis_conn.get('sms_%s' % mobile)
        # print(send_sms_flag)
        if send_sms_flag:
            return http.JsonResponse({'code': RETCODE.THROTTLINGERR, 'errmsg': '发送短信过于频繁'})
        
        # 生成短信验证码：生成6位数验证码
        sms_code = '%06d' % random.randint(0, 999999)
        
        # 保存短信验证码
        redis_conn.setex('sms_%s' % mobile, constants.SMS_CODE_REDIS_EXPIRES, sms_code)
        # 为了在测试期间减少短信发送，可以屏蔽发送短信功能，使用打印的方式来获取到真正的验证码值
        # 发送短信验证码
        send_sms(mobile, sms_code)
        # sms_app.send_task('sms.send_sms', args=[mobile, sms_code])
        print(">>>>短信验证码>>>>>>", sms_code)
        
        # 响应结果
        return http.JsonResponse({'code': RETCODE.OK, 'errmsg': '发送短信成功'})
        
class WechatPaymentView(APIView):
    """微信支付二维码的类"""
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        json_data = json.loads(request.body.decode())
        price = json_data.get('price')
        single_price = json_data.get('single_price')
        name = json_data.get('name')
        # 获取数量
        count  = json_data.get('count')
        # 计算价格
        total_price = float(count) * float(single_price)
        print(total_price != price)
        if total_price != price:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        # 校验是否合法
        if name == 'v1' and total_price < 9.9:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        elif name == 'v2' and total_price < 190:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        elif name == 'v3' and total_price < 1200:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        else:
            pass
        member_num = None
        if name == 'v1':
            member_num = 2
        elif name == 'v2':
            member_num = 3
        elif name == 'v3':
            member_num = 4
        else:
            return http.JsonResponse({'code': 500, 'error': '二维码失效，请稍后再试,非法请求。'})
        price_wx = int(price *100)
        body = 'pay'  # 商品描述
        total_fee = price_wx  # 付款金额，单位是分，必须是整数
        out_trade_no = create_orderId(5) # 这是一个自定义的函数 目的是生成订单号
        data_dict = wxpay(body, total_fee, out_trade_no)  # 这是一个自定义的函数
        if data_dict.get('return_code') == 'SUCCESS':
            code_url = get_code_url(data_dict) # 这是一个自定义函数,目的是获取支付二维码的地址
            img = create_image(code_url) # 创建支付二维码的图片
            qrcode_name = out_trade_no + 'wxpay.png' # 订单号加后缀 作为图片
            img.save(r'static' + '/wxpay/' + qrcode_name) # 保存图片
            savepath = '/static/wxpay/'+qrcode_name
            # 创建订单
            # 保存订单
            # 显式的开启一个事务
            with transaction.atomic():
                # 创建事务保存点
                save_id = transaction.savepoint()
                try:
                    order = OrderInfo.objects.create(user=request.user,order_id=out_trade_no,total_amount=price,pay_method=1,status=OrderInfo.ORDER_STATUS_ENUM['UNCOMMENT'],unit_price=single_price,member_type_id=int(member_num))
                except Exception as e:
                    print(e)
                    # 事务回滚  
                    transaction.savepoint_rollback(save_id)
                    return http.JsonResponse(
                        {'code': 2005, 'error': '订单核对失败，请稍后再试', 'alipay_url': 'https://www.xsmartanalysis.com/recharge/'})
                # 提交订单成功，显式的提交一次事务
                transaction.savepoint_commit(save_id)
            return http.JsonResponse({'code': 200, 'error': 'OK', 'qrcode_name': savepath,'out_trade_no':out_trade_no}) # 前端获取返回的订单号 轮询订单是否支付完成
            # return render(request, 'index/chongzhi.html', {'qrcode_name': savepath, 'out_trade_no': out_trade_no}) # 前端获取返回的订单号 轮询订单是否支付完成
        # return render(request, 'index/chongzhi.html', {'err_msg': '二维码失效，请稍后再试'})
        return http.JsonResponse({'code': 200, 'error': '二维码失效，请稍后再试'})

class WXPaymentView(APIView):
    """微信支付的类"""
    # authentication_classes = [MyBaseAuthentication, ]
    def post(self,request):
        data_dict = trans_xml_to_dict(request.body)  # 回调数据转字典
        # print('支付回调结果', data_dict)
        sign = data_dict.pop('sign')  # 取出签名
        back_sign = get_sign(data_dict, '16073862171607386217160738621716')  # 计算签名
        # 验证签名是否与回调签名相同
        if sign == back_sign and data_dict['return_code'] == 'SUCCESS':
            '''
            检查对应业务数据的状态，判断该通知是否已经处理过，如果没有处理过再进行处理，如果处理过直接返回结果成功。
            '''
            print('微信支付成功会回调！')
            # print(data_dict)
            # print(data_dict['out_trade_no'])
            order_wx = OrderInfo.objects.get(order_id =data_dict['out_trade_no'])
            # print(order_wx)
            user_member = order_wx.member_type_id # 充值会员类型
            total_amount = order_wx.total_amount # 总价
            unit_price = order_wx.unit_price
            member_day = int(total_amount // unit_price)
            user_id = order_wx.user_id
            print(user_id)
            user_member_save = Member.objects.get(user_id=user_id)
            # 充值v1会员
            if int(user_member) == 2:
                user_member_save.member_type_id = 2
                # 充值当前时间
                time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 充值后的时间
                time_last = (datetime.datetime.now() + datetime.timedelta(days = int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
            # 充值v2会员
            if int(user_member) == 3:
                user_member_save.member_type_id = 3
                # 充值当前时间
                time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 充值后的时间
                time_last = (datetime.datetime.now() + relativedelta(months=+int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
            # 充值v3会员
            if int(user_member) == 4:
                user_member_save.member_type_id = 4
                # 充值当前时间
                time_current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 充值后的时间
                time_last = (datetime.datetime.now() + relativedelta(years=+int(member_day))).strftime('%Y-%m-%d %H:%M:%S')
            # 处理支付成功逻辑，向前端页面发送实时消息
            # out_trade_no = data_dict['out_trade_no']
            # 保存到会员表
            user_member_save.member_initial_time = time_current
            user_member_save.member_last_time = time_last
            user_member_save.save()
            order_wx.status = 1
            order_wx.save()
            # print(3333333333333)
            # print(data_dict)
            """
            此处可做支付成功后的业务逻辑
            """
            # send(out_trade_no, '支付成功') # 返回前端的消息
            # 返回接收结果给微信，否则微信会每隔8分钟发送post请求
            return HttpResponse(trans_dict_to_xml({'return_code': 'SUCCESS', 'return_msg': 'OK'}))
        return HttpResponse(trans_dict_to_xml({'return_code': 'FAIL', 'return_msg': 'SIGNERROR'}))


class Monitor(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        json_data = json.loads(request.body.decode())
        order = json_data.get('order')
        try:
            order_wx = OrderInfo.objects.get(order_id =order)
        except Exception as e:
            print(e)
            return http.JsonResponse({'code': 500, 'error': '无效的订单号'})
        status = str(order_wx.status)
        return http.JsonResponse({'code': 200, 'stat': status})

class WXimg(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        json_data = json.loads(request.body.decode())
        order = json_data.get('order')
        qrcode_name = order + 'wxpay.png' # 订单号加后缀 作为图片
        try:
            project_path=get_path(0)  # 得上一层路径  /www/wwwroot/znfxpt.ceshi.vip/ini/Intelligent
            wx_img = str(project_path)+'/static/wxpay/'+qrcode_name
            print(wx_img)
            os.remove(wx_img)
        except Exception as e:
            print(e)
        # img.save(r'static' + '/wxpay/' + qrcode_name) # 保存图片
        return http.JsonResponse({'code': 200, 'error':'删除成功'}) 
class Login_WX(View):
    # authentication_classes = [MyBaseAuthentication, ]
    def post(self, request):
        json_data = json.loads(request.body.decode())
        duration = json_data.get("time")
        try:
            time_current = datetime.datetime.now().strftime('%Y-%m-%d')
            time_endByWeekAgo = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            time_endByMonthAgo = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            sql_sentence = "select date_format(analysis_time,'%%Y-%%m-%%d')as date,login_total,total_total,real_number from tb_user_analysis where date_format(analysis_time,'%%Y-%%m-%%d')>%s" if duration!='all' else "select date_format(analysis_time,'%Y-%m-%d')as date,login_total,total_total,real_number from tb_user_analysis "
            params = [time_endByMonthAgo] if duration=='month' else [time_endByWeekAgo]
            cursor = connection.cursor()
            records = cursor.execute(sql_sentence, params)  if  duration!='all' else cursor.execute(sql_sentence)
            data = cursor.fetchall()
            resultData = []
            dict = {}
            parseArgs = ['date', 'loginNum', 'registerNum','real_number']
            for i in data:
                for index in range(len(i)):
                    dict[parseArgs[index]] = i[index]
                resultData.append(dict)
                dict={}
            #sql_area="select count(country) as areaNum,SUBSTRING(country,1,2) as country from tb_users where country is not null and country <> '' and country <> '未知'  group by  SUBSTRING(country,1,2)"
            sql_area = "select areaNum,case country when '英格' then '英格兰' else country end as country  from(" \
                       "select count(country) as areaNum,SUBSTRING(country,1,2) as country from tb_users where country is not null and country <> '' and country <> '未知'  group by  SUBSTRING(country,1,2))As A"
            cursor.execute(sql_area)
            areaResult=cursor.fetchall()
            areaResultData = []
            areadict = {}
            parseArgs = ['value', 'name']
            for i in areaResult:
                for index in range(len(i)):
                    areadict[parseArgs[index]] = i[index]
                areaResultData.append(areadict)
                areadict = {}
        except Exception as e:
           print(e)
        return http.JsonResponse({'code': 200, 'error': '查询成功','resultData':resultData,'areaResultData':areaResultData})

class UsernameMobileView(APIView):
    """判断用户名是否有手机号"""

    def post(self, request):
        """
        :param request: 请求对象
        :param username: 用户名
        :return: JSON
        """
        print(request.body)
        json_data = json.loads(request.body.decode())
        print(json_data)
        username = json_data.get('username')
        try:
            print(username)
            count = User.objects.get(username=username)
            print(count.id)
            data = count.mobile
            print(data)
            if data == "":
                data = None
        except Exception as e:
            print(e)
            data = None
        
        return http.JsonResponse({'code': 200, 'errmsg': 'OK', 'mobile': data})

class BDSMS(APIView):
    authentication_classes = [MyBaseAuthentication, ]
    def post(self,request):
        json_data = json.loads(request.body.decode())
        sms_code = json_data.get('sms')
        username = json_data.get('username')
        phone = json_data.get('mobile')
        # 从cookie中获取当前用户
        redis_conn = get_redis_connection('code')
        sms_code_saved = redis_conn.get('sms_%s' % phone)
        if sms_code_saved is None:
            return http.JsonResponse({'code': 1009, 'error': '无效的短信验证码'}) 
        if sms_code != sms_code_saved.decode():
            return http.JsonResponse({'code': 1010, 'error': '输入短信验证码有误'}) 
        try:
            user = User.objects.get(username=username)
            
        except Exception as e:
            return http.JsonResponse({"code": 0, "errmsg": "用户不存在"})
        try:
            user.mobile = phone
            user.save()
        except Exception as e:
            print(e)
            return http.JsonResponse({"code": 0, "errmsg": "修改失败"})

        return http.JsonResponse({'code': 200, 'errmsg': '修改成功'})
