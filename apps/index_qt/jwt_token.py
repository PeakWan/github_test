from rest_framework.authentication import BaseAuthentication
from rest_framework.permissions import BasePermission
from rest_framework_jwt.authentication import jwt_decode_handler
from rest_framework.exceptions import AuthenticationFailed
import jwt
from apps.index_qt.models import User
import re
from django.shortcuts import render, redirect
from django.urls import reverse

class MyBaseAuthentication(BaseAuthentication):
    # 重写authenticate
    def authenticate(self, request):
        # 获取token的第二部分
        # print(request.META)
        try:
            jwt_value = request.META.get('HTTP_AUTHORIZATION')  # 测试环境下
            jwt_value = str(jwt_value).split()[1]  # 测试环境下
        except:
            jwt_value = request.META.get('HTTP_COOKIE')  # 正式开发下
            a =jwt_value.split(';')
            b = None
            for i in a:
                if 'token' in i:
                    b = i
            # print(b)
            jwt_value = b.split('=')
            jwt_value = jwt_value[1]
        if not jwt_value:
            raise AuthenticationFailed({'code':'401', 'message': '请先登录认证!'})
        try:
            payload = jwt_decode_handler(jwt_value)
        except jwt.ExpiredSignature:
            raise AuthenticationFailed({'code': '401', 'message': '登录超时,请重新登录!'})
        except jwt.InvalidTokenError:
            raise AuthenticationFailed({'code': '401', 'message': '非法用户,请确认是否登录!'})

        except Exception as e:
            raise AuthenticationFailed(str(e))
        user = User.objects.filter(username=payload.get('username')).first()
        return user,jwt_value