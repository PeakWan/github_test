"""
Django settings for Intelligent project.

Generated by 'django-admin startproject' using Django 3.0.7.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.0/ref/settings/
"""
import datetime
import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'a-^!uz2#!sn&zdt64ban7(t80khrb@zp#_a-3y!u#x^=rlp!4&'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
# TeMPLATE_DEBUG = False
# DEBUG = False
# ALLOWED_HOSTS = ['http://ncdefs.hk.8dfish.site/']
ALLOWED_HOSTS = ['*']
# 
# ALLOWED_HOSTS = ['127.0.0.1']
# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'apps.index',  # 后台主页面
    'apps.admin_meber',  # 后台会员管理
    "apps.Administrator",  # 管理员
    'apps.upload_exel',  # 上传文件
    'apps.index_qt',  # 前台主页面文件
    'apps.verifications',  # 图片验证码
    'apps.Smart',  # 智能统计分析
    'apps.chart',  # 智能图表
    'apps.Advanced',  # 高级分析
    'apps.modelBase', # 模型库
    'apps.help_document', # 帮助中心视频教程
    
]

MIDDLEWARE = [
    # 'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'Intelligent.urls'

TEMPLATES = [
    {
        # 'BACKEND': 'django.template.backends.django.DjangoTemplates',  # Django默认引擎
        'BACKEND': 'django.template.backends.jinja2.Jinja2',  # jinja2模板引擎
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            # 补充Jinja2模板引擎环境
            'environment': 'utils.jinja2_env.jinja2_environment',
        },
    },
]
WSGI_APPLICATION = 'Intelligent.wsgi.application'

# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',  # 数据库引擎
        'HOST': '47.92.236.49',  # 数据库主机
        'PORT': 3306,  # 数据库端口
        'USER': 'znfxpt_8dfish_vi',  # 数据库用户名
        'PASSWORD': 'XLihaSLj2Ld7RR8R',  # 数据库用户密码
        'NAME': 'znfxpt_8dfish_vi',  # 数据库名字
        
        'OPTIONS': {'charset': 'utf8mb4'},
    },
}

# Password validation
# https://docs.djangoproject.com/en/3.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'
#TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.0/howto/static-files/

STATIC_URL = '/static/'
# 配置静态文件加载路径
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# 配置Redis数据库
CACHES = {
    "default": {  # 保存原始数据
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/0",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
    "session": {  # session
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
    "code": {  # 验证码
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/2",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
    "history": {  # 用户浏览记录
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/3",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
}
REST_FRAMEWORK = {
    # 引入JWT认证机制，当客户端将jwt token传递给服务器之后
    # 此认证机制会自动校验jwt token的有效性，无效会直接返回401(未认证错误)
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ),
    'PAGE_SIZE': 100,  # 显示多少数据
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',  # 防止跳出 警告
}

# JWT扩展配置
JWT_AUTH = {
    # 重新定义jwt认证成功后返回的数据 days(天) hours(时) minutes(分)
    'JWT_EXPIRATION_DELTA': datetime.timedelta(days=1),  # 定义token过期时间1天
}


SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "session"


# from django.conf.global_settings import *

# 指定本项目用户模型类
AUTH_USER_MODEL = 'index_qt.User'

# 指定自定义的用户认证后端
AUTHENTICATION_BACKENDS = [
    'apps.index_qt.utils.UsernameMobileAuthBackend'
]

# 配置当用户未登录，反而要访问某个需要登录的URL时，Django自动跳转的URL
LOGIN_URL = '/login/'


# 配置微信登录
# AppID = 'wxf2250adaf0b7712a'  # 项目的AppID
# AppSecret = '10f89090e6feacff671f534bd4ee3ff3'  # 密钥、

AppID = 'wxffa1c117af3308dc'  # 项目的AppID
AppSecret = '15971c954ee5745cd8c0e5410f44baa6'  # 密钥


# 配置RabbitMQ信息
BROKER_IP = "localhost"
BROKER_USER = "guest"
BACKEND_IP = "localhost"
# 设置一个全局变量
# SET_A = True

# 支付宝沙箱配置信息
ALIPAY_APPID = '2021002127625199'
ALIPAY_DEBUG = True
ALIPAY_URL = 'https://openapi.alipay.com/gateway.do'
ALIPAY_RETURN_URL = 'http://znceshi.8dfish.vip/recharge/'
APP_PRIVATE_KEY_PATH = os.path.join(BASE_DIR, 'apps/verifications/keys/app_private_key.pem')
ALIPAY_PUBLIC_KEY_PATH = os.path.join(BASE_DIR, 'apps/verifications/keys/alipay_public_key.pem')


APP_ID = "wxffa1c117af3308dc"  # 微信公众号ID
MCH_ID = "1607386217"  # 商户号ID
API_KEY = "16073862171607386217160738621716"  # APK密钥(在商户平台里-->>API密钥里面设置)(建议 关于密钥都设置成统一的 好记)
UFDOOER_URL = "https://api.mch.weixin.qq.com/pay/unifiedorder"  # 微信统一下单地址 默认的
# 注意 内网穿透的地址是只能维持几个小时的 如果时间到了 请自行重启
NOTIFY_URL = "http://znceshi.8dfish.vip/recharge/"  # 回调，可使用内网穿透工具进行测试
CREATE_IP = "47.92.236.49"  # 服务器的公网IP 测试使用本地


