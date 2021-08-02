from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """自定义用户模型类"""
    mobile = models.CharField(max_length=11, unique=True, verbose_name='手机号', default=None)
    super_u = models.IntegerField(default=0, verbose_name='管理员')
    member = models.IntegerField(default=0, verbose_name='会员')
    dft_file = models.CharField(max_length=65535, verbose_name='默认文件名', default=None)
    image_tou = models.CharField(max_length=65535, verbose_name='头像', default=None)
    userunionid = models.CharField(max_length=3000, verbose_name='微信id', default=None)
    country = models.CharField(max_length=65535, verbose_name='地区', default='未知')

    class Meta:
        db_table = 'tb_users'
        verbose_name = '用户'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.username


class File_old(models.Model):
    """项目名称表"""

    project_name = models.CharField(max_length=65535, verbose_name='项目名称', default=None)  # 保存项目名称
    background = models.CharField(max_length=65535, verbose_name='项目背景', default=None)  # 保存项目背景
    outline = models.CharField(max_length=65535, verbose_name='项目概要', default=None)  # 保存项目概要
    file_name = models.CharField(max_length=65535, verbose_name='保存的文件名')  # 保存的文件名
    path = models.CharField(max_length=65535, verbose_name='保存文件的路径', default=None)  # 保存的文件的路径
    create_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')  # 记录的创建时间
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user', verbose_name='用户')
    last_time = models.DateTimeField(auto_now=True, verbose_name='修改时间')  # 记录的创建时间
    path_pa = models.CharField(max_length=65535, verbose_name='保存文件的路径', default=None)  # 保存的文件的路径

    class Meta:
        db_table = 'tb_file'
        verbose_name = "上传记录"


class Browsing_process(models.Model):
    """浏览记录"""

    process_info = models.CharField(max_length=65535, verbose_name='具体浏览过程', default=None)  # 保存项目名称
    create_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')  # 记录的创建时间
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_process', verbose_name='用户')
    file_old = models.ForeignKey(File_old, on_delete=models.CASCADE, related_name='project', verbose_name='项目id',default=None)
    order = models.IntegerField( verbose_name='顺序编号')  # 顺序编号
    is_delete = models.CharField(max_length=11, verbose_name='是否显示',default=0)  # 显示
    is_latest = models.IntegerField(verbose_name='是否是最新的数据',default=0)  #  是否最新的数据

    class Meta:
        db_table = 'tb_process'
        verbose_name = "浏览记录"
        

class MemberType(models.Model):
    """会员类型表"""
    member_name = models.CharField(max_length=65535, verbose_name='会员名称', default=None)
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="价格")
    number = models.IntegerField(verbose_name="分析条数")
    projects = models.IntegerField(verbose_name="项目数量",default=0)
    flow_number = models.IntegerField(verbose_name="流程数目",default=0)
    class Meta:
        db_table = 'tb_type'
        verbose_name = "会员类型表"


class Member(models.Model):
    """会员表"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_member', verbose_name='用户')
    member_type = models.ForeignKey(MemberType, on_delete=models.CASCADE, related_name='tb_type', verbose_name='会员级别')
    is_delete = models.CharField(max_length=11, verbose_name='是否删除', default=0)  # 是否删除
    member_initial_time = models.DateTimeField(verbose_name='会员初始时间',default=None) #  会员初始时间
    member_last_time = models.DateTimeField(verbose_name='会员到期时间',default=None) #  会员到期时间
    class Meta:
        db_table = 'tb_member'
        verbose_name = "会员表"
        
class Modelbase(models.Model):
    """模型库"""

    model_name = models.CharField(max_length=65535, verbose_name='模型库名称', default=None)  # 模型库名称
    model_background = models.CharField(max_length=65535, verbose_name='模型库说明', default=None)  # 模型库说明
    model_outline = models.CharField(max_length=65535, verbose_name='模型库方法', default=None)  # 模型库方法
    model_info = models.CharField(max_length=65535, verbose_name='分析结果图片', default=None)  # 分析结果图片
    model_type = models.CharField(max_length=65535, verbose_name='模型类型', default=None)  # 模型类型
    model_path = models.CharField(max_length=65535, verbose_name='保存文件的路径', default=None)  # 保存的文件的路径
    create_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')  # 记录的创建时间
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='model_user', verbose_name='用户')
    last_time = models.DateTimeField(auto_now=True, verbose_name='修改时间')  # 记录的创建时间
    versions = models.CharField(max_length=65535, verbose_name='版本数', default=None)  # 版本数
    url = models.CharField(max_length=65535, verbose_name='链接', default=None)  # 链接
    number = models.CharField(max_length=65535, verbose_name='人数', default=None)  # 模型库名
    # source_num = models.CharField(max_length=65535, verbose_name='评分平均分', default=None)  # 评分平均分
    
    class Meta:
        db_table = 'tb_model_base'
        verbose_name = "模型列表"

class Help(models.Model):
    """视频教程模型类"""
    video_name = models.CharField(max_length=65535, verbose_name='视频教程标题', default=None)  # 视频教程标题
    video_background = models.CharField(max_length=65535, verbose_name='视频教程说明', default=None)  # 视频教程说明
    video_info = models.CharField(max_length=65535, verbose_name='视频图片', default=None)  # 视频教程图片
    video_link = models.CharField(max_length=65535, verbose_name='视频链接', default=None)  # 视频教程链接
    class Meta:
        db_table = 'tb_help_video'
        verbose_name = "视频教程"

class Commits_books(models.Model):
    """定义文章用户评论模块"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='commit_library', verbose_name='用户')
    content = models.CharField(max_length=10000, verbose_name='评论内容')
    create_time = models.DateField(auto_now_add=True, verbose_name="创建时间")
    parent = models.ForeignKey('self', verbose_name='父评论', default=None, related_name='comm_subs')
    commit_username = models.CharField(max_length=10000,default=None ,verbose_name='用户名字')

    class Meta:
        db_table = 'tb_commit'
        verbose_name = '评论'

class Post_user(models.Model):
    """定义用户发布文章的表"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='post_library', verbose_name='用户')
    post_content = models.CharField(max_length=65535, verbose_name='发布内容')
    create_time = models.DateField(auto_now_add=True, verbose_name="创建时间")
    post_image = models.CharField(max_length=65535, verbose_name='发布图片')

    class Meta:
        db_table = 'tb_post'
        verbose_name = '发帖'

class Score(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='score_library', verbose_name='用户')
    modelbase = models.ForeignKey(Modelbase, on_delete=models.CASCADE, related_name='score_library', verbose_name='模型库')
    score = models.IntegerField(verbose_name="评分分数")

    class Meta:
        db_table = 'tb_score'
        verbose_name = '评分'
        
class OrderInfo(models.Model):
    """订单信息"""
    PAY_METHODS_ENUM = {
        "CASH": 1,
        "ALIPAY": 2
    }
    PAY_METHOD_CHOICES = (
        (1, "微信支付"),
        (2, "支付宝"),
    )
    ORDER_STATUS_ENUM = {
        "UNCOMMENT": 2,
        "FINISHED": 1
    }
    ORDER_STATUS_CHOICES = (
        (1, "已完成"),
        (2, "已取消"),
    )
    order_id = models.CharField(max_length=64, primary_key=True, verbose_name="订单号")
    user = models.ForeignKey(User,related_name='orders', on_delete=models.PROTECT, verbose_name="下单用户")
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="总金额")
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="单价", default=None)
    pay_method = models.SmallIntegerField(choices=PAY_METHOD_CHOICES, default=1, verbose_name="支付方式")
    status = models.SmallIntegerField(choices=ORDER_STATUS_CHOICES, default=1, verbose_name="订单状态")
    member_type_id = models.IntegerField(verbose_name='充值会员级别',default=None)
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        db_table = "tb_order_info"
        verbose_name = '订单基本信息'
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return self.order_id



class Payment(models.Model):
    """支付信息"""
    order_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='order_user', verbose_name='用户')
    trade_id = models.CharField(max_length=100, unique=True, null=True, blank=True, verbose_name="支付编号")
    order_payment = models.ForeignKey(OrderInfo,related_name='orders', on_delete=models.CASCADE, verbose_name="订单表信息", default=None)
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        db_table = 'tb_payment'
        verbose_name = '支付信息'
        verbose_name_plural = verbose_name

class Analysis(models.Model):
    """用户分析数据表"""
    analysis_time = models.DateTimeField(auto_now_add=True, verbose_name='当天时间')
    login_total = models.IntegerField(verbose_name="登录用户量",default=0)
    total_total = models.IntegerField(verbose_name="当天注册量",default=0)
    real_number = models.IntegerField(verbose_name="当天真实登录量", default=0)
    
    class Meta:
        db_table = 'tb_user_analysis'
        verbose_name = "每天用户数据量"

class LoginHistory(models.Model):
    """用户登录记录表"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_login_history', verbose_name='用户')
    login_time = models.DateTimeField(auto_now_add=True, verbose_name='登录时间')


    class Meta:
        db_table = 'tb_login_history'
        verbose_name = "用户登录记录"
        
class MethodDesc(models.Model):
    method_name = models.CharField(max_length=255, verbose_name='方法名称', default=None)
    method_desc = models.CharField(max_length=65535, verbose_name='方法描述', default=None)
    method_url = models.CharField(max_length=255, verbose_name='方法链接', default=None)
    class Meta:
        db_table = "tb_method_desc"
        verbose_name = '方法描述'
        verbose_name_plural = verbose_name
    def __str__(self):
        return f'method_name={self.method_name} method_desc={self.method_desc}'    
