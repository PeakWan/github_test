# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2021-01-28 02:45
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0020_auto_20210128_0157'),
    ]

    operations = [

        migrations.CreateModel(
            name='OrderInfo',
            fields=[
                ('order_id', models.CharField(max_length=64, primary_key=True, serialize=False, verbose_name='订单号')),
                ('total_amount', models.DecimalField(decimal_places=2, max_digits=10, verbose_name='总金额')),
                ('pay_method', models.SmallIntegerField(choices=[(1, '微信支付'), (2, '支付宝')], default=1, verbose_name='支付方式')),
                ('status', models.SmallIntegerField(choices=[(1, '已完成'), (2, '已取消')], default=1, verbose_name='订单状态')),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('update_time', models.DateTimeField(auto_now=True, verbose_name='更新时间')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='orders', to=settings.AUTH_USER_MODEL, verbose_name='下单用户')),
            ],
            options={
                'verbose_name': '订单基本信息',
                'verbose_name_plural': '订单基本信息',
                'db_table': 'tb_order_info',
            },
        ),

    ]
