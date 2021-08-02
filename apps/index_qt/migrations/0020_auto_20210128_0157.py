# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2021-01-28 01:57
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0019_auto_20210112_0646'),
    ]

    operations = [

        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('trade_id', models.CharField(blank=True, max_length=100, null=True, unique=True, verbose_name='支付编号')),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('update_time', models.DateTimeField(auto_now=True, verbose_name='更新时间')),
                ('order_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='order_user', to=settings.AUTH_USER_MODEL, verbose_name='用户')),
            ],
            options={
                'verbose_name': '支付信息',
                'verbose_name_plural': '支付信息',
                'db_table': 'tb_payment',
            },
        ),

    ]