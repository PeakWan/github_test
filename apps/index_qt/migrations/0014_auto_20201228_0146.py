# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2020-12-28 01:46
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0013_auto_20200713_1547'),
    ]

    operations = [
        migrations.CreateModel(
            name='Modelbase',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(default=None, max_length=65535, verbose_name='模型库名称')),
                ('model_background', models.CharField(default=None, max_length=65535, verbose_name='模型库说明')),
                ('model_outline', models.CharField(default=None, max_length=65535, verbose_name='模型库方法')),
                ('model_info', models.CharField(default=None, max_length=65535, verbose_name='分析结果图片')),
                ('model_type', models.CharField(default=None, max_length=65535, verbose_name='模型类型')),
                ('model_path', models.CharField(default=None, max_length=65535, verbose_name='保存文件的路径')),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('last_time', models.DateTimeField(auto_now=True, verbose_name='修改时间')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='model_user', to=settings.AUTH_USER_MODEL, verbose_name='用户')),
            ],
            options={
                'verbose_name': '模型列表',
                'db_table': 'tb_model_base',
            },
        ),
    ]
