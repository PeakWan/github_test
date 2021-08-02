# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2020-07-08 02:42
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0003_auto_20200706_1049'),
    ]

    operations = [
        migrations.CreateModel(
            name='File_old',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.CharField(max_length=256, verbose_name='保存的文件名')),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
            ],
            options={
                'verbose_name': '上传记录',
                'db_table': 'tb_file',
            },
        ),
        migrations.AddField(
            model_name='user',
            name='dft_file',
            field=models.CharField(default=None, max_length=256, verbose_name='默认文件名'),
        ),
        migrations.AddField(
            model_name='file_old',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='user', to=settings.AUTH_USER_MODEL, verbose_name='用户'),
        ),
    ]