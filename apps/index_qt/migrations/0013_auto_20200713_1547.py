# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2020-07-13 07:47
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0012_auto_20200713_1133'),
    ]

    operations = [
        migrations.AlterField(
            model_name='browsing_process',
            name='order',
            field=models.CharField(max_length=65535, verbose_name='顺序编号'),
        ),
        migrations.AlterField(
            model_name='browsing_process',
            name='process_info',
            field=models.CharField(default=None, max_length=65535, verbose_name='具体浏览过程'),
        ),
        migrations.AlterField(
            model_name='file_old',
            name='background',
            field=models.CharField(default=None, max_length=65535, verbose_name='项目背景'),
        ),
        migrations.AlterField(
            model_name='file_old',
            name='file_name',
            field=models.CharField(max_length=65535, verbose_name='保存的文件名'),
        ),
        migrations.AlterField(
            model_name='file_old',
            name='outline',
            field=models.CharField(default=None, max_length=65535, verbose_name='项目概要'),
        ),
        migrations.AlterField(
            model_name='file_old',
            name='path',
            field=models.CharField(default=None, max_length=65535, verbose_name='保存文件的路径'),
        ),
        migrations.AlterField(
            model_name='file_old',
            name='project_name',
            field=models.CharField(default=None, max_length=65535, verbose_name='项目名称'),
        ),
        migrations.AlterField(
            model_name='user',
            name='dft_file',
            field=models.CharField(default=None, max_length=65535, verbose_name='默认文件名'),
        ),
        migrations.AlterField(
            model_name='user',
            name='image_tou',
            field=models.CharField(default=None, max_length=65535, verbose_name='头像'),
        ),
    ]
