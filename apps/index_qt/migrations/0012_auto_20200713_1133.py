# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2020-07-13 03:33
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0011_browsing_process_is_delete'),
    ]

    operations = [
        migrations.AlterField(
            model_name='browsing_process',
            name='order',
            field=models.CharField(max_length=255, verbose_name='顺序编号'),
        ),
    ]
