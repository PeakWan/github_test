# -*- coding: utf-8 -*-
# Generated by Django 1.11.11 on 2021-01-11 08:29
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('index_qt', '0016_auto_20210111_0316'),
    ]

    operations = [
        migrations.CreateModel(
            name='Commits_books',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.CharField(max_length=100, verbose_name='评论内容')),
                ('create_time', models.DateField(auto_now_add=True, verbose_name='创建时间')),
                ('parent', models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, related_name='comm_subs', to='index_qt.Commits_books', verbose_name='父评论')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='commit_library', to=settings.AUTH_USER_MODEL, verbose_name='用户')),
            ],
            options={
                'verbose_name': '评论',
                'db_table': 'tb_commit',
            },
        ),
    
    ]
