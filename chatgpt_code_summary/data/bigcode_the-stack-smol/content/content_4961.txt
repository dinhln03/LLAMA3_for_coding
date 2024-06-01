# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_auto_20141114_1935'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='registryentry',
            name='fixity_algorithm',
        ),
        migrations.RemoveField(
            model_name='registryentry',
            name='fixity_value',
        ),
        migrations.RemoveField(
            model_name='registryentry',
            name='last_fixity_date',
        ),
        migrations.RemoveField(
            model_name='registryentry',
            name='published',
        ),
    ]
