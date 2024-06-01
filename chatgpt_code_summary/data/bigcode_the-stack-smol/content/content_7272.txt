# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('report_database', '0003_auto_20160501_1646'),
    ]

    operations = [
        migrations.AddField(
            model_name='report',
            name='shared_users',
            field=models.ManyToManyField(related_name='shared_users', to=settings.AUTH_USER_MODEL),
        ),
    ]
