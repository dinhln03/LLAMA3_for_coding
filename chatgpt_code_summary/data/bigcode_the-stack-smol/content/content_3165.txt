# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):
    dependencies = [
        ('focus', '0006_auto_20160209_1200'),
    ]

    operations = [
        migrations.AlterField(
            model_name='remedial',
            name='focusRoom',
            field=models.ForeignKey(help_text=b'The focusroom that this remedial is assigned to', to='focus.FocusRoom'),
        ),
    ]
