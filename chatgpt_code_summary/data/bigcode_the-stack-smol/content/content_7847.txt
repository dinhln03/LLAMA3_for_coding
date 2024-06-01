# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AllFieldsModel',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('char_field', models.CharField(help_text=b'write something', max_length=500)),
                ('int_field', models.IntegerField(help_text=b'Put a number, magic number')),
                ('text_field', models.TextField(help_text=b'Put a large test here')),
                ('big_integer_field', models.BigIntegerField(help_text=b'An big integer field')),
                ('binary_field', models.BinaryField(help_text=b'A binary field')),
                ('date_field', models.DateField(help_text=b'A date field')),
                ('datetime_field', models.DateTimeField(help_text=b'A datetime field')),
                ('boolean_field', models.BooleanField(help_text=b'A boolean field')),
                ('comma_separated_integer_field', models.CommaSeparatedIntegerField(help_text=b'A comma sepparated integer field', max_length=200)),
                ('decimal_field', models.DecimalField(help_text=b'A decimal field', max_digits=100, decimal_places=10)),
                ('duration_field', models.DurationField(help_text=b'A duration field')),
                ('email_field', models.EmailField(help_text=b'A email field', max_length=254)),
                ('file_field', models.FileField(help_text=b'A file field', upload_to=b'')),
                ('file_path_field', models.FilePathField(help_text=b'A file path field')),
                ('float_field', models.FloatField(help_text=b'A float field')),
                ('generic_ip_addr_field', models.GenericIPAddressField(help_text=b'A generic ip addr field')),
                ('image_field', models.ImageField(help_text=b'A image field', upload_to=b'')),
                ('null_boolean_field', models.NullBooleanField(help_text=b'A null boolean field')),
                ('positive_integer_field', models.PositiveIntegerField(help_text=b'A positive integer')),
                ('positive_small_integer_field', models.PositiveSmallIntegerField(help_text=b'A positive small integer field')),
                ('slug_field', models.SlugField(help_text=b'A slug field')),
                ('small_integer_field', models.SmallIntegerField(help_text=b'A small integer field')),
                ('time_field', models.TimeField(help_text=b'A time field')),
                ('url_field', models.URLField(help_text=b'A url field')),
                ('uuid_field', models.UUIDField(help_text=b'A uuid field')),
            ],
        ),
        migrations.CreateModel(
            name='ForeingModel',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(help_text=b'write something', max_length=500)),
                ('age', models.PositiveSmallIntegerField()),
                ('birthday', models.DateField()),
                ('foreign_key_field', models.ForeignKey(help_text=b'A foreign_key field', to='second_app.AllFieldsModel')),
            ],
        ),
        migrations.CreateModel(
            name='ManyToManyModel',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(help_text=b'write something', max_length=500)),
            ],
        ),
        migrations.AddField(
            model_name='allfieldsmodel',
            name='many_to_many_field',
            field=models.ManyToManyField(help_text=b'A many to many field', to='second_app.ManyToManyModel'),
        ),
    ]
