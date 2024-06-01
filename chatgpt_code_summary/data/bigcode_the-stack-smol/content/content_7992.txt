# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

SECRET_KEY = 'psst'
INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'hfut_auth'
)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
        'USER': '',
        'PASSWORD': '',
        'HOST': '',
        'PORT': '',
    }
}

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'hfut_auth.backends.HFUTBackend'
)

# default
HFUT_AUTH_CAMPUS = 'ALL'
