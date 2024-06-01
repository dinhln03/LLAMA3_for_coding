#!/usr/bin/env python

from django.contrib.auth import get_user_model
import django
import configparser
import sys
import os

django_project_name = os.environ.get('DJANGO_PROJECT_NAME')
sys.path.append('/app')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", django_project_name+'.settings')

django.setup()

def django_superuser_creation(**kwargs):
    User = get_user_model()
    if User.objects.filter(username=kwargs['username']).count()==0:
        User.objects.create_superuser(**kwargs)
        print('[django] Superuser created.')
    else:
        print('[django] Superuser creation skipped.')

config = configparser.ConfigParser()
config.read('/run/secrets/djangodb.cnf')

try:
    user_detail = dict(config['superuser'])
except KeyError as e:
    print('[django] Config file for django does not has superuser detail.')
    sys.exit(1)
else:
    django_superuser_creation(**user_detail)
