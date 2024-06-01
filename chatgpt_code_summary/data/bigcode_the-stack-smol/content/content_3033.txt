#!/bin/env
"""
Help new users configure the database for use with social networks.
"""

import os
from datetime import datetime

# Fix Python 2.x.
try:
    input = raw_input
except NameError:
    pass

import django
from django.conf import settings
from django.core.management.utils import get_random_secret_key

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

settings.configure(
    DEBUG=True,
    TEMPLATES=[dict(
        # DEBUG = True,
        BACKEND='django.template.backends.django.DjangoTemplates',
        APP_DIRS=True,
        DIRS=[
            os.path.join(BASE_DIR, 'allauthdemo'),
        ],
    )],
)

try:
    django.setup()  # for Django >= 1.7
except AttributeError:
    pass  # must be < Django 1.7

from django.template.loader import get_template
from django.template import engines

commands_template = engines['django'].from_string("""
Run these commands:

    python manage.py makemigrations allauthdemo_auth
    python manage.py migrate
    python manage.py createsuperuser
    {% if facebook %}# Facebook
    python manage.py set_auth_provider facebook {{facebook.client_id}} {{facebook.secret}}{% endif %}
    {% if google %}# Google
    python manage.py set_auth_provider google {{google.client_id}} {{google.secret}}{% endif %}
    {% if github %}# GitHub
    python manage.py set_auth_provider github {{github.client_id}} {{github.secret}}{% endif %}
    {% if vk %}# VK
    python manage.py set_auth_provider vk {{vk.client_id}} {{vk.secret}}{% endif %}

If you have other providers you can add them in that way.
""")

settings_template = get_template("settings.template.py")



def heading(text):
    text = text.strip()
    line = '-' * len(text)
    print("\n%s\n%s\n%s\n" % (line, text, line))

def print_list(ls):
    max_len = max([len(i) for i in ls])
    num = len(str(len(ls))) #TODO: full list providers
    line = '-' * (2+num+3+max_len+2)

    for i in range(len(ls)):
        print(line)
        print("| %d | %s "% (i+1, ls[i]))
        

def ask_text(need, default=None):
    need = need.strip()
    if default:
        msg = "\n%s? Default: [%s] > " % (need, default)
    else:
        msg = "\n%s? > " % need

    while True:
        response = input(msg)
        if response:
            return response
        elif default is not None:
            return default
        else:
            pass  # raw_input('Please enter a value.')

providers = ['facebook', 'google', 'github', 'vk']

if __name__ == "__main__":
    
    context = {
        'now': str(datetime.now()),
        'secret_key': get_random_secret_key(),
    }
    
    print_list(providers)
    print("Please list comma-separated providers. Example: 1,2,3,4")
    corrct_providers = [int(i)-1 for i in input("Please enter: ").split(',')]

    for i in corrct_providers:
        p = providers[i]
        heading(p)
        
        secret = ask_text("{} Secret"%(p))
        client_id = ask_text("{} Client ID"%(p))
        context[p] = dict(secret=secret, client_id=client_id)
    
    heading("Rendering settings...")
    with open('allauthdemo/settings.py', 'w') as out:
        out.write(settings_template.render(context, request=None))
    print("OK")

    heading("Next steps")
    print(commands_template.render(context, request=None))
    heading("Done")
