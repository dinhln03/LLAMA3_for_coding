""" core app configuration """
import os

environment = os.getenv('LAMBTASTIC_ENV', 'development')

if environment == 'testing':
    from .testing import *
elif environment == 'production':
    from .production import *
else:
    from .development import *
