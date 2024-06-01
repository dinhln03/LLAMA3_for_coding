from .base import *

ALLOWED_HOSTS = ["localhost", "0.0.0.0", "127.0.0.1"]

DATABASES = {
    'default': {
        'ENGINE': os.getenv('SQL_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.getenv('SQL_DATABASE', os.path.join(BASE_DIR, 'db.sqlite3')),
        'USER': os.getenv('SQL_USER', 'user'),
        'PASSWORD': os.getenv('SQL_PASSWORD', 'password'),
        'HOST': os.getenv('SQL_HOST', 'localhost'),
        'PORT': os.getenv('SQL_PORT', '5432'),
    }
}

ASSETS_ASSET_FILE_PART_SIZE = 60
