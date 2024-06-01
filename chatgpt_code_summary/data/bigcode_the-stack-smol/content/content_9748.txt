# author : Sam Rapier
from deploy_django_to_azure.settings.base import *

DEBUG = False

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = 'https://exeblobstorage.blob.core.windows.net/static-files/'


# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'sql_server.pyodbc',
        'NAME': 'DB-exeterOrientation',
        'USER': 'user-admin',
        'PASSWORD': 'v%mRn3os#9P2JnjnV*dJ',
        'HOST': 'db-exeter-orientation.database.windows.net',
        'PORT': '1433',
		'OPTIONS': {
             'driver': 'ODBC Driver 17 for SQL Server',
             'MARS_Connection': 'True',
         }
    }
}