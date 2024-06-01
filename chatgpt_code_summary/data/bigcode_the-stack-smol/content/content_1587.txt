"""
Configuration file for py.test
"""

import django


def pytest_configure():
    from django.conf import settings
    settings.configure(
        DEBUG=True,
        USE_TZ=True,
        USE_I18N=True,
        ROOT_URLCONF="tests.urls",
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": "test.sqlite3",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.auth",
            "django.contrib.admin",
            "django.contrib.contenttypes",
            "django.contrib.sessions",
            "django.contrib.sites",
            "simple_auth",
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "django.middleware.csrf.CsrfViewMiddleware",
            "django.contrib.sessions.middleware.SessionMiddleware",
            "django.contrib.auth.middleware.AuthenticationMiddleware",
            "simple_auth.middleware.SimpleAuthMiddleware",
        ],
        TEMPLATES = [
            {
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'APP_DIRS': True,
            },
        ],
        SITE_ID=1,
    )
    django.setup()
