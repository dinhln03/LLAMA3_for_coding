from django.urls import path

from . import views

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('settings', views.project_settings, name='settings'),
    path('envs', views.os_envs, name='envs'),
]
