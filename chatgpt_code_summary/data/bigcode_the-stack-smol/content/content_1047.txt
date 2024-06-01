from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.login_root, name='login_root'),
    url(r'^success/$', views.login_success, name='login_success'),
    url(r'^fail/$', views.login_fail, name='login_fail'),
]
