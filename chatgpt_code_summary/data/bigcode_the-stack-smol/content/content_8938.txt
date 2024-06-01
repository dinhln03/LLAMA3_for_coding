from django.conf.urls import url
from visitorcounter import views

urlpatterns = [
    url(r'^$', views.index),
]
