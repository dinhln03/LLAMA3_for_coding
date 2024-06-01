from django.urls import path, include
from . import views

app_name = 'test_app'

urlpatterns = [
    path('', views.index, name='index'),
]