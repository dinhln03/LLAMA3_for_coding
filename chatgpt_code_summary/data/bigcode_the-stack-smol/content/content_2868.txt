from django.urls import path, include
from .views import *
from django.contrib import admin


app_name = 'Blog'

urlpatterns = [
    path('', Blog_index.as_view(), name='blog_index'),
    path('', include('Usermanagement.urls', namespace='Usermanagement'), name='usermanagement'),
    path('create_article/', article_create, name='article_create'),
    path('<str:column_name>/', Column_detail.as_view(), name='column_detail'),
    path('<str:column_name>/<slug:article_name>/', Article_detail.as_view(), name='article_detail'),
    path('<str:column_name>/<slug:article_name>/delete/', article_delete, name='article_delete'),
    ]
