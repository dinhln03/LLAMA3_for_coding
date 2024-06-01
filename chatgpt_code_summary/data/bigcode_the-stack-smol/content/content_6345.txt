# coding: utf-8
from django.conf.urls import url
from api_v1 import views
urlpatterns = [
    url(r'^register/$', views.register),
    url(r'^login/$', views.login),
    url(r'^images/$', views.images),
    url(r'^reccomend/$', views.reccomend),
    url(r'^user_post/$', views.get_user_post),
]

# api viewer (debug用)
from rest_framework import routers
from .views import UserViewSet, TokenViewSet, ImageViewSet, PostViewSet, FavoriteViewSet

router = routers.DefaultRouter()
router.register(r'user', UserViewSet)
router.register(r'token', TokenViewSet)
router.register(r'image', ImageViewSet)
router.register(r'post', PostViewSet)
# router.register(r'favorite', FavoriteViewSet)
