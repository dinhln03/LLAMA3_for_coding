from django.urls import path
from . import views

urlpatterns = [
    path('list', views.list_view),
    path('add', views.add_view),
]