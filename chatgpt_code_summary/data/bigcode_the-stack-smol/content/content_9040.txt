from . import views
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', views.index, name='home'),
    path('login/', views.login, name='login_user'),
    path('logout/', views.logout, name='logout_user'),
    path('signup/', views.signup, name='signup_user'),
    path('description/', views.deskripsi, name='desc'),
    path('about/', views.tentang, name='tentang'),
    path('user/<slug:id>/', views.detail_user, name='detail_user'),
    path('api/user/', include('login.urls')),
]
