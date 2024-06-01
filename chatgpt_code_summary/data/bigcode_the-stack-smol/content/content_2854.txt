from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('security/', auth_views.PasswordChangeView.as_view(), name='security_settings'),
    path('security/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    path('account/', views.profile_settings, name='profile_settings'),
    path('edit/', views.edit_profile, name='edit_profile'), #remove this later

]