from django.urls import path

from .views import DetailView

app_name = 'comments'

urlpatterns = [
    path('<slug:model>/<slug:slug>', DetailView.as_view(), name='detail')
]
