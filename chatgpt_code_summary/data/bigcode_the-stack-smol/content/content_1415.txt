from django.urls import path
from django.views.i18n import JavaScriptCatalog

from .views import HelloWorldView


app_name = 'pokus1'  # make possible use {% url 'pokus1:..' %}
# however this is maybe deprecated; you can achieve same in include(), see project level urls.py

urlpatterns = [
    path('jsi18n/pokus1/', JavaScriptCatalog.as_view(), name='javascript-catalog'),  # /pokus1/: unique app url, probably important
    path('', HelloWorldView.as_view(), name='hello')
]
