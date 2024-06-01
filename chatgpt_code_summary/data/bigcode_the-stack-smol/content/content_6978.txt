"""trefechanwen URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from . import views as views
from rest_framework import routers
from bookings import views as bookings_views

router = routers.DefaultRouter()
router.register(r'availabilitydates', bookings_views.AvailabilityDateViewSet, base_name='AvailabilityDates')


urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^cottage$', views.cottage, name='cottage'),
    url(r'^barn$', views.barn, name='barn'),
    url(r'^availability$', views.availability, name='availability'),
    url(r'^localinfo$', views.localinfo, name='localinfo'),
    url(r'^location$', views.location, name='location'),
    url(r'^walking$', views.walking, name='walking'),
    url(r'^beaches$', views.beaches, name='beaches'),
    url(r'^wildlife', views.wildlife, name='wildlife'),
    url(r'^contact$', views.contact, name='contact'),
    url(r'^covid$', views.covid, name='covid'),
    url(r'^admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
