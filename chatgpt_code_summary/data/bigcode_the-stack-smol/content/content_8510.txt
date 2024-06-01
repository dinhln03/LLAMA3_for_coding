# -*- coding: utf-8 -*-
from django.conf.urls import url
from django.views.generic import TemplateView

from . import views


app_name = 'services_communicator'
urlpatterns = [
    url(
        regex="^ServiceList/~create/$",
        view=views.ServiceListCreateView.as_view(),
        name='ServiceList_create',
    ),
    url(
        regex="^ServiceList/(?P<pk>\d+)/~delete/$",
        view=views.ServiceListDeleteView.as_view(),
        name='ServiceList_delete',
    ),
    url(
        regex="^ServiceList/(?P<pk>\d+)/$",
        view=views.ServiceListDetailView.as_view(),
        name='ServiceList_detail',
    ),
    url(
        regex="^ServiceList/(?P<pk>\d+)/~update/$",
        view=views.ServiceListUpdateView.as_view(),
        name='ServiceList_update',
    ),
    url(
        regex="^ServiceList/$",
        view=views.ServiceListListView.as_view(),
        name='ServiceList_list',
    ),
	]
