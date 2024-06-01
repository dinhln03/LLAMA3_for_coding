from django.conf.urls import url

from .views import (
    TableListAPIView,
    SalePointTableListAPIView
    )


urlpatterns = [
    url(r'^$', TableListAPIView.as_view(),
        name='api-table-list'),
    url(r'^sale-point/(?P<pk>[0-9]+)$',
        SalePointTableListAPIView.as_view(),
        name='api-sale_point-table'),
]

