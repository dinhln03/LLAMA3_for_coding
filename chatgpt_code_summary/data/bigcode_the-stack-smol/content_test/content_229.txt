from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'tables/$', views.report_tables, name='tables'),
    url(r'^api/stop_enforcement/', views.stop_enforcement_json_view, name='stop_enforcement'),
    url(r'^api/residency/', views.resident_json_view, name='residency'),
    url(r'^api/nature_of_stops/', views.nature_of_stops_json_view, name='nature_of_stop'),
    url(r'^api/disposition/', views.disposition_json_view, name='disposition'),
    url(r'^api/statutory_authority/', views.statutory_authority_json_view, name='stop_authority'),
    url(r'^api/stops_by_month/', views.monthly_stops_json_view, name='stops_by_month'),
    url(r'^api/stops_by_hour/', views.stops_by_hour_json_view, name='stops_by_hour'),
    url(r'^api/stops_by_age/', views.stops_by_age_json_view, name='stops_by_age'),
    url(r'^api/search_information/', views.search_information_json_view, name='search_information'),
    url(r'^api/search_authority/', views.search_authority_json_view, name='search_authority'),
    url(r'^api/traffic_stops/', views.traffic_stops_json_view, name='stops'),
    url(r'^api/departments/', views.department_json_view, name='departments')
    ]
