from django.conf.urls import url
from django.conf.urls import patterns

from events import views

urlpatterns = patterns('',
    # Events
    url(r'^microcosms/(?P<microcosm_id>\d+)/create/event/$', views.create, name='create-event'),
    url(r'^events/(?P<event_id>\d+)/$', views.single, name='single-event'),
    url(r'^events/(?P<event_id>\d+)/csv/$', views.csv, name='csv-event'),
    url(r'^events/(?P<event_id>\d+)/edit/$', views.edit, name='edit-event'),
    url(r'^events/(?P<event_id>\d+)/delete/$', views.delete, name='delete-event'),
    url(r'^events/(?P<event_id>\d+)/newest/$', views.newest, name='newest-event'),
    # RSVP to an event
    url(r'^events/(?P<event_id>\d+)/rsvp/$', views.rsvp, name='rsvp-event'),

    # Proxy geocoding requests to the backend
    url(r'^geocode/$', views.geocode, name='geocode'),
)