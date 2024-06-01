try:
    from django.conf.urls import *
except ImportError:  # django < 1.4
    from django.conf.urls.defaults import *

urlpatterns = patterns("rush_forms.views",
                       #url(r"^(?P<pk>\d+)/$", 'form_view', name='form-detail'),
                       url(r"^(?P<pk>\d+)/$", 'rush_form_view', name='detail'),
                       url(r"^(?P<pk>\d+)/(?P<user_id>\d+)/$", 'rush_form_user_entry_view', name='user_entry'),
                       )
