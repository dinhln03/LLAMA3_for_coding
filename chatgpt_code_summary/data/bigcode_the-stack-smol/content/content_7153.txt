"""
URLConf for OCR presets.
"""
from django.conf.urls.defaults import *
from django.contrib.auth.decorators import login_required
from ocradmin.presets import views


urlpatterns = patterns('',
	(r'^builder/?$', login_required(views.builder)),
	(r'^builder/(?P<pid>[^/]+)/?$', login_required(views.builder_doc_edit)),
    (r'^clear_cache/?$', login_required(views.clear_cache)),
    (r'^clear_node_cache/?$', views.clear_node_cache),
    (r'^create/?$', login_required(views.presetcreate)),
    (r'^createjson/?$', login_required(views.createjson)),
    (r'^data/(?P<slug>[-\w]+)/?$', views.data),
    (r'^delete/(?P<slug>[-\w]+)/?$', login_required(views.presetdelete)),
    (r'^download/(?P<slug>[-\w]+)/?$', views.download),
    (r'^edit/(?P<slug>[-\w]+)/?$', login_required(views.presetedit)),
    (r'^fetch/?$', views.fetch),
    (r'^layout_graph/?$', views.layout_graph),
    (r'^list/?$', views.presetlist),
    (r'^query/?$', views.query_nodes),
    (r'^run/?$', views.run_preset),
    (r'^show/(?P<slug>[-\w]+)/?$', views.presetdetail),
    (r'^update_data/(?P<slug>[-\w]+)/?$', views.update_data),
    (r'^upload/?$', login_required(views.upload_file)),
    (r'^(?P<slug>[-\w]+)/?$', views.presetdetail),
)
