from django.conf.urls import patterns, include, url
from django.views.generic import TemplateView
import settings.base
# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    url(r'^$', 'profiles.views.index', name='index'),
	url(r'^accounts/', include('allauth.urls')),
    # Examples:
    # url(r'^$', 'explorind_project.views.home', name='home'),
    # url(r'^explorind_project/', include('explorind_project.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),
	url(r'^locations/', include('locations.urls')),	
	url(r'^static/(?P<path>.*)$', 'django.views.static.serve', {
        'document_root': settings.base.STATIC_ROOT,	
    }),
	
	url(r'^media/(?P<path>.*)$', 'django.views.static.serve', {
		'document_root': settings.base.MEDIA_ROOT}),
	url(r'^login$', 'profiles.views.login_view'), # login
    url(r'^logout$', 'profiles.views.logout_view'), # logout
    url(r'^signup$', 'profiles.views.signup'), # signup
	url(r'^submit$', 'profiles.views.submit'),
	url(r'^reviews$', 'profiles.views.public'),
	url(r'^users/$', 'profiles.views.users'),
    url(r'^users/(?P<username>.{0,30})/$', 'profiles.views.users'),
    url(r'^follow$', 'profiles.views.follow'),
)
