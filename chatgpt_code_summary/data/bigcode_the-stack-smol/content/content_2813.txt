from django.urls import include, path
from django.contrib import admin
from django.views.generic import RedirectView

urlpatterns = [
    path('polls/', include('polls.urls')),
    path('admin/', admin.site.urls),
]

urlpatterns += [
    path('', RedirectView.as_view(url='/polls/')),
]