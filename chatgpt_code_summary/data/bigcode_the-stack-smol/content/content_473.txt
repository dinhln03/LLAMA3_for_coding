from django.conf.urls import url


from . import views

app_name = 'reports'
urlpatterns = [
  
    # url(r'^graph/', views.graph,  name='graph'),
    url(r'^graph/', views.statistics,  name='graph'),
    url(r'^csv_export/', views.csv_export,  name='csv_export'),

]
