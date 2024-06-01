from django.urls import path

from el_galleria import views

urlpatterns = [
    path('', views.index, name="home"),
    path('category/<str:selected_category>/', views.category, name="category"),
    path('search/<str:search_str>/', views.search, name="search")

]