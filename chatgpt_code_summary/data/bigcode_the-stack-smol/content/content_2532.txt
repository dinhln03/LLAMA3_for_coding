from django.urls import include, path, re_path
from .models import *
from .views import *

urlpatterns = [
	path('imagenes/', lista_galerias_img, name='lista-galerias-img'),
	path('imagenes/<tema>', filtro_temas_img, name='filtro_temas_img'),
	path('imagenes/<id>/', detalle_galerias_img, name='detalle-galerias-img'),
	path('videos/', lista_galerias_videos, name='lista-galerias-videos'),
	path('videos/<tema>', filtro_temas_vid, name='filtro_temas_vid'),
]