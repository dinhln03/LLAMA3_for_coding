from django.contrib import admin
from leaflet.admin import LeafletGeoAdmin

from .models import ProblemLabel, ProblemStatus

# Register your models here.


admin.site.register(ProblemLabel, LeafletGeoAdmin)
admin.site.register(ProblemStatus)
