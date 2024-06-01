from django.contrib import admin
from claims import models

# Register your models here.
admin.site.register(models.AddressCountry)
admin.site.register(models.AddressRegion)
admin.site.register(models.AddressCity)
admin.site.register(models.ProjectStatus)
