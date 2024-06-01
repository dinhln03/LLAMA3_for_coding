#Django
from django.contrib import admin

#Model
from cride.rides.models import Ride

@admin.register(Ride)
class RideAdmin(admin.ModelAdmin):
    pass