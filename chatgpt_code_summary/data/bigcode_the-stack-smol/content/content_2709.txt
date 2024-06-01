from django.contrib import admin
from .models import Car, CarShop, RepairStation, RepairWork, Reapir, Person, Component
# Register your models here.

admin.site.register(Car)
admin.site.register(CarShop)
admin.site.register(Reapir)
admin.site.register(RepairWork)
admin.site.register(RepairStation)
admin.site.register(Person)
admin.site.register(Component)
