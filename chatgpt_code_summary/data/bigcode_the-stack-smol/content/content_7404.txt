from django.contrib import admin

from models import Connection, Network


class ConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", 'device_1', 'device_2', 'network', 'concetion_type', )
    list_filter = ['network', 'concetion_type']
    search_fields = ['device_1__property_number', 'device_2__property_number']

admin.site.register(Connection, ConnectionAdmin)
admin.site.register(Network)
