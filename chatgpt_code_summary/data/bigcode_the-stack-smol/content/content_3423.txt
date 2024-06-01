from django.contrib import admin
from graphite.events.models import Event


class EventsAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {
            'fields': ('when', 'what', 'data', 'tags',)
        }),
    )
    list_display = ('when', 'what', 'data',)
    list_filter = ('what',)
    search_fields = ('tags', )

admin.site.register(Event, EventsAdmin)
