#-*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from salmonella.admin import SalmonellaMixin

from edw.models.postal_zone import PostZoneModel


# ===========================================================================================
# PostalZoneAdmin
# ===========================================================================================
class PostalZoneAdmin(SalmonellaMixin, admin.ModelAdmin):
    model = PostZoneModel

    list_display = ['name', 'active']

    fields = ['term', 'postal_codes', 'active']

    search_fields = ('term__name', 'postal_codes')

    salmonella_fields = ('term',)


admin.site.register(PostZoneModel, PostalZoneAdmin)