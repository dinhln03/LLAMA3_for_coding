# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import Location,Category,Image
# Register your models here.

admin.site.register(Location)
admin.site.register(Category)
admin.site.register(Image)


class Image(admin.ModelAdmin):
    search_fields = ('image_category')