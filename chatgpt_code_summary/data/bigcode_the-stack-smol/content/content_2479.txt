from django.contrib import admin
from .models import *
from django.contrib.auth.models import User

class ImageAdmin(admin.ModelAdmin):
  fields = ( 'image','name','caption','profile','post_date', 'user', )
  readonly_fields = ('profile', 'post_date', 'user',)

#registering the models
# admin.site.register(Image, ImageAdmin)
admin.site.register(Profile)
admin.site.register(Image)
admin.site.register(Like)
admin.site.register(Comment)
