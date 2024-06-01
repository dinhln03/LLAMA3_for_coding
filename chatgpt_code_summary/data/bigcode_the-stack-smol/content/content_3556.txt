from django.contrib import admin
from profiles_api import models

# Register your models here.
admin.site.register(models.UserProfile) # registers the model on the admin site
admin.site.register(models.ProfileFeedItem)
