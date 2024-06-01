from django.contrib import admin

from .models import Arts, Comments, Tags, ArtworksTags, Stili, Umetnina, Umetnik

# Register your models here.

admin.site.register(Umetnik)
admin.site.register(Umetnina)
admin.site.register(Stili)
admin.site.register(Arts)
admin.site.register(Comments)
admin.site.register(Tags)
admin.site.register(ArtworksTags)
# admin.site.register(ArtworkLikes)
