from django.contrib import admin

# Register your models here.
from account.models import UserProfile
from blog.models import BlogArticles


class BlogArticlesAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "publish")
    list_filter = ("publish", "author")
    search_fields = ("title", "body")
    raw_id_fields = ("author",)
    date_hierarchy = "publish"
    ordering = ("-publish", "author")


admin.site.register(BlogArticles, BlogArticlesAdmin)


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "birth", "phone")
    list_filter = ("phone",)


admin.site.register(UserProfile, UserProfileAdmin)
