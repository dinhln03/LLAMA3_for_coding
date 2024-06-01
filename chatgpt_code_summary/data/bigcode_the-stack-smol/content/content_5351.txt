from django.contrib import admin
from menus.models import Dish, Menu


@admin.register(Menu)
class MenuAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description', 'created', 'updated')
    search_fields = ('id', 'name', 'description')
    list_filter = ('created', 'updated')
    raw_id_fields = ('dishes',)


@admin.register(Dish)
class DishAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'description',
        'price',
        'time_to_prepare',
        'is_vegetarian',
        'created',
        'updated',
    )
    search_fields = ('id', 'name', 'description')
    list_filter = ('is_vegetarian', 'created', 'updated')
