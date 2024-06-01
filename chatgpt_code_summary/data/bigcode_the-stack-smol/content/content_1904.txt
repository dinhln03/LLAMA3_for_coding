from django.contrib import admin

from .. import models


@admin.register(models.Code)
class CodeAdmin(admin.ModelAdmin):
    """
    코드 정보
    """
    list_display = ['id', 'author', 'problem', 'language', 'name', 'available_game', 'date' ]

    class Meta:
        model = models.Code

