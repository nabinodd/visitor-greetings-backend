from django.contrib import admin

from .models import Log, Visitor


@admin.register(Visitor)
class VisitorAdmin(admin.ModelAdmin):
   list_display = ('id', 'name', 'calc_emb')
   list_display_links = list_display
   search_fields = ('name',)


@admin.register(Log)
class LogAdmin(admin.ModelAdmin):
   list_display = ('id', 'visitor', 'reg_datetime', 'remarks')
   list_display_links = list_display
   search_fields = ('visitor__name',)