from django.contrib import admin
from django.utils.html import format_html

from .models import Guest, Log, Visitor


@admin.register(Visitor)
class VisitorAdmin(admin.ModelAdmin):
   list_display = (
      'id', 'addressing', 'name', 'calc_emb', 
      'image_cropped_preview'
   )
   list_display_links = list_display
   search_fields = ('name',)

   def image_cropped_preview(self, obj):
      if obj.image_cropped:
         return format_html(
            '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 10%;" />', 
            obj.image_cropped.url
         )


@admin.register(Log)
class LogAdmin(admin.ModelAdmin):
   list_display = ('id', 'visitor', 'reg_datetime', 'image_preview', 'remarks')
   list_display_links = list_display
   search_fields = ('visitor__name',)

   def image_preview(self, obj):
      if obj.image:
         return format_html(
            '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 10%;" />', 
            obj.image.url
         )


@admin.register(Guest)
class GuestAdmin(admin.ModelAdmin):
   list_display = ('id', 'created_at', 'greeting_text', 'image_preview')
   list_display_links = list_display


   def image_preview(self, obj):
      if obj.image:
         return format_html(
            '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 10%;" />', 
            obj.image.url
         )