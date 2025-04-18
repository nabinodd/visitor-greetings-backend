from django.contrib import admin
from django.urls import path

admin.site.site_header = 'superadmin'
admin.site.site_title = 'Greetings Superadmin'

urlpatterns = [
    path('admin/', admin.site.urls),
]
