from django.contrib import admin
from django.urls import include, path

admin.site.site_header = 'superadmin'
admin.site.site_title = 'Greetings Superadmin'

urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('api/v1/visitors/', include('visitors.urls')),
]
