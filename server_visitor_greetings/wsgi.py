import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server_visitor_greetings.settings')

application = get_wsgi_application()
