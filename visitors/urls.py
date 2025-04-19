from django.urls import path

from .views import IdentifyVisitorAPIView

urlpatterns = [
   path('identify-visitor/', IdentifyVisitorAPIView.as_view(), name='identify-visitor'),
]