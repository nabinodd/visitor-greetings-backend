from django.core.management.base import BaseCommand

from recognition.tracker import run_recognition_pipeline


class Command(BaseCommand):
   help = "Run real-time visitor recognition with greeting"

   def handle(self, *args, **kwargs):
      run_recognition_pipeline()
