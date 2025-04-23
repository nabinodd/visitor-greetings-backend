import time

import cv2
from django.core.management.base import BaseCommand

from greetings.capture import (capture_guest_image, flush_camera,
                               initialize_camera, load_model, warmup_camera)
from greetings.describe_and_greet import describe_and_greet


def now():
   return int(time.time())

class Command(BaseCommand):
   help = 'Continuously captures guest images, generates descriptions, and greets via TTS'

   def handle(self, *args, **options):
      self.stdout.write("Starting guest greeting loop...")

      # Initialize camera and model ONCE
      cap = initialize_camera(camera_index=1)
      warmup_camera(cap)
      model = load_model()

      try:
         while True:
               self.stdout.write(self.style.HTTP_REDIRECT(f"[INFO @ {now()}] Looking for the next guest..."))
               flush_camera(cap, frames=10)

               guest, visitor = capture_guest_image(cap, model)

               if guest == "EXIT":
                  self.stdout.write(self.style.WARNING(f"[WARNING @ {now()}] Exiting guest greeting loop."))
                  break

               if not guest:
                  self.stderr.write(f"[ERROR @ {now()}] No guest captured. Retrying...")
                  continue

               self.stdout.write(self.style.SUCCESS(f"[SUCCESS @ {now()}] Guest captured. Guest ID: {guest.id}"))

               description = describe_and_greet(guest.image.path, visitor)

               if description:
                  guest.greeting_text = description
                  guest.save()
                  self.stdout.write(self.style.SUCCESS(f"[SUCCESS @ {now()}] Greeting saved for Guest {guest.id}"))
                  time.sleep(0.5)
               else:
                  self.stderr.write(f"[ERROR @ {now()}] Failed to generate greeting for Guest {guest.id}")
                  time.sleep(0.1)

               flush_camera(cap, frames=10)
               capture_guest_image(cap, model, overlay_only=True)

      finally:
         cap.release()
         cv2.destroyAllWindows()
