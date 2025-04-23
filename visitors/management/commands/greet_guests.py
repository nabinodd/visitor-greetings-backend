import time

import cv2
from django.core.management.base import BaseCommand

from greetings.capture import (capture_guest_image, initialize_camera,
                               load_model, warmup_camera)
from greetings.describe_and_greet import describe_and_greet


class Command(BaseCommand):
   help = 'Continuously captures guest images, generates descriptions, and greets via TTS'

   def handle(self, *args, **options):
      self.stdout.write("Starting guest greeting loop...")

      # Initialize camera and model ONCE
      cap = initialize_camera(camera_index=0)
      warmup_camera(cap)
      model = load_model()

      try:
         while True:
               self.stdout.write("Looking for the next guest...")

               guest = capture_guest_image(cap, model)

               if guest == "EXIT":  # 🔥 Handle exit signal
                  self.stdout.write(self.style.WARNING("Exiting guest greeting loop."))
                  break  # Exit the infinite loop gracefully

               if not guest:
                  self.stderr.write("❌ No guest captured. Retrying...")
                  continue

               self.stdout.write(self.style.SUCCESS(f"✅ Guest captured. Guest ID: {guest.id}"))

               description = describe_and_greet(guest.image.path)

               if description:
                  guest.greeting_text = description
                  guest.save()
                  self.stdout.write(self.style.SUCCESS(f"✅ Greeting saved for Guest {guest.id}: {description}"))
               else:
                  self.stderr.write(f"❌ Failed to generate greeting for Guest {guest.id}")
               time.sleep(0.1)

      finally:
         cap.release()
         cv2.destroyAllWindows()
