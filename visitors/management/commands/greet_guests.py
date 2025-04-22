import time

from django.core.management.base import BaseCommand

from greetings.capture import capture_guest_image
from greetings.describe_and_greet import describe_and_greet
from visitors.models import Guest


class Command(BaseCommand):
   help = 'Captures a guest image, generates a description, and greets the guest via TTS'

   def handle(self, *args, **options):
      self.stdout.write("Starting guest capture process...")
      while True:
         # Step 1: Capture image and save Guest instance
         guest = capture_guest_image()

         if not guest:
            self.stderr.write("❌ No guest captured.")
            return

         self.stdout.write(self.style.SUCCESS(f"✅ Guest image captured. Guest ID: {guest.id}"))

         # Step 2: Generate description and greet
         description = describe_and_greet(guest.image.path)

         if description:
            # Step 3: Save the description back to the Guest model
            guest.greeting_text = description
            guest.save()
            self.stdout.write(self.style.SUCCESS(f"✅ Greeting generated and saved for Guest {guest.id}: {description}"))
         else:
            self.stderr.write(f"❌ Failed to generate greeting for Guest {guest.id}")

         time.sleep(1)