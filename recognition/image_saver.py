import os

import cv2
from django.conf import settings
from django.utils.timezone import now

from visitors.models import Log


def save_recognized_image(frame, visitor):
   # Create directory if not exists
   save_dir = os.path.join(settings.MEDIA_ROOT, 'visitor_photos')
   os.makedirs(save_dir, exist_ok=True)

   # Generate filename
   timestamp = now().strftime("%Y%m%d_%H%M%S")
   filename = f"{visitor.name.replace(' ', '_')}_{timestamp}.jpg"
   filepath = os.path.join(save_dir, filename)

   # Save image
   cv2.imwrite(filepath, frame)

   # Create Log with image field
   rel_path = f"visitor_photos/{filename}"  # relative to MEDIA_ROOT
   log = Log.objects.create(
      visitor=visitor,
      reg_datetime=now(),
      remarks="Identified in frame",
      image=rel_path
   )

   return log, filepath
