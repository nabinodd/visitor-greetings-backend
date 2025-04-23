import os
from io import BytesIO

import cv2
import numpy as np
from deepface import DeepFace
from django.core.files.base import ContentFile
from django.db.models.signals import post_delete, post_save, pre_save
from django.dispatch import receiver
from PIL import Image

from .models import Visitor, Log, Guest
from .utils import detect_and_crop_single_face


@receiver(pre_save, sender=Visitor)
def delete_old_files_on_image_change(sender, instance, **kwargs):
   """Delete old image and cropped image if image has changed."""
   if not instance.pk:
      return  # New object, nothing to delete yet

   try:
      old = Visitor.objects.get(pk=instance.pk)

      if old.image and instance.image and old.image != instance.image:
         # Delete old original image and cropped face
         old.image.delete(save=False)
         if old.image_cropped:
               old.image_cropped.delete(save=False)

         # Mark for recalculation
         instance.calc_emb = False

   except Visitor.DoesNotExist:
      pass


@receiver(post_save, sender=Visitor)
def create_cropped_and_embedding(sender, instance, created, **kwargs):
   """Generate embedding and cropped image after save, if needed."""
   if not instance.calc_emb and instance.image:
      try:
         face_img, error_msg = detect_and_crop_single_face(instance.image.path)

         if error_msg:
               print(f"[!] Skipping embedding: {error_msg}")
               return

         # Save cropped image
         face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
         buffer = BytesIO()
         face_pil.save(buffer, format='JPEG')
         image_content = ContentFile(buffer.getvalue())

         instance.image_cropped.save(
               f"cropped_{os.path.basename(instance.image.name)}",
               image_content,
               save=False
         )

         # Generate embedding
         embedding_result = DeepFace.represent(
               img_path=np.array(face_img),
               model_name='ArcFace',
               enforce_detection=False,
               detector_backend='skip'
         )
         instance.embedding = embedding_result[0]["embedding"]
         instance.calc_emb = True
         instance.save()

         print("[✓] Embedding and cropped image updated.")

      except Exception as e:
         print(f"[!] Error generating embedding: {e}")


@receiver(post_delete, sender=Visitor)
def delete_visitor_images(sender, instance, **kwargs):
   """Cleanup both original and cropped images when Visitor is deleted."""
   if instance.image:
      instance.image.delete(save=False)
   if instance.image_cropped:
      instance.image_cropped.delete(save=False)

@receiver(post_delete, sender=Log)
def delete_log_images(sender, instance, **kwargs):
   """Cleanup both original and cropped images when Visitor is deleted."""
   if instance.image:
      instance.image.delete(save=False)


@receiver(post_delete, sender=Guest)
def delete_guest_image(sender, instance, **kwargs):
   if instance.image:
      instance.image.delete(save=False)