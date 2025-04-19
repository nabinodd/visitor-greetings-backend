import os
from io import BytesIO

import cv2
import numpy as np
from deepface import DeepFace
from django.core.files.base import ContentFile
from django.db import models
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from pgvector.django import VectorField
from PIL import Image

from server_visitor_greetings.models import TimestampedModel

from .utils import detect_and_crop_single_face


class Visitor(TimestampedModel):
   name = models.CharField(max_length=255)
   image = models.ImageField(upload_to='visitor_images/')
   image_cropped = models.ImageField(
      upload_to='visitor_faces/', 
      blank=True, null=True
   )
   embedding = VectorField(dimensions=512, blank=True)
   calc_emb = models.BooleanField(default=False)

   class Meta:
      ordering = ('id',)

   def __str__(self) -> str:
      return self.name

   def save(self, *args, **kwargs):
      if not self.id:
         self.embedding = [0.0] * 512
      return super().save(*args, **kwargs)

@receiver(post_save, sender=Visitor)
def create_embeddings(sender, instance, created, **kwargs):
   if not instance.calc_emb and instance.image:
      try:
         face_img, error_msg = detect_and_crop_single_face(instance.image.path)

         if error_msg:
               print(f"[!] Skipping embedding: {error_msg}")
               return

         face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
         buffer = BytesIO()
         face_pil.save(buffer, format='JPEG')
         image_content = ContentFile(buffer.getvalue())

         instance.image_cropped.save(f"cropped_{os.path.basename(instance.image.name)}", image_content, save=False)

         embedding_result = DeepFace.represent(
               img_path=np.array(face_img),
               model_name='ArcFace',
               enforce_detection=True,
               detector_backend='skip'
         )
         instance.embedding = embedding_result[0]["embedding"]
         instance.calc_emb = True
         instance.save()

         print("[✓] Embedding saved successfully.")

      except Exception as e:
         print(f"[!] Error generating embedding: {e}")

@receiver(post_delete, sender=Visitor)
def delete_image_cropped(sender, instance, **kwargs):
   if instance.image:
      instance.image.delete(save=False)

   if instance.image_cropped:
      instance.image_cropped.delete(save=False)



class Log(TimestampedModel):
   visitor = models.ForeignKey(Visitor, on_delete = models.CASCADE)
   reg_datetime = models.DateTimeField(null = True, blank = True)
   remarks = models.CharField(max_length = 255, null = True, blank = True)

   class Meta:
      ordering = ('id',)

   def __str__(self) -> str:
      return self.visitor.name