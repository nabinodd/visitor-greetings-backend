from deepface import DeepFace
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from pgvector.django import VectorField

from server_visitor_greetings.models import TimestampedModel


class Visitor(TimestampedModel):
   name = models.CharField(max_length=255)
   image = models.ImageField(upload_to='visitor_images/')
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


@receiver(post_save, sender = Visitor)
def create_embeddings(sender, instance, created, **kwargs):
   if not instance.calc_emb:
      try:
         img_path = instance.image.path

         embedding_dict = DeepFace.represent(img_path=img_path, model_name='ArcFace', enforce_detection=False)
         embedding_vector = embedding_dict[0]["embedding"]

         instance.embedding = embedding_vector
         instance.calc_emb = True
         instance.save()

      except Exception as e:
         print(f"Error generating embedding: {e}")


class Log(TimestampedModel):
   visitor = models.ForeignKey(Visitor, on_delete = models.CASCADE)
   reg_datetime = models.DateTimeField(null = True, blank = True)
   remarks = models.CharField(max_length = 255, null = True, blank = True)

   class Meta:
      ordering = ('id',)

   def __str__(self) -> str:
      return self.visitor.name