from django.db import models
from pgvector.django import VectorField

from server_visitor_greetings.models import TimestampedModel


class Visitor(TimestampedModel):
   name = models.CharField(max_length=255)
   image = models.ImageField(upload_to='visitor_images/')
   image_cropped = models.ImageField(
      upload_to='visitor_faces/',
      blank=True, null=True
   )
   embedding = VectorField(
      dimensions=512, blank=True,
      null = True
   )
   calc_emb = models.BooleanField(default=False)

   class Meta:
      ordering = ('id',)

   def __str__(self) -> str:
      return self.name


class Log(TimestampedModel):
   visitor = models.ForeignKey('Visitor', on_delete=models.CASCADE)
   reg_datetime = models.DateTimeField(null=True, blank=True)
   remarks = models.CharField(max_length=255, null=True, blank=True)
   image = models.ImageField(
      upload_to='visitor_photos/', 
      null=True, blank=True
   )

   class Meta:
      ordering = ('id',)

   def __str__(self):
      return self.visitor.name