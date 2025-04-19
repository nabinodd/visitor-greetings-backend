import tempfile

import numpy as np
from deepface import DeepFace
from django.core.files.uploadedfile import InMemoryUploadedFile
from numpy.linalg import norm
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Visitor
from rest_framework.permissions import AllowAny

def cosine_similarity(vec1, vec2):
   vec1 = np.array(vec1)
   vec2 = np.array(vec2)
   return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


class IdentifyVisitorAPIView(APIView):
   permission_classes = (AllowAny,)
   def post(self, request):
      image = request.FILES.get('image')

      if not image:
         return Response({'error': 'Image is required.'}, status=status.HTTP_400_BAD_REQUEST)

      # Save uploaded image to temp file
      with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
         for chunk in image.chunks():
               temp_file.write(chunk)
         temp_file.flush()

         try:
               embedding_data = DeepFace.represent(
                  img_path=temp_file.name,
                  model_name='ArcFace',
                  enforce_detection=False
               )

               query_embedding = embedding_data[0]['embedding']
         except Exception as e:
               return Response({'error': f'Error generating embedding: {str(e)}'}, status=500)

      best_match = None
      best_score = -1
      threshold = 0.7  # Adjust based on testing

      for visitor in Visitor.objects.all():
         try:
            db_embedding = np.array(visitor.embedding)
            if db_embedding.size == 512:
                  score = cosine_similarity(query_embedding, db_embedding)
                  if score > best_score:
                     best_score = score
                     best_match = visitor
         except Exception as e:
            continue

      if best_score >= threshold:
         return Response({'name': best_match.name, 'score': best_score}, status=200)
      else:
         return Response({'message': 'No match found', 'score': best_score}, status=404)
