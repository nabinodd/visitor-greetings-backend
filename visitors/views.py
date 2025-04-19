import tempfile

import numpy as np
from deepface import DeepFace
from pgvector.django import CosineDistance
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Visitor
from .utils import detect_and_crop_single_face


class IdentifyVisitorAPIView(APIView):
   permission_classes = (AllowAny,)

   def post(self, request):
      image = request.FILES.get('image')

      if not image:
         return Response({'error': 'Image is required.'}, status=status.HTTP_400_BAD_REQUEST)

      with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
         for chunk in image.chunks():
               temp_file.write(chunk)
         temp_file.flush()

         face_img, error_msg = detect_and_crop_single_face(temp_file.name)

         if error_msg:
               return Response({'error': error_msg}, status=status.HTTP_400_BAD_REQUEST)

         try:
               embedding_data = DeepFace.represent(
                  img_path=np.array(face_img),
                  model_name='ArcFace',
                  enforce_detection=False,
                  detector_backend='skip'
               )
               query_embedding = embedding_data[0]['embedding']
         except Exception as e:
               return Response({'error': f'Error generating embedding: {str(e)}'}, status=500)

      match = Visitor.objects.annotate(
         distance=CosineDistance('embedding', query_embedding)
      ).order_by('distance').first()

      if match and match.distance <= 0.3:  # cosine distance threshold ~0.7 similarity
         return Response({'name': match.name, 'score': 1 - match.distance}, status=200)
      else:
         return Response({'message': 'No match found'}, status=404)
