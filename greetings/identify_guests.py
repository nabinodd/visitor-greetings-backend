from io import BytesIO

import cv2
import numpy as np
from deepface import DeepFace
from django.core.files.base import ContentFile
from pgvector.django import CosineDistance

from visitors.models import Log, Visitor

from .configurations import now

# Face detection model (OpenCV DNN)
DNN_PROTO_PATH = 'deploy.prototxt'
DNN_MODEL_PATH = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

# Thresholds
FACE_DETECTION_CONFIDENCE = 0.8
FACE_IDENTIFICATION_THRESHOLD = 0.4

# Load face detector once
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)


def detect_face(person_crop):
   """
   Detects a face in the person_crop image using OpenCV DNN.
   Returns the face crop and box if found, else None.
   """
   (h, w) = person_crop.shape[:2]
   blob = cv2.dnn.blobFromImage(person_crop, 1.0, (300, 300), (104.0, 177.0, 123.0))
   dnn_net.setInput(blob)
   detections = dnn_net.forward()

   for i in range(detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > FACE_DETECTION_CONFIDENCE:
         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
         (x1, y1, x2, y2) = box.astype("int")
         face_crop = person_crop[y1:y2, x1:x2]
         return face_crop
   return None

def identify_guest(person_crop):
   """
   Detects, embeds, matches, and logs the guest if identified.
   """
   face_crop = detect_face(person_crop)
   if face_crop is None:
      print(f"[ERROR @ {now()}] FACE NOT FOUND] No face detected.")
      return None

   embedding = DeepFace.represent(face_crop, model_name='ArcFace', enforce_detection=False)[0]["embedding"]

   visitors = Visitor.objects.annotate(distance=CosineDistance('embedding', embedding)).order_by('distance')
   if visitors.exists() and visitors.first().distance <= FACE_IDENTIFICATION_THRESHOLD:
      visitor = visitors.first()

      # Convert person_crop to Django ContentFile for Log image
      _, buffer = cv2.imencode('.jpg', person_crop)
      io_buffer = BytesIO(buffer.tobytes())
      django_file = ContentFile(io_buffer.getvalue(), name='log_capture.jpg')

      # Create log with image
      log, created = Log.objects.get_or_create(visitor=visitor, image=django_file)
      if created:
         log.image = django_file
         log.save()

      print(f"[SUCCESS @ {now()}] IDENTIFIED] {visitor.name} has been logged with image.")
      return visitor
   else:
      print(f"[ERROR @ {now()}] NOT IDENTIFIED] Face detected but no matching visitor.")
      return None
