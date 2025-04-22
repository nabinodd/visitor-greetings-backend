import cv2
import numpy as np
from deepface import DeepFace
from pgvector.django import CosineDistance

from visitors.models import Visitor

DNN_NET = cv2.dnn.readNetFromCaffe(
   "deploy.prototxt",
   "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

FACE_DETECTION_THRESHOLD = 0.9
FACE_IDENTIFICATION_THRESHOLD = 0.6

def detect_and_match_face(person_crop, offset_x=0, offset_y=0):
   blob = cv2.dnn.blobFromImage(person_crop, 1.0, (300, 300),
                              [104.0, 177.0, 123.0], swapRB=False, crop=False)
   DNN_NET.setInput(blob)
   detections = DNN_NET.forward()

   max_conf = 0
   best_box = None

   for i in range(detections.shape[2]):
      conf = detections[0, 0, i, 2]
      if conf > FACE_DETECTION_THRESHOLD and conf > max_conf:
         box = detections[0, 0, i, 3:7] * np.array([
               person_crop.shape[1], person_crop.shape[0],
               person_crop.shape[1], person_crop.shape[0]
         ])
         (fx1, fy1, fx2, fy2) = box.astype("int")
         fx1, fy1 = max(0, fx1), max(0, fy1)
         fx2, fy2 = min(fx2, person_crop.shape[1]), min(fy2, person_crop.shape[0])
         best_box = (fx1, fy1, fx2, fy2)
         max_conf = conf

   if best_box:
      fx1, fy1, fx2, fy2 = best_box
      face_crop = person_crop[fy1:fy2, fx1:fx2]

      try:
         embedding = DeepFace.represent(
               img_path=face_crop,
               model_name='ArcFace',
               enforce_detection=False,
               detector_backend='skip'
         )[0]['embedding']

         match = Visitor.objects.annotate(
               distance=CosineDistance('embedding', embedding)
         ).order_by('distance').first()

         if match and match.distance <= FACE_IDENTIFICATION_THRESHOLD:
               return match, (offset_x + fx1, offset_y + fy1, offset_x + fx2, offset_y + fy2)

      except Exception as e:
         print(f"[!] DeepFace Error: {e}")

   return None, None
