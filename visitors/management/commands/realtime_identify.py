import cv2
import numpy as np
from deepface import DeepFace
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from pgvector.django import CosineDistance

from visitors.models import Log, Visitor

# Load DNN face detection model
dnn_net = cv2.dnn.readNetFromCaffe(
   "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

class Command(BaseCommand):
   help = "Run real-time visitor recognition via webcam"

   def handle(self, *args, **kwargs):
      cap = cv2.VideoCapture(0)
      print("[INFO] Webcam feed started. Press 'q' to quit.")

      while True:
         ret, frame = cap.read()
         if not ret:
               break

         h, w = frame.shape[:2]
         blob = cv2.dnn.blobFromImage(
               frame, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False
         )
         dnn_net.setInput(blob)
         detections = dnn_net.forward()

         faces = []
         boxes = []

         for i in range(detections.shape[2]):
               confidence = detections[0, 0, i, 2]
               if confidence > 0.90:
                  box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                  (startX, startY, endX, endY) = box.astype("int")
                  face = frame[startY:endY, startX:endX]
                  if face.size > 0:
                     faces.append(face)
                     boxes.append((startX, startY, endX, endY, confidence))

         if len(faces) != 1:
               cv2.putText(
                  frame, "more than one person",
                  (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
               )
         else:
               face_img = faces[0]
               (startX, startY, endX, endY, score) = boxes[0]

               try:
                  embedding_result = DeepFace.represent(
                     img_path=np.array(face_img),
                     model_name='ArcFace',
                     enforce_detection=False,
                     detector_backend='skip'
                  )
                  face_embedding = embedding_result[0]["embedding"]

                  # Use pgvector to find best match by cosine distance
                  match = Visitor.objects.annotate(
                     distance=CosineDistance('embedding', face_embedding)
                  ).order_by('distance').first()

                  # Set threshold for cosine distance
                  print(match, match.distance)
                  if match and match.distance <= 0.3:
                     name = match.name

                     # Overlay name
                     cv2.putText(
                           frame, name, (startX, endY + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                     )

                     # Log only once
                     if not Log.objects.filter(visitor=match).exists():
                           Log.objects.create(
                              visitor=match,
                              reg_datetime=now(),
                              remarks="Identified via webcam"
                           )
                           print(f"[LOGGED] {name} at {now().strftime('%Y-%m-%d %H:%M:%S')}")

               except Exception as e:
                  print(f"[!] Error during embedding or lookup: {e}")

               # Draw the bounding box and confidence score
               label = f"{score:.2f}"
               cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
               cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

         cv2.imshow("Visitor Identification", frame)

         if cv2.waitKey(1) & 0xFF == ord("q"):
               break

      cap.release()
      cv2.destroyAllWindows()
