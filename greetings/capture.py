from io import BytesIO

import cv2
import numpy as np
from django.core.files.base import ContentFile
from ultralytics import YOLO

from visitors.models import Guest

from .configurations import (DNN_FACE_DETECTION_CONFIDENCE,
                             ENABLE_SIZE_REPORTING, FACE_BLUR_THRESHOLD,
                             GPU_ACCELERATION, HEIGHT_THRESHOLD,
                             PERSON_BLUR_THRESHOLD, WIDTH_THRESHOLD,
                             YOLO_MODEL_PATH, YOLO_PERSON_CONFIDENCE_THRESHOLD)
from .identify_guests import identify_guest

# Face detection model (OpenCV DNN)
DNN_PROTO_PATH = 'deploy.prototxt'
DNN_MODEL_PATH = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)

CENTER_OVERLAP_THRESHOLD = 0.9
SAFE_ZONE_MARGIN = 0.3

def load_model():
   return YOLO(YOLO_MODEL_PATH)

def initialize_camera(camera_index=0):
   cap = cv2.VideoCapture(camera_index)
   if not cap.isOpened():
      raise RuntimeError("Error: Cannot open webcam.")
   return cap

def warmup_camera(cap, frames=30):
   for _ in range(frames):
      cap.read()

def calculate_sharpness(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_guest_image(image):
   _, buffer = cv2.imencode('.jpg', image)
   io_buffer = BytesIO(buffer.tobytes())
   django_file = ContentFile(io_buffer.getvalue(), name='guest_capture.jpg')
   guest = Guest.objects.create(image=django_file)
   return guest

def detect_face(person_crop):
   (h, w) = person_crop.shape[:2]
   blob = cv2.dnn.blobFromImage(person_crop, 1.0, (300, 300), (104.0, 177.0, 123.0))
   dnn_net.setInput(blob)
   detections = dnn_net.forward()
   for i in range(detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > DNN_FACE_DETECTION_CONFIDENCE:
         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
         (x1, y1, x2, y2) = box.astype("int")
         face_crop = person_crop[y1:y2, x1:x2]
         return face_crop, (x1, y1, x2, y2)
   return None, None

def calculate_overlap_ratio(box1, box2):
   x1 = max(box1[0], box2[0])
   y1 = max(box1[1], box2[1])
   x2 = min(box1[2], box2[2])
   y2 = min(box1[3], box2[3])
   overlap_width = max(0, x2 - x1)
   overlap_height = max(0, y2 - y1)
   overlap_area = overlap_width * overlap_height
   box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
   return overlap_area / box1_area if box1_area > 0 else 0

def capture_guest_image(cap, model):
   """Capture a sharp face in the center using an existing camera and model."""
   while True:
      ret, frame = cap.read()
      if not ret:
         continue
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
      display_frame = frame.copy()
      frame_h, frame_w = frame.shape[:2]

      # Define center safe zone
      sz_x1 = int(frame_w * SAFE_ZONE_MARGIN)
      sz_y1 = int(frame_h * SAFE_ZONE_MARGIN)
      sz_x2 = int(frame_w * (1 - SAFE_ZONE_MARGIN))
      sz_y2 = int(frame_h * (1 - SAFE_ZONE_MARGIN))
      cv2.rectangle(display_frame, (sz_x1, sz_y1), (sz_x2, sz_y2), (0, 255, 255), 2)

      close_persons = []
      far_persons = []

      # Person detection
      results = model(frame, classes=[0], conf=YOLO_PERSON_CONFIDENCE_THRESHOLD, verbose=False, device=GPU_ACCELERATION)
      for result in results:
         for box in result.boxes:
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               width, height = x2 - x1, y2 - y1
               person_crop = frame[y1:y2, x1:x2]
               if ENABLE_SIZE_REPORTING:
                  print(width, height)
               if width >= WIDTH_THRESHOLD and height >= HEIGHT_THRESHOLD:
                  close_persons.append((x1, y1, x2, y2, person_crop))
               else:
                  far_persons.append((x1, y1, x2, y2))

      # Handle far persons (blur + red boxes)
      for x1, y1, x2, y2 in far_persons:
         person_area = display_frame[y1:y2, x1:x2]
         blurred = cv2.GaussianBlur(person_area, (51, 51), 0)
         display_frame[y1:y2, x1:x2] = blurred
         cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

      # Handle close persons
      if len(close_persons) == 1:
         x1, y1, x2, y2, person_crop = close_persons[0]
         sharpness = calculate_sharpness(person_crop)

         person_color = (0, 255, 0) if sharpness >= PERSON_BLUR_THRESHOLD else (0, 0, 255)
         cv2.rectangle(display_frame, (x1, y1), (x2, y2), person_color, 2)
         cv2.putText(display_frame, f'Person Sharpness: {sharpness:.2f}', (x1 + 10, y1 + 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, person_color, 2)

         if sharpness >= PERSON_BLUR_THRESHOLD:
               face_crop, face_box = detect_face(person_crop)
               if face_crop is not None:
                  face_sharpness = calculate_sharpness(face_crop)
                  fx1, fy1, fx2, fy2 = face_box

                  # Check face centering
                  abs_face_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                  face_overlap_ratio = calculate_overlap_ratio(abs_face_box, (sz_x1, sz_y1, sz_x2, sz_y2))
                  face_centered = face_overlap_ratio >= CENTER_OVERLAP_THRESHOLD

                  face_color = (0, 255, 0) if face_sharpness >= FACE_BLUR_THRESHOLD and face_centered else (0, 0, 255)
                  cv2.rectangle(display_frame, (abs_face_box[0], abs_face_box[1]), (abs_face_box[2], abs_face_box[3]), face_color, 2)
                  cv2.putText(display_frame, f'Face Sharpness: {face_sharpness:.2f}', (abs_face_box[0], abs_face_box[3] + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, face_color, 2)

                  if face_sharpness >= FACE_BLUR_THRESHOLD and face_centered:
                     guest = save_guest_image(person_crop)
                     print(f"Captured sharp guest image with sharpness {sharpness:.2f}. Guest ID: {guest.id}")

                     identify_guest(person_crop)
                     cv2.imshow("Preview", display_frame)
                     cv2.waitKey(1000)
                     return guest

      elif len(close_persons) > 1:
         display_frame = cv2.GaussianBlur(display_frame, (51, 51), 0)
         for x1, y1, x2, y2, _ in close_persons:
               cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # Show ongoing preview
      cv2.imshow("Preview", display_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         return 'EXIT'
