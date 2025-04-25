import time
from io import BytesIO

import cv2
import numpy as np
from django.core.files.base import ContentFile
from ultralytics import YOLO

from visitors.models import Guest

from .configurations import (BOX_HEIGHT, BOX_WIDTH,
                             DNN_FACE_DETECTION_CONFIDENCE,
                             ENABLE_SIZE_REPORTING, FACE_BLUR_THRESHOLD,
                             FONT_SCALE, FONT_THICKNESS, GPU_ACCELERATION,
                             GUSSAIN_BLUR_KERNEL_SIZE, HEIGHT_THRESHOLD,
                             OVERLAY_ONLY_TIME, PERSON_BLUR_THRESHOLD,
                             WIDTH_THRESHOLD, Y_DOWN, YOLO_MODEL_PATH,
                             YOLO_PERSON_CONFIDENCE_THRESHOLD, now)
from .identify_guests import identify_guest

# Face detection model (OpenCV DNN)
DNN_PROTO_PATH = 'deploy.prototxt'
DNN_MODEL_PATH = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)

CENTER_OVERLAP_THRESHOLD = 0.9

cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


overlay_only_started_time = None

def load_model():
   return YOLO(YOLO_MODEL_PATH)

def initialize_camera(camera_index=0):
   cap = cv2.VideoCapture(camera_index)
   if not cap.isOpened():
      raise RuntimeError(f"[ERROR @ {now()}] Cannot open webcam.")
   return cap

def warmup_camera(cap, frames=30):
   for _ in range(frames):
      cap.read()

def flush_camera(cap, frames=10):
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

def capture_guest_image(cap, model, overlay_only=False):
   global overlay_only_started_time
   if overlay_only and overlay_only_started_time is None:
      overlay_only_started_time = time.time()

   while True:
      if overlay_only_started_time is not None and time.time() - overlay_only_started_time > OVERLAY_ONLY_TIME:
         overlay_only_started_time = None
         break

      ret, frame = cap.read()
      if not ret:
         continue
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
      display_frame = frame.copy()
      frame_h, frame_w = frame.shape[:2]

      # 🔥 Custom center-safe zone box
      box_w = int(frame_w * BOX_WIDTH)
      box_h = int(frame_h * BOX_HEIGHT)
      box_x1 = (frame_w - box_w) // 2
      box_x2 = box_x1 + box_w
      box_y1 = int(frame_h * Y_DOWN)
      box_y2 = box_y1 + box_h
      # Draw the yellow rectangle
      box_color = (0, 255, 255)
      cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)

      # Add "Your face here" text inside the box (above lower breadth)
      text = "Your face here"
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = FONT_SCALE  # Use your defined font scale
      thickness = FONT_THICKNESS
      (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

      # Center horizontally, position above lower breadth
      text_x = box_x1 + (box_w - text_width) // 2
      text_y = box_y2 - 10  # 10 pixels above the bottom edge of the box

      cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, box_color, thickness)

      close_persons = []
      far_persons = []

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

      for x1, y1, x2, y2 in far_persons:
         person_area = display_frame[y1:y2, x1:x2]
         blurred = cv2.GaussianBlur(person_area, GUSSAIN_BLUR_KERNEL_SIZE, 0)
         display_frame[y1:y2, x1:x2] = blurred
         cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

      if len(close_persons) == 1:
         x1, y1, x2, y2, person_crop = close_persons[0]
         sharpness = calculate_sharpness(person_crop)

         person_color = (0, 255, 0) if sharpness >= PERSON_BLUR_THRESHOLD else (0, 0, 255)
         cv2.rectangle(display_frame, (x1, y1), (x2, y2), person_color, 2)
         cv2.putText(display_frame, f'Person Analysis: {sharpness:.2f}', (x1 + 10, y1 + 50),
                     cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, person_color, FONT_THICKNESS)

         if sharpness >= PERSON_BLUR_THRESHOLD:
               face_crop, face_box = detect_face(person_crop)
               if face_crop is not None:
                  face_sharpness = calculate_sharpness(face_crop)
                  fx1, fy1, fx2, fy2 = face_box
                  abs_face_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                  face_overlap_ratio = calculate_overlap_ratio(abs_face_box, (box_x1, box_y1, box_x2, box_y2))
                  face_centered = face_overlap_ratio >= CENTER_OVERLAP_THRESHOLD

                  face_color = (0, 255, 0) if face_sharpness >= FACE_BLUR_THRESHOLD and face_centered else (0, 0, 255)
                  cv2.rectangle(display_frame, (abs_face_box[0], abs_face_box[1]), (abs_face_box[2], abs_face_box[3]), face_color, 2)
                  cv2.putText(display_frame, f'Face Analysis: {face_sharpness:.2f}', (abs_face_box[0], abs_face_box[3] + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, face_color, FONT_THICKNESS)

                  if face_sharpness >= FACE_BLUR_THRESHOLD and face_centered and not overlay_only:
                     cv2.imshow("Preview", display_frame)
                     cv2.waitKey(1000)
                     guest = save_guest_image(person_crop)
                     print(f"[SUCCESS @ {now()}]Captured sharp guest image with sharpness {sharpness:.2f}. Guest ID: {guest.id}")
                     visitor = identify_guest(person_crop)
                     return guest, visitor

      elif len(close_persons) > 1:
         display_frame = cv2.GaussianBlur(display_frame, GUSSAIN_BLUR_KERNEL_SIZE, 0)
         for x1, y1, x2, y2, _ in close_persons:
               cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

      cv2.imshow("Preview", display_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         return 'EXIT', None

   return None, None