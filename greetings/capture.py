from io import BytesIO

import cv2
from django.core.files.base import ContentFile
from ultralytics import YOLO

from visitors.models import Guest

from .configurations import (BLUR_THRESHOLD, CONFIDENCE_THRESHOLD,
                             HEIGHT_THRESHOLD, WIDTH_THRESHOLD,
                             YOLO_MODEL_PATH)


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

def capture_guest_image(cap, model):
   """Capture a sharp close person using an existing camera and model."""
   while True:
      ret, frame = cap.read()
      if not ret:
         continue
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
      display_frame = frame.copy()
      close_persons = []
      far_persons = []

      # Detection
      results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False, device='mps')

      for result in results:
         for box in result.boxes:
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               width, height = x2 - x1, y2 - y1
               person_crop = frame[y1:y2, x1:x2]

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

         # Green box + sharpness
         cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
         cv2.putText(display_frame, f'Sharpness: {sharpness:.2f}', (x1, y1 + 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

         if sharpness >= BLUR_THRESHOLD:
               guest = save_guest_image(person_crop)
               print(f"Captured sharp guest image with sharpness {sharpness:.2f}. Guest ID: {guest.id}")

               # Show last frame (predictable display before TTS)
               cv2.imshow("Preview", display_frame)
               cv2.waitKey(1000)  # Display for 1 second

               return guest

      elif len(close_persons) > 1:
         display_frame = cv2.GaussianBlur(display_frame, (51, 51), 0)
         for x1, y1, x2, y2, _ in close_persons:
               cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # Show ongoing preview
      cv2.imshow("Preview", display_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         return 'EXIT'
