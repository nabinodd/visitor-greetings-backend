import cv2
import numpy as np
from ultralytics import YOLO

# Constants
BLUR_THRESHOLD = 40  # Adjust based on your requirements
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections

# Load YOLOv10 model
model = YOLO("yolov10n.pt")  # Ensure the model file is in the working directory

def is_blurry(image, threshold=BLUR_THRESHOLD):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
   return lap_var < threshold, lap_var

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change the index if multiple cameras are connected
if not cap.isOpened():
   print("Error: Cannot open webcam.")
   exit()

# Warm-up frames
for _ in range(30):
   ret, _ = cap.read()

print("Camera ready. Looking for a sharp person...")

while True:
   ret, frame = cap.read()
   if not ret:
      break

   # Perform person detection
   results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False, device = 'mps')  # Class 0 corresponds to 'person'

   for result in results:
      for box in result.boxes:
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         person_crop = frame[y1:y2, x1:x2]
         blurry, sharpness = is_blurry(person_crop)

         # Draw bounding box and sharpness score
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
         cv2.putText(frame, f'Sharpness: {sharpness:.2f}', (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

         if not blurry:
               cv2.imwrite("captured_person.jpg", person_crop)
               print(f"Captured sharp person! Sharpness: {sharpness:.2f}")
               cap.release()
               cv2.destroyAllWindows()
               exit()

   cv2.imshow("USYC_2025", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
