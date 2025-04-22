import time
from threading import Thread

import cv2
from ultralytics import YOLO

from recognition.descriptor import describe_and_greet
from recognition.face_utils import detect_and_match_face
from recognition.image_saver import save_recognized_image
from visitors.models import Log

CAMERA_SOURCE = 0
CONF_THRESHOLD = 0.8
IOU_THRESHOLD = 0.3
CLASSES = [0]

MIN_WIDTH = 150
MIN_HEIGHT = 250
ROTATE_FRAME = True  # Rotate to portrait

def run_recognition_pipeline():
   model = YOLO("yolov10n.pt")
   cap = cv2.VideoCapture(CAMERA_SOURCE)

   actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   print(actual_width, actual_height)

   tracked_names = {}

   print("[INFO] Visitor recognition started. Press 'q' to quit.")

   while True:
      time.sleep(0.01)
      ret, frame = cap.read()
      if not ret:
         break

      # Rotate frame to portrait orientation
      if ROTATE_FRAME:
         frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

      results = model.track(
         frame, persist=False, conf=CONF_THRESHOLD,
         iou=IOU_THRESHOLD, classes=CLASSES, verbose=False,
         device='mps'
      )
      boxes = results[0].boxes

      if boxes.id is None or len(boxes) == 0:
         # No persons at all
         if tracked_names:
               print("[INFO] Forgetting all IDs due to no person detected")
               tracked_names.clear()
         cv2.imshow("Visitor Recognition", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
               break
         continue

      person_boxes = []
      far_boxes = []

      # Separate valid and far persons
      for box in boxes:
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         w, h = x2 - x1, y2 - y1
         if w >= MIN_WIDTH and h >= MIN_HEIGHT:
               person_boxes.append((box, w * h))  # valid with area
         else:
               far_boxes.append(box)

      # Draw red boxes for far persons
      for box in far_boxes:
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

      # Forget IDs unless exactly one valid person
      if len(person_boxes) != 1:
         if tracked_names:
               reason = 'multiple persons' if len(person_boxes) > 1 else 'no valid person'
               print(f"[INFO] Forgetting all IDs due to {reason}")
               tracked_names.clear()

      # No valid person
      if len(person_boxes) == 0:
         cv2.imshow("Visitor Recognition", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
               break
         continue

      # Multiple valid persons
      if len(person_boxes) > 1:
         frame = cv2.blur(frame, (30, 30))
         for box, _ in person_boxes:
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
         print('Multiple persons')
         cv2.imshow("Visitor Recognition", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
               break
         continue

      # Exactly one valid person
      target_box, _ = person_boxes[0]

      # Blur far persons (but keep red boxes)
      for box in far_boxes:
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (30, 30))
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

      # Process target box
      pid = int(target_box.id[0])
      x1, y1, x2, y2 = map(int, target_box.xyxy[0])
      crop = frame[y1:y2, x1:x2]

      if crop.size == 0:
         continue

      # Already recognized?
      if tracked_names.get(pid) not in [None, "Unknown"]:
         name = tracked_names[pid]
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
         cv2.putText(frame, name, (x1 + 20, y1 + 80),
                     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
         cv2.imshow("Visitor Recognition", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
               break
         continue

      # Face recognition
      match, face_box = detect_and_match_face(crop, x1, y1)
      if match:
         if Log.objects.filter(visitor=match).exists():
               tracked_names[pid] = match.name
               continue

         log_instance, image_path = save_recognized_image(frame, match)
         Thread(target=describe_and_greet, args=(image_path, match.name)).start()

         tracked_names[pid] = match.name
         print(f"[LOGGED] {match.name}")
      else:
         tracked_names[pid] = "Unknown"
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
         cv2.putText(frame, 'waiting for recognition...', (x1 + 20, y1 + 80),
                     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

      cv2.imshow("Visitor Recognition", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
         break

   cap.release()
   cv2.destroyAllWindows()
