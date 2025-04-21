from threading import Thread

import cv2
from ultralytics import YOLO

from recognition.descriptor import describe_and_greet
from recognition.face_utils import detect_and_match_face
from recognition.image_saver import save_recognized_image
from visitors.models import Log

CAMERA_SOURCE = 1
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
CLASSES = [0] 
MIN_WIDTH = 150
MIN_HEIGHT = 250

def run_recognition_pipeline():
   model = YOLO("yolov10n.pt")

   cap = cv2.VideoCapture(CAMERA_SOURCE)
   tracked_names = {}

   print("[INFO] Visitor recognition started. Press 'q' to quit.")

   while True:
      ret, frame = cap.read()
      if not ret:
         break

      results = model.track(
          frame, persist=True, conf=CONF_THRESHOLD, 
          iou=IOU_THRESHOLD, classes=CLASSES, verbose=False, 
          device = 'mps'
      )
      boxes = results[0].boxes

      if boxes.id is None:
         cv2.imshow("Visitor Recognition", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
               break
         continue

      tracked_ids = boxes.id.cpu().tolist()

      for i, box in enumerate(boxes):
         pid = int(box.id[0])
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         w, h = x2 - x1, y2 - y1
         crop = frame[y1:y2, x1:x2]

         if crop.size == 0 or w < MIN_WIDTH or h < MIN_HEIGHT:
               continue

         # Skip already matched
         if tracked_names.get(pid) not in [None, "Unknown"]:
               name = tracked_names[pid]
               cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
               continue

         match, face_box = detect_and_match_face(crop, x1, y1)
         if match:
               # Save frame + Log instance

               if Log.objects.filter(visitor=match).exists():
                  tracked_names[pid] = match.name
                  continue

               log_instance, image_path = save_recognized_image(frame, match)

               # Describe & speak
               Thread(target=describe_and_greet, args=(image_path, match.name)).start()

               tracked_names[pid] = match.name
               print(f"[LOGGED] {match.name}")
         else:
               tracked_names[pid] = "Unknown"

      cv2.imshow("Visitor Recognition", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
         break

   cap.release()
   cv2.destroyAllWindows()
