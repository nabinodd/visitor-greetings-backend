import cv2

common_resolutions = [
   (1920, 1080),
   (1280, 720),
   (1024, 768),
   (800, 600),
   (640, 480),
   (320, 240)
]

cap = cv2.VideoCapture(0)

print("[INFO] Testing supported resolutions...")
for width, height in common_resolutions:
   fps_options = [15, 30, 60, 120]
   for fps in fps_options:
      cap.set(cv2.CAP_PROP_FPS, fps)
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

      actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      actual_fps = cap.get(cv2.CAP_PROP_FPS)

      if (actual_width, actual_height) == (width, height):
         print(f"[SUPPORTED] {width}x{height} @ {fps}fps")
      else:
         print(f"[NOT SUPPORTED] Tried {width}x{height} @ {fps}fps, got {actual_width}x{actual_height} @ {actual_fps}fps")

cap.release()