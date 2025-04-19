import cv2

dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def detect_and_crop_single_face(image_path):
   image = cv2.imread(image_path)
   h, w = image.shape[:2]

   blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
   dnn_net.setInput(blob)
   detections = dnn_net.forward()

   faces = []

   for i in range(detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > 0.95:
         box = detections[0, 0, i, 3:7] * [w, h, w, h]
         (startX, startY, endX, endY) = box.astype("int")
         face = image[startY:endY, startX:endX]
         if face.size > 0:
               faces.append(face)

   if len(faces) != 1:
      return None, f"Expected 1 face, found {len(faces)}"
   return faces[0], None