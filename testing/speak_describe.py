import base64
import json

import cv2
import numpy as np
import requests
import sounddevice as sd
from piper.voice import PiperVoice

import describe_speak.config as config

IMAGE_PATH = "rsz.jpg"
PROCESSED_IMAGE_PATH = "processed.jpg"

# === Load Piper Voice ===
# Update to your preferred model path
PIPER_MODEL_PATH = "piper-voices/en/hfc_female/medium/en_US-hfc_female-medium.onnx"
voice = PiperVoice.load(PIPER_MODEL_PATH)


def preprocess_image_opencv(input_path, output_path, max_size):
   img = cv2.imread(input_path)
   height, width = img.shape[:2]
   scale = min(max_size / float(height), max_size / float(width))
   new_width = int(width * scale)
   new_height = int(height * scale)
   resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
   cv2.imwrite(output_path, resized)


def encode_image_base64(path):
   with open(path, "rb") as f:
      return base64.b64encode(f.read()).decode("utf-8")


def speak_with_piper(text):
   stream = sd.OutputStream(
      samplerate=voice.config.sample_rate,
      channels=1,
      dtype='int16'
   )
   stream.start()
   for audio_bytes in voice.synthesize_stream_raw(text):
      int_data = np.frombuffer(audio_bytes, dtype=np.int16)
      stream.write(int_data)
   stream.stop()
   stream.close()


def describe_image():
   preprocess_image_opencv(IMAGE_PATH, PROCESSED_IMAGE_PATH, config.image_resolution)
   base64_image = encode_image_base64(PROCESSED_IMAGE_PATH)

   headers = {
      "Authorization": f"Bearer {config.api_key}",
      "Content-Type": "application/json"
   }

   payload = {
      "model": config.default_payload["model"],
      "temperature": config.default_payload["temperature"],
      "response_format": config.default_payload["response_format"],
      "messages": [
         {"role": "system", "content": config.system_prompt},
         {
               "role": "user",
               "content": [
                  {"type": "text", "text": config.user_prompt},
                  {
                     "type": "image_url",
                     "image_url": {
                           "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                  }
               ]
         }
      ]
   }

   response = requests.post(config.api_url, headers=headers, json=payload)

   if response.status_code == 200:
      description = json.loads(response.json()["choices"][0]["message"]["content"])['description']
      speak_with_piper(description)
   else:
      print(f"\n❌ Error {response.status_code}: {response.text}")


if __name__ == "__main__":
   describe_image()
