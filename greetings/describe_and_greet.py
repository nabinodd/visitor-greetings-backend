import base64
import json

import cv2
import numpy as np
import requests
import sounddevice as sd
from piper.voice import PiperVoice

from .configurations import (API_KEY, API_URL, DEFAULT_PAYLOAD,
                             IMAGE_RESOLUTION, PIPER_MODEL_PATH, SYSTEM_PROMPT,
                             USER_PROMPT_TEMPLATE)


def preprocess_image(image_path, max_size=IMAGE_RESOLUTION):
   """Load and resize image, then encode as base64."""
   img = cv2.imread(image_path)
   height, width = img.shape[:2]
   scale = min(max_size / float(height), max_size / float(width))
   resized = cv2.resize(img, (int(width * scale), int(height * scale)))
   _, buffer = cv2.imencode('.jpg', resized)
   return base64.b64encode(buffer).decode('utf-8')

def generate_description(image_path):
   """Send image to OpenAI API and get a flattering description."""
   base64_image = preprocess_image(image_path)
   user_prompt = USER_PROMPT_TEMPLATE  # No formatting needed

   payload = {
      "model": DEFAULT_PAYLOAD["model"],
      "temperature": DEFAULT_PAYLOAD["temperature"],
      "response_format": DEFAULT_PAYLOAD["response_format"],
      "messages": [
         {"role": "system", "content": SYSTEM_PROMPT},
         {
               "role": "user",
               "content": [
                  {"type": "text", "text": user_prompt},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
               ]
         }
      ]
   }

   headers = {
      "Authorization": f"Bearer {API_KEY}",
      "Content-Type": "application/json"
   }

   response = requests.post(API_URL, headers=headers, json=payload)

   if response.status_code == 200:
      try:
         description = json.loads(response.json()["choices"][0]["message"]["content"])["description"]
         return description
      except (KeyError, json.JSONDecodeError) as e:
         print(f"❌ Error parsing GPT response: {e}")
         return None
   else:
      print(f"❌ GPT API error: {response.status_code}\n{response.text}")
      return None

def speak(text):
   """Speak the given text using Piper TTS."""
   voice = PiperVoice.load(PIPER_MODEL_PATH)
   print(f"[TTS] {text}")
   stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
   stream.start()
   for audio_bytes in voice.synthesize_stream_raw(text):
      stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
   stream.stop()
   stream.close()

def describe_and_greet(image_path):
   """Generate a description and speak it."""
   description = generate_description(image_path)
   if description:
      speak(description)
   return description
