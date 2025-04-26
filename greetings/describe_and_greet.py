import base64
import json
import random

import cv2
import numpy as np
import requests
import requests.exceptions
import sounddevice as sd
from piper.voice import PiperVoice

from .configurations import (API_KEY, API_TIMEOUT, API_URL, DEFAULT_PAYLOAD,
                             DEFAULT_SYSTEM_PROMPT,
                             DEFAULT_USER_PROMPT_TEMPLATE,
                             DEFAULT_VISITOR_SYSTEM_PROMPT,
                             DEFAULT_VISITOR_USER_PROMPT_TEMPLATE,
                             IMAGE_RESOLUTION, PIPER_MODEL_PATH,
                             SPECIAL_VISITOR_USER_PROMPT_TEMPLATE,
                             SPECIALT_VISITOR_SYSTEM_PROMPT, now)

# Prefetched fallback greetings
prefetched_greetings = [
   'Hey there, Welcome to U-S-Y-C 2025',
   'Hello, Welcome to U-S-Y-C 2025',
   'Hey!! Welcome to U-S-Y-C 2025',
   'Welcome aboard! U-S-Y-C 2025 is ready to dazzle you.',
   'Hey there! Step into the excitement of U-S-Y-C 2025.',
   'Hello and welcome! Let U-S-Y-C 2025 amaze you.',
   "Hey!! You've arrived at the heart of U-S-Y-C 2025!",
   'Greetings! Get ready for the magic of U-S-Y-C 2025.',
   'Welcome! The U-S-Y-C 2025 experience begins now.',
   'Hello there! U-S-Y-C 2025 is thrilled to have you.',
   'Hey you! Welcome to the unforgettable U-S-Y-C 2025.',
   'Step right in! U-S-Y-C 2025 awaits your brilliance.',
   'Welcome to U-S-Y-C 2025, where the energy is electric!',
]

def preprocess_image(image_path, max_size=IMAGE_RESOLUTION):
   """Load and resize image, then encode as base64."""
   img = cv2.imread(image_path)
   height, width = img.shape[:2]
   scale = min(max_size / float(height), max_size / float(width))
   resized = cv2.resize(img, (int(width * scale), int(height * scale)))
   _, buffer = cv2.imencode('.jpg', resized)
   return base64.b64encode(buffer).decode('utf-8')

def generate_description(image_path, visitor):
   """Send image to OpenAI API and get a flattering description or fallback greeting."""
   base64_image = preprocess_image(image_path)

   if visitor is not None and not visitor.addressing:
      api_timeout = 10
      SYSTEM_PROMPT = DEFAULT_VISITOR_SYSTEM_PROMPT
      USER_PROMPT_TEMPLATE = DEFAULT_VISITOR_USER_PROMPT_TEMPLATE.format(name=visitor.name)

   elif visitor is not None and visitor.addressing:
      api_timeout = 10
      SYSTEM_PROMPT = SPECIALT_VISITOR_SYSTEM_PROMPT
      USER_PROMPT_TEMPLATE = SPECIAL_VISITOR_USER_PROMPT_TEMPLATE.format(name=visitor.name, addressing=visitor.addressing)

   else:
      api_timeout = API_TIMEOUT
      SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
      USER_PROMPT_TEMPLATE = DEFAULT_USER_PROMPT_TEMPLATE

   payload = {
      "model": DEFAULT_PAYLOAD["model"],
      "temperature": DEFAULT_PAYLOAD["temperature"],
      "response_format": DEFAULT_PAYLOAD["response_format"],
      "messages": [
         {"role": "system", "content": SYSTEM_PROMPT},
         {
               "role": "user",
               "content": [
                  {"type": "text", "text": USER_PROMPT_TEMPLATE},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
               ]
         }
      ]
   }

   headers = {
      "Authorization": f"Bearer {API_KEY}",
      "Content-Type": "application/json"
   }

   try:
      response = requests.post(API_URL, headers=headers, json=payload, timeout = api_timeout)

      if response.status_code == 200:
         try:
               description = json.loads(response.json()["choices"][0]["message"]["content"])["description"]
               return description
         except (KeyError, json.JSONDecodeError, TypeError) as e:
               print(f"[ERROR @ {now()}] Error parsing GPT response: {e}")
      else:
         print(f"[ERROR @ {now()}] GPT API error: {response.status_code}\n{response.text}")

   except requests.exceptions.Timeout:
      print(f"[ERROR @ {now()}] GPT API request timed out.")

   # Fallback: Pick a random greeting
   fallback = random.choice(prefetched_greetings)
   print(f"[WARNING @ {now()}] Using fallback greeting: {fallback}")
   return fallback

def speak(text):
   """Speak the given text using Piper TTS."""
   voice = PiperVoice.load(PIPER_MODEL_PATH)
   print(f"[TTS @ {now()}] {text}")
   stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
   stream.start()
   for audio_bytes in voice.synthesize_stream_raw(text):
      stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
   stream.stop()
   stream.close()


def display_description(description):
   img = np.ones((200, 1920, 3), dtype=np.uint8) * 255  # White background

   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = 1.5
   thickness = 2
   text_color = (0, 0, 255)

   max_width = img.shape[1] - 100  # 50px padding on both sides
   words = description.split(' ')
   lines = []
   current_line = ''

   # Wrap words into lines that fit max_width
   for word in words:
      test_line = current_line + ' ' + word if current_line else word
      (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
      if line_width <= max_width:
         current_line = test_line
      else:
         lines.append(current_line)
         current_line = word
   if current_line:
      lines.append(current_line)

   # Calculate vertical positioning
   total_height = len(lines) * (cv2.getTextSize('Test', font, font_scale, thickness)[0][1] + 20)
   start_y = (img.shape[0] - total_height) // 2 + cv2.getTextSize('Test', font, font_scale, thickness)[0][1]

   # Render each line
   for i, line in enumerate(lines):
      (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
      text_x = (img.shape[1] - line_width) // 2
      text_y = start_y + i * (line_height + 20)
      cv2.putText(img, line, (text_x, text_y), font, font_scale, text_color, thickness)

   # Show the image in a new window
   cv2.namedWindow('Greeting', cv2.WINDOW_NORMAL)
   cv2.imshow('Greeting', img)
   cv2.waitKey(500)
   cv2.destroyWindow('Greeting')


def describe_and_greet(image_path, vistor):
   """Generate a description and speak it."""
   description = generate_description(image_path, vistor)
   if description:
      display_description(description)
      speak(description)
   return description
