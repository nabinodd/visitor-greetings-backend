import time

from django.conf import settings

# YOLO model
YOLO_MODEL_PATH = "yolov10n.pt"
GPU_ACCELERATION = settings.GPU_ACCELERATION

WIDTH_THRESHOLD = 200
HEIGHT_THRESHOLD = 500

# 🔥 Adjustable Safe Zone Parameters
Y_DOWN = 0.1     # 10% from top
BOX_HEIGHT = 0.3  # 30% of frame height
BOX_WIDTH = 0.5   # 50% of frame width

PERSON_BLUR_THRESHOLD = 50
FACE_BLUR_THRESHOLD = 10

DNN_FACE_DETECTION_CONFIDENCE = 0.9
YOLO_PERSON_CONFIDENCE_THRESHOLD = 0.89

ENABLE_SIZE_REPORTING = False

FONT_SCALE=0.5
FONT_THICKNESS=1

OVERLAY_ONLY_TIME=10

GUSSAIN_BLUR_KERNEL_SIZE=(11, 11)

# OpenAI API
API_KEY = settings.OPENAI_API_KEY
API_URL = "https://api.openai.com/v1/chat/completions"
API_TIMEOUT = 5

DEFAULT_PAYLOAD = {
   "model": "gpt-4.1",
   "temperature": 0,
   "response_format": {
      "type": "json_schema",
      "json_schema": {
         "name": "outfit_description",
         "schema": {
               "type": "object",
               "properties": {
                  "description": {
                     "type": "string",
                     "description": "Flattering outfit description and welcome message"
                  }
               },
               "required": ["description"],
               "additionalProperties": False
         },
         "strict": True
      }
   }
}

SYSTEM_PROMPT = (
   '''
   You are an enthusiastic greeter at U-S-Y-C 2025. 
   Greet each guest by looking at their outfit, appearance and expression in a stylish and dramatic way.
   Always end with greeting words.
   Be warm, flattering, and short in two sentences.
   '''
)

USER_PROMPT_TEMPLATE = (
   '''
   Greet the person in the image warmly. 
   Start with an enthusiastic greeting like "Hey there", "Hello", "Welcome", etc.
   Describe their outfit stylishly as they enter the U-S-Y-C 2025 entrance.
   '''
)


PIPER_MODEL_PATH = "piper-voices/en/ljspeech/medium/en_US-ljspeech-medium.onnx"

IMAGE_RESOLUTION = 1024

def now():
   return int(time.time())