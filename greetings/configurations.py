from django.conf import settings

# YOLO model
YOLO_MODEL_PATH = "yolov10n.pt"
WIDTH_THRESHOLD = 1400
HEIGHT_THRESHOLD = 1000
BLUR_THRESHOLD = 20
CONFIDENCE_THRESHOLD = 0.89

# OpenAI API
API_KEY = settings.OPENAI_API_KEY
API_URL = "https://api.openai.com/v1/chat/completions"

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

# Piper TTS
PIPER_MODEL_PATH = "piper-voices/en/ljspeech/medium/en_US-ljspeech-medium.onnx"

# Image preprocessing
IMAGE_RESOLUTION = 1024
