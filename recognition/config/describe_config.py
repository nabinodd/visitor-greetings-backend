from django.conf import settings

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
   Greet each guest by looking at their outfit, appearence and expression in a stylish and dramatic way.
   Always end with greeting words.
   Be warm, flattering, and short in two sentence. 
   '''
)

USER_PROMPT_TEMPLATE = (
   '''
   Name of the person is {name}, greet them with their first name 
   and welcome to them the U-S-Y-C 2025 entrance! Describe their outfit stylishly.
   '''
)

PIPER_MODEL_PATH = "piper-voices/en/ljspeech/medium/en_US-ljspeech-medium.onnx"

IMAGE_RESOLUTION = 1024
