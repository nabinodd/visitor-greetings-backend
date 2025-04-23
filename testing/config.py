from django.conf import settings

api_key = settings.OPENAI_API_KEY

api_url = "https://api.openai.com/v1/chat/completions"

# Model and output format
default_payload = {
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
                        "description": "A detailed description of the person's outfit including items, style, and colors."
                     }
                  },
                  "required": ["description"],
                  "additionalProperties": False
            },
            "strict": True
         }
      }
}

# Prompts
system_prompt = (
   '''
   You are an enthusiastic event organizer at the entrance gate of U-S-Y-C 2025.
   Your job is to warmly welcome each guest as they arrive by looking at their outfit and 
   appearence in a flattering, stylish, and engaging way. Maintain a positive, inclusive tone and 
   add a short greeting at the end.
   '''
)

user_prompt = (
   '''
   Welcome this person as if they are arriving at the entrance 
   to U-S-Y-C 2025. Include a warm and stylish greeting. 
   Make it shortest possibe and sound like you are excited with preposition like hey, wow, etc
   '''
)

# Image settings
image_resolution = 1024
