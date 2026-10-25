import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "nvidia/nemotron-3.5-super-vl:free",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'? also tell me what llm model you are"
        }
      ],
    "reasoning": {"enabled": True}
  })
)

# Extract the assistant message with reasoning_details
response = response.json()
response = response['choices'][0]['message']

print(json.dumps(response, indent=4))

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.get('content'),
    "reasoning_details": response.get('reasoning_details')  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
    headers={
    "Authorization": f"Bearer {os.environ.get('OPENROTER_API_KEY')}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "nvidia/nemotron-3.5-super-vl:free",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True}
  })
)

print('\n\n\n')


response2 = response2.json()

with open("resul2.json", "w", encoding="utf-8") as file:
    json.dump(response2, file, indent=4)

print(json.dumps(response2, indent=4))

# response2 = response2['choices'][0]['message']

